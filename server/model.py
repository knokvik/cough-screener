"""
PyTorch CNN Architecture and Classifier for Cough Prescreening
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from server.config import config

logger = logging.getLogger("cough_screener.model")


class CoughCNN(nn.Module):
    """
    Compact 2D Convolutional Neural Network for Log-Mel Spectrogram Classification.
    Input shape: (batch_size, 1, n_mels=64, time_steps=128)
    Output shape: (batch_size, 3) [healthy, cough_suspect, no_cough]
    """

    def __init__(self, num_classes: int = 3, dropout_rate: float = 0.25):
        super().__init__()

        # Conv Block 1: 1 -> 32 filters
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (64, 128) -> (32, 64)
            nn.Dropout2d(dropout_rate * 0.6),
        )

        # Conv Block 2: 32 -> 64 filters
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (32, 64) -> (16, 32)
            nn.Dropout2d(dropout_rate * 0.8),
        )

        # Conv Block 3: 64 -> 128 filters
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (16, 32) -> (8, 16)
            nn.Dropout2d(dropout_rate),
        )

        # Conv Block 4: 128 -> 128 filters + Global Average Pooling
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # (8, 16) -> (1, 1)
        )

        # Fully connected classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (B, 1, 64, 128)
        Returns:
            Logits of shape (B, 3)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        logits = self.classifier(x)
        return logits


class DummyCoughClassifier:
    """
    Fallback classifier used when models/model.pt is not yet trained.
    Analyzes basic spectral energy distribution and returns realistic pseudo-probabilities
    so the entire real-time pipeline (ESP32 -> Server -> UI) can be tested immediately.
    """

    def __init__(self):
        self.is_dummy = True
        logger.warning(
            "[MODEL NOTICE] Running in DUMMY MODEL MODE (models/model.pt not found). "
            "Simulating classification for end-to-end pipeline testing. "
            "To train a real model, run: python -m server.train"
        )

    def predict(self, mel_spec: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        mel_spec: shape (64, 128)
        Returns: (label, confidence, prob_dict)
        """
        # Distinguish cough vs noise via spectral contrast between mid-high frequencies
        # Coughs typically have intense burst energy in the 500 Hz - 4000 Hz range
        low_band = np.mean(mel_spec[:16, :])    # ~50-800 Hz
        mid_band = np.mean(mel_spec[16:48, :])  # ~800-4000 Hz
        high_band = np.mean(mel_spec[48:, :])   # ~4000-8000 Hz

        # Burstiness: variance along time axis
        time_variance = np.var(np.mean(mel_spec, axis=0))

        # Synthetic logits based on acoustic features + subtle pseudo-random variation
        rng = np.random.RandomState(int(abs(time_variance * 10000)) % (2**31 - 1))
        
        if time_variance > 0.05 and mid_band > low_band * 0.8:
            # Looks like a cough acoustic event
            suspect_score = 1.8 + rng.uniform(0.0, 1.2)
            healthy_score = 1.5 + rng.uniform(0.0, 1.2)
            noise_score = 0.2 + rng.uniform(0.0, 0.4)
        else:
            # Ambient or non-cough sound
            noise_score = 2.2 + rng.uniform(0.5, 1.0)
            healthy_score = 0.5 + rng.uniform(0.0, 0.5)
            suspect_score = 0.3 + rng.uniform(0.0, 0.4)

        raw_scores = np.array([healthy_score, suspect_score, noise_score])
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        probs = exp_scores / np.sum(exp_scores)

        classes = config.CLASSES
        top_idx = int(np.argmax(probs))
        top_label = classes[top_idx]
        confidence = float(probs[top_idx])

        prob_dict = {
            classes[0]: round(float(probs[0]), 4),
            classes[1]: round(float(probs[1]), 4),
            classes[2]: round(float(probs[2]), 4),
        }

        return top_label, confidence, prob_dict


class CoughClassifierEngine:
    """
    Inference Engine: Handles model loading, norm stats loading,
    tensor conversion, and prediction with fallback to Dummy classifier.
    """

    def __init__(self, model_path: Optional[Path] = None, norm_stats_path: Optional[Path] = None):
        self.model_path = model_path or config.MODEL_PATH
        self.norm_stats_path = norm_stats_path or config.NORM_STATS_PATH
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        
        self.model: Optional[nn.Module] = None
        self.dummy_model: Optional[DummyCoughClassifier] = None
        self.is_dummy = False
        
        self.mean: float = 0.0
        self.std: float = 1.0
        self.has_norm_stats: bool = False

        self._load_norm_stats()
        self._load_model()

    def _load_norm_stats(self):
        """Load precomputed dataset mean and std for feature z-scoring."""
        if self.norm_stats_path.exists():
            try:
                with open(self.norm_stats_path, "r") as f:
                    data = json.load(f)
                    self.mean = float(data.get("mean", 0.0))
                    self.std = float(data.get("std", 1.0))
                    self.has_norm_stats = True
                    logger.info(f"[MODEL] Loaded normalization stats from {self.norm_stats_path}: mean={self.mean:.4f}, std={self.std:.4f}")
            except Exception as e:
                logger.warning(f"[MODEL] Could not read norm stats ({e}); using per-sample z-score.")
        else:
            logger.info("[MODEL] No norm_stats.json found; per-sample z-score normalization will be used.")

    def _load_model(self):
        """Loads trained weights into CoughCNN, or initializes DummyCoughClassifier."""
        if self.model_path.exists():
            try:
                logger.info(f"[MODEL] Loading PyTorch model from {self.model_path} onto {self.device}...")
                model = CoughCNN(num_classes=len(config.CLASSES))
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.model = model
                self.is_dummy = False
                logger.info("[MODEL] Successfully loaded trained weights into CoughCNN.")
            except Exception as e:
                logger.error(f"[MODEL] Failed loading model weights from {self.model_path}: {e}")
                self.dummy_model = DummyCoughClassifier()
                self.is_dummy = True
        else:
            self.dummy_model = DummyCoughClassifier()
            self.is_dummy = True

    def predict(self, mel_spec: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Runs inference on a log-mel spectrogram matrix.
        mel_spec: shape (64, 128) float array
        Returns: (top_label, confidence, prob_dict)
        """
        if self.is_dummy or self.model is None:
            return self.dummy_model.predict(mel_spec)

        # Normalize features
        if self.has_norm_stats and self.std > 1e-6:
            norm_spec = (mel_spec - self.mean) / self.std
        else:
            s_mean = np.mean(mel_spec)
            s_std = np.std(mel_spec)
            norm_spec = (mel_spec - s_mean) / (s_std + 1e-6)

        # Convert to Tensor (1, 1, 64, 128)
        tensor_in = torch.tensor(norm_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_in)
            probs_t = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        classes = config.CLASSES
        top_idx = int(np.argmax(probs_t))
        top_label = classes[top_idx]
        confidence = float(probs_t[top_idx])

        prob_dict = {
            classes[0]: round(float(probs_t[0]), 4),
            classes[1]: round(float(probs_t[1]), 4),
            classes[2]: round(float(probs_t[2]), 4),
        }

        return top_label, confidence, prob_dict
