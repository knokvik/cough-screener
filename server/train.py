"""
Offline Training Script for Cough Prescreening CNN
Trained on Coswara / Coughvid acoustic recordings or synthetic demo dataset.

Outputs:
  - models/model.pt        : PyTorch state dictionary for CoughCNN
  - models/norm_stats.json : Global dataset mean and standard deviation for inference normalization

Usage:
  # Quick synthetic demo training (produces working model in ~15 seconds without datasets):
  python -m server.train --synthetic-demo --epochs 10

  # Train on extracted Coswara dataset:
  python -m server.train --coswara-dir /path/to/Coswara-Data --epochs 20 --batch-size 32
"""

import argparse
import json
import logging
from pathlib import Path
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchaudio

from server.config import config
from server.model import CoughCNN
from server.pipeline import extract_log_mel_spectrogram

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cough_screener.train")

# Set deterministic seeds
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class AudioSpectrogramDataset(Dataset):
    """PyTorch Dataset holding log-mel spectrogram tensors and class labels."""

    def __init__(
        self,
        spectrograms: List[np.ndarray],
        labels: List[int],
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ):
        self.spectrograms = spectrograms
        self.labels = labels
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.spectrograms)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        spec = self.spectrograms[idx].copy()

        # Apply dataset z-score normalization
        if self.mean is not None and self.std is not None and self.std > 1e-6:
            spec = (spec - self.mean) / self.std
        else:
            s_mean = np.mean(spec)
            s_std = np.std(spec)
            spec = (spec - s_mean) / (s_std + 1e-6)

        # Convert to tensor shape (1, n_mels=64, time_steps=128)
        spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return spec_tensor, label_tensor


def generate_synthetic_audio(label_name: str, duration_sec: float = 1.2, sr: int = 16000) -> np.ndarray:
    """
    Synthesizes acoustic waveforms with spectral profiles simulating the 3 classes:
      - 'healthy': Transient high-velocity air burst with smooth exponential decay (500-2500 Hz).
      - 'cough_suspect': Deeper wheezy cough with irregular harsh harmonics and stridor (800-4500 Hz).
      - 'no_cough': Ambient background noise, speech-like formants, or room reverberation.
    """
    n_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    if label_name == "healthy":
        # Double-burst healthy cough profile
        envelope = np.exp(-t * 8.0) * (np.sin(2 * np.pi * 3.5 * t) ** 2)
        noise = np.random.normal(0, 0.4, n_samples)
        f1 = np.sin(2 * np.pi * 650 * t)
        f2 = 0.5 * np.sin(2 * np.pi * 1400 * t)
        signal = (noise * 0.7 + f1 * 0.2 + f2 * 0.1) * envelope

    elif label_name == "cough_suspect":
        # Harsh, prolonged explosive cough with wheezing stridor harmonics
        envelope = np.exp(-t * 5.0) * (np.sin(2 * np.pi * 2.8 * t) ** 2)
        noise = np.random.normal(0, 0.6, n_samples)
        stridor = 0.35 * np.sin(2 * np.pi * 2200 * t + np.sin(2 * np.pi * 40 * t))
        subharmonics = 0.25 * np.sin(2 * np.pi * 380 * t)
        signal = (noise * 0.6 + stridor + subharmonics) * envelope

    else:  # 'no_cough'
        # Ambient room noise + intermittent speech vowel formant
        noise = np.random.normal(0, 0.08, n_samples)
        formant = 0.15 * np.sin(2 * np.pi * 400 * t) * (1.0 + 0.3 * np.sin(2 * np.pi * 5 * t))
        signal = noise + formant

    # Normalize to 16-bit range
    signal = signal / (np.max(np.abs(signal)) + 1e-6)
    pcm = (signal * 24000).astype(np.int16)
    return pcm


def load_coswara_dataset(coswara_dir: Path) -> Tuple[List[np.ndarray], List[int]]:
    """
    Parses an extracted Coswara-Data directory.
    Coswara metadata: combined_data.csv (id, covid_status, ...)
    Audio subdirectories contain: cough-heavy.wav, cough-shallow.wav, breathing-deep.wav, vowel-a.wav
    """
    import csv

    spectrograms: List[np.ndarray] = []
    labels: List[int] = []

    csv_path = coswara_dir / "combined_data.csv"
    if not csv_path.exists():
        # Search recursively for combined_data.csv
        found = list(coswara_dir.glob("**/combined_data.csv"))
        if found:
            csv_path = found[0]
        else:
            raise FileNotFoundError(f"Could not locate combined_data.csv in {coswara_dir}")

    logger.info(f"Parsing Coswara metadata: {csv_path}")

    covid_status_map: Dict[str, str] = {}
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("id", "").strip()
            status = row.get("covid_status", "").strip().lower()
            if pid and status:
                covid_status_map[pid] = status

    logger.info(f"Found {len(covid_status_map)} participant records in Coswara metadata.")

    # Find audio files
    audio_files = list(coswara_dir.glob("**/*.wav"))
    logger.info(f"Discovered {len(audio_files)} .wav files across Coswara directory.")

    counts = {0: 0, 1: 0, 2: 0}

    for wav_file in audio_files:
        filename = wav_file.name.lower()
        participant_id = wav_file.parent.name

        status = covid_status_map.get(participant_id, "unknown")

        target_class: Optional[int] = None

        if "cough" in filename:
            if status in ["healthy", "resp_ill_not_covid", "no_resp_illness_exposed"]:
                target_class = 0  # healthy
            elif any(s in status for s in ["positive", "confirmed", "mild", "moderate"]):
                target_class = 1  # cough_suspect
        elif any(k in filename for k in ["vowel", "breathing", "counting"]):
            target_class = 2  # no_cough (respiratory non-cough sounds)

        if target_class is None:
            continue

        try:
            # Load audio using torchaudio or scipy
            waveform, sr = torchaudio.load(str(wav_file))
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != config.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, config.SAMPLE_RATE)
                waveform = resampler(waveform)

            audio_pcm = (waveform.squeeze(0).numpy() * 32767).astype(np.int16)
            spec = extract_log_mel_spectrogram(
                audio_pcm,
                sample_rate=config.SAMPLE_RATE,
                n_mels=config.N_MELS,
                n_fft=config.N_FFT,
                hop_length=config.HOP_LENGTH,
                target_frames=config.TARGET_SPEC_FRAMES,
            )

            spectrograms.append(spec)
            labels.append(target_class)
            counts[target_class] += 1

        except Exception as e:
            logger.debug(f"Error loading {wav_file}: {e}")

    # Check for empty classes and augment if necessary
    for c_idx, c_name in enumerate(["healthy", "cough_suspect", "no_cough"]):
        if counts[c_idx] < 10:
            deficit = max(25, counts[0] // 2 if counts[0] > 0 else 30) - counts[c_idx]
            logger.info(f"Class '{c_name}' has few real samples ({counts[c_idx]}). Synthesizing {deficit} samples to balance 3-class dataset.")
            for _ in range(deficit):
                synth_audio = generate_synthetic_audio(c_name, duration_sec=random.uniform(0.8, 1.8), sr=config.SAMPLE_RATE)
                spec = extract_log_mel_spectrogram(
                    synth_audio,
                    sample_rate=config.SAMPLE_RATE,
                    n_mels=config.N_MELS,
                    n_fft=config.N_FFT,
                    hop_length=config.HOP_LENGTH,
                    target_frames=config.TARGET_SPEC_FRAMES,
                )
                spectrograms.append(spec)
                labels.append(c_idx)
                counts[c_idx] += 1

    logger.info(f"Final loaded dataset distribution: healthy={counts[0]}, cough_suspect={counts[1]}, no_cough={counts[2]}")
    return spectrograms, labels


def generate_synthetic_dataset(num_per_class: int = 150) -> Tuple[List[np.ndarray], List[int]]:
    """Generates synthetic training set for immediate verification."""
    logger.info(f"Generating synthetic training dataset ({num_per_class} samples per class)...")
    spectrograms: List[np.ndarray] = []
    labels: List[int] = []

    class_names = ["healthy", "cough_suspect", "no_cough"]
    for class_idx, name in enumerate(class_names):
        for _ in range(num_per_class):
            duration = random.uniform(0.6, 1.8)
            audio = generate_synthetic_audio(name, duration_sec=duration, sr=config.SAMPLE_RATE)
            spec = extract_log_mel_spectrogram(
                audio,
                sample_rate=config.SAMPLE_RATE,
                n_mels=config.N_MELS,
                n_fft=config.N_FFT,
                hop_length=config.HOP_LENGTH,
                target_frames=config.TARGET_SPEC_FRAMES,
            )
            spectrograms.append(spec)
            labels.append(class_idx)

    return spectrograms, labels


def train_model(
    spectrograms: List[np.ndarray],
    labels: List[int],
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    output_model_path: Path = config.MODEL_PATH,
    output_norm_path: Path = config.NORM_STATS_PATH,
):
    """Executes model training, validation, metric reporting, and artifact export."""
    total_samples = len(spectrograms)
    if total_samples < 10:
        raise ValueError("Insufficient dataset size to train model.")

    # Stratified Train/Val split (80% train, 20% validation)
    indices_by_class: Dict[int, List[int]] = {0: [], 1: [], 2: []}
    for idx, l in enumerate(labels):
        indices_by_class[l].append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []

    for c, idxs in indices_by_class.items():
        random.shuffle(idxs)
        split = int(0.8 * len(idxs))
        train_indices.extend(idxs[:split])
        val_indices.extend(idxs[split:])

    random.shuffle(train_indices)
    random.shuffle(val_indices)

    train_specs = [spectrograms[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_specs = [spectrograms[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    # Compute global dataset normalization stats (mean and std across all train pixels)
    stacked_train = np.stack(train_specs)
    dataset_mean = float(np.mean(stacked_train))
    dataset_std = float(np.std(stacked_train))

    logger.info(f"Dataset Normalization Stats: Mean={dataset_mean:.4f}, Std={dataset_std:.4f}")

    # Save normalization stats
    output_norm_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_norm_path, "w") as f:
        json.dump({"mean": dataset_mean, "std": dataset_std}, f, indent=2)
    logger.info(f"Saved feature normalization statistics to {output_norm_path}")

    # Build PyTorch Datasets & DataLoaders
    train_dataset = AudioSpectrogramDataset(train_specs, train_labels, mean=dataset_mean, std=dataset_std)
    val_dataset = AudioSpectrogramDataset(val_specs, val_labels, mean=dataset_mean, std=dataset_std)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Class weighting for balanced loss
    class_counts = [len(indices_by_class[c]) for c in range(3)]
    total_c = sum(class_counts)
    class_weights = [total_c / (3.0 * max(1, count)) for count in class_counts]
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    logger.info(f"Training device: {device} | Class weights: {[round(w, 2) for w in class_weights]}")

    model = CoughCNN(num_classes=3, dropout_rate=0.25).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_acc = 0.0
    best_state_dict = None

    logger.info("=====================================================================")
    logger.info(f"  Starting Training: {epochs} Epochs, {len(train_dataset)} Train, {len(val_dataset)} Val")
    logger.info("=====================================================================")

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == y_batch).sum().item()
            total_train += len(y_batch)

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = correct_train / total_train

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(x_batch)
                loss = criterion(logits, y_batch)

                val_loss += loss.item() * len(y_batch)
                preds = torch.argmax(logits, dim=1)
                correct_val += (preds == y_batch).sum().item()
                total_val += len(y_batch)

                all_val_preds.extend(preds.cpu().numpy().tolist())
                all_val_targets.extend(y_batch.cpu().numpy().tolist())

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        scheduler.step(epoch_val_acc)

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.1f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.1f}%"
        )

        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            best_state_dict = model.state_dict().copy()

    # Save best model
    if best_state_dict is not None:
        output_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state_dict, output_model_path)
        logger.info(f"Saved best model weights (Val Acc: {best_val_acc * 100:.1f}%) to {output_model_path}")

    # Compute final metrics & Confusion Matrix
    print_evaluation_report(all_val_targets, all_val_preds)


def print_evaluation_report(y_true: List[int], y_pred: List[int]):
    """Calculates and prints precision, recall, F1, and an ASCII confusion matrix."""
    from sklearn.metrics import classification_report, confusion_matrix

    target_names = ["healthy", "cough_suspect", "no_cough"]
    print("\n" + "=" * 65)
    print("                FINAL MODEL EVALUATION REPORT")
    print("=" * 65)
    print(classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=target_names, digits=3, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    print("-" * 65)
    print("CONFUSION MATRIX:")
    print("                 Predicted: healthy   cough_suspect   no_cough")
    print(f"True: healthy            [   {cm[0][0]:<4}         {cm[0][1]:<4}         {cm[0][2]:<4} ]")
    print(f"True: cough_suspect      [   {cm[1][0]:<4}         {cm[1][1]:<4}         {cm[1][2]:<4} ]")
    print(f"True: no_cough           [   {cm[2][0]:<4}         {cm[2][1]:<4}         {cm[2][2]:<4} ]")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Train Cough Prescreening CNN Model")
    parser.add_argument("--coswara-dir", type=str, default=None, help="Path to extracted Coswara dataset directory")
    parser.add_argument("--synthetic-demo", action="store_true", help="Generate synthetic acoustic dataset for fast test")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs (default: 12)")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument("--output-model", type=str, default=str(config.MODEL_PATH), help="Destination model path")
    parser.add_argument("--output-norm", type=str, default=str(config.NORM_STATS_PATH), help="Destination norm stats path")

    args = parser.parse_args()

    if args.coswara_dir:
        coswara_path = Path(args.coswara_dir)
        if not coswara_path.exists():
            logger.error(f"Provided Coswara path does not exist: {coswara_path}")
            sys.exit(1)
        spectrograms, labels = load_coswara_dataset(coswara_path)
    elif args.synthetic_demo:
        spectrograms, labels = generate_synthetic_dataset(num_per_class=150)
    else:
        logger.warning(
            "Neither --coswara-dir nor --synthetic-demo provided. "
            "Defaulting to --synthetic-demo so you can verify the ML pipeline immediately!"
        )
        spectrograms, labels = generate_synthetic_dataset(num_per_class=150)

    train_model(
        spectrograms,
        labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_model_path=Path(args.output_model),
        output_norm_path=Path(args.output_norm),
    )


if __name__ == "__main__":
    main()
