"""
Configuration for Cough Prescreening Server and Audio Pipeline
"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ServerConfig:
    # Network settings
    HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("SERVER_PORT", "8000"))

    # Audio stream parameters
    SAMPLE_RATE: int = 16000          # 16 kHz audio
    SAMPLE_WIDTH: int = 2             # 16-bit PCM = 2 bytes
    CHANNELS: int = 1                 # Mono
    CHUNK_SAMPLES: int = 1024         # ESP32 sends 1024-sample frames
    RING_BUFFER_SECONDS: float = 3.0  # Rolling 3-second buffer (48,000 samples)
    ANALYSIS_INTERVAL_SEC: float = 0.1 # Run audio VAD analysis every 100 ms

    # Sound Event Detection (Adaptive RMS + MAD)
    NOISE_FLOOR_HISTORY_LEN: int = 100 # Last 10 seconds of 100ms frames
    MAD_MULTIPLIER: float = 3.0       # Threshold = median + 3.0 * MAD
    MIN_RMS_THRESHOLD: float = 120.0  # Minimum 16-bit RMS threshold (avoids dead-room triggers)
    SILENCE_HANGOVER_SEC: float = 0.5 # Silence duration to close event
    MIN_EVENT_DURATION_SEC: float = 0.1 # Discard short clicks/pops (<100 ms)
    MAX_EVENT_DURATION_SEC: float = 3.0 # Discard long droning sounds (>3.0 s)
    PRE_EVENT_PADDING_SEC: float = 0.15 # Capture initial attack transient

    # Mel-Spectrogram Feature Extraction
    N_MELS: int = 64                  # 64 mel filterbanks
    N_FFT: int = 512                  # 512-point FFT (32 ms window)
    HOP_LENGTH: int = 256             # 256-point hop (16 ms step)
    F_MIN: float = 50.0               # Lower frequency cutoff
    F_MAX: float = 8000.0             # Nyquist frequency for 16 kHz
    TARGET_SPEC_FRAMES: int = 128     # Fixed time frames for CNN input (~2.05 s)

    # Classification & Model
    CLASSES: tuple = ("healthy", "cough_suspect", "no_cough")
    CLASS_LABELS: tuple = ("Healthy", "Cough Suspect", "No Cough / Noise")

    # File paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODEL_PATH: Path = BASE_DIR / "models" / "model.pt"
    NORM_STATS_PATH: Path = BASE_DIR / "models" / "norm_stats.json"
    EVENTS_DIR: Path = BASE_DIR / "server" / "events"
    STATIC_DIR: Path = BASE_DIR / "server" / "static"


config = ServerConfig()
