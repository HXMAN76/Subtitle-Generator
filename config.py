"""Configuration settings for the Subtitle Generator application."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
TEMP_DIR = BASE_DIR / "temp"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Whisper settings
WHISPER_MODEL_SIZE = "tiny"  # Options: tiny, base, small, medium, large
WHISPER_DEVICE = "cpu"  # Options: cpu, cuda

# Silero VAD settings
VAD_THRESHOLD = 0.5
VAD_MIN_SPEECH_DURATION_MS = 250
VAD_MAX_SPEECH_DURATION_S = 30
VAD_MIN_SILENCE_DURATION_MS = 100

# Audio settings
AUDIO_FORMAT = "mp3"
AUDIO_BITRATE = "192k"

# Translation settings
TRANSLATION_MODEL_PATH = MODELS_DIR / "translation" / "model.pt"
TRANSLATION_VOCAB_PATH = MODELS_DIR / "translation" / "vocab.json"
SOURCE_LANGUAGE = "en"
TARGET_LANGUAGE = "es"  # Change as needed

# Subtitle settings
SUBTITLE_FORMAT = "srt"  # Options: srt, vtt
MAX_SUBTITLE_LENGTH = 42  # Characters per line
MAX_SUBTITLE_LINES = 2

# Threading
NUM_THREADS = 1  # For torch operations

# Create directories if they don't exist
os.makedirs(TEMP_DIR / "voice", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR / "translation", exist_ok=True)
os.makedirs(DATA_DIR / "raw", exist_ok=True)
os.makedirs(DATA_DIR / "processed", exist_ok=True)
