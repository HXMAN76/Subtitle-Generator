# 🎬 Subtitle Generator & Translator

A production-ready, offline subtitle generation and translation application using **Whisper** for speech-to-text, **Silero VAD** for voice activity detection, and a custom-trained neural translation model.

---

## ✨ Features

- **Audio Extraction** - Extract audio from video files (MP4, AVI, MKV, MOV, WebM, FLV)
- **Voice Activity Detection** - Detect speech segments using Silero VAD
- **Speech-to-Text** - Transcribe audio using OpenAI Whisper
- **Custom Translation** - Translate subtitles using a custom-trained neural model (no cloud APIs)
- **Subtitle Generation** - Generate SRT and VTT subtitle files
- **Offline Operation** - Runs completely locally with no internet connection required
- **Production Structure** - Clean, modular, and maintainable codebase

---

## 📁 Project Structure

```
Subtitle-Generator/
├── app.py                          # Main application entry point
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── README.md                       # Documentation (this file)
│
├── src/                            # Source code modules
│   ├── __init__.py
│   ├── audio_processor.py          # Audio extraction and segmentation
│   ├── vad.py                      # Voice activity detection
│   ├── transcriber.py              # Whisper transcription
│   ├── translator.py               # Custom translation model
│   └── subtitle_generator.py       # SRT/VTT generation
│
├── models/                         # Trained models
│   └── translation/                # Translation model files
│       ├── model.pt
│       └── vocab.json
│
├── data/                           # Training data
│   ├── raw/                        # Raw translation datasets
│   └── processed/                  # Processed training data
│       └── train_data.json
│
├── scripts/                        # Utility scripts
│   ├── train_translator.py         # Train translation model
│   └── download_dataset.py         # Download training data
│
├── notebooks/                      # Jupyter notebooks
│   └── data_exploration.ipynb      # Data exploration notebook
│
├── tests/                          # Test files
│   ├── __init__.py
│   ├── test_app.py                 # Application tests
│   └── test_utils.py               # Test utilities
│
├── examples/                       # Example files
│   └── sample_video.mp4            # Sample video for testing
│
├── output/                         # Generated subtitle files
└── temp/                           # Temporary files
    └── voice/                      # Segmented audio files
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **FFmpeg** (for audio processing)

### 1. Install FFmpeg

| Platform | Command |
|----------|---------|
| **Linux** | `sudo apt-get install ffmpeg` |
| **macOS** | `brew install ffmpeg` |
| **Windows** | Download from [ffmpeg.org](https://ffmpeg.org/download.html) |

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python app.py
```

> **Note:** Update the `video_path` in `app.py` to point to your video file before running.

### 4. Output

Subtitles will be generated in the `output/` folder:
- `video_name_original.srt` — Original transcription
- `video_name_es.srt` — Translated version (if translation model is trained)

---

## ⚙️ Configuration

Edit `config.py` to customize settings:

```python
# Whisper settings
WHISPER_MODEL_SIZE = "tiny"     # Options: tiny, base, small, medium, large
WHISPER_DEVICE = "cpu"          # Options: cpu, cuda

# Languages
SOURCE_LANGUAGE = "en"
TARGET_LANGUAGE = "es"          # Change as needed

# Subtitle settings
SUBTITLE_FORMAT = "srt"         # Options: srt, vtt

# VAD sensitivity
VAD_THRESHOLD = 0.5             # Range: 0.0 to 1.0
```

### Whisper Model Comparison

| Model | Speed | Accuracy | Memory | Use Case |
|-------|-------|----------|--------|----------|
| `tiny` | ⚡ Fastest | Low | ~1 GB | Testing |
| `base` | Fast | Good | ~1.5 GB | **Production** |
| `small` | Medium | Better | ~2 GB | Quality |
| `medium` | Slow | High | ~4 GB | Accuracy |
| `large` | Slowest | Highest | ~8 GB | Best quality |

---

## 📖 Advanced Usage

### Programmatic API

```python
from src.audio_processor import AudioProcessor
from src.vad import VoiceActivityDetector
from src.transcriber import Transcriber
from src.translator import Translator
from src.subtitle_generator import SubtitleGenerator

# Initialize components
audio_processor = AudioProcessor()
vad = VoiceActivityDetector()
transcriber = Transcriber()
translator = Translator()
subtitle_gen = SubtitleGenerator()

# Process video
audio_path = audio_processor.convert_video_to_audio("video.mp4")
speech_timestamps = vad.detect_speech(audio_path)
segments = audio_processor.segment_audio(audio_path, speech_timestamps)
transcriptions = transcriber.transcribe_segments(segments)

# Generate subtitles
subtitle_gen.generate_subtitles(transcriptions, "output", format="srt")

# Translate (if model is trained)
translated = translator.translate_subtitles(transcriptions)
subtitle_gen.generate_subtitles(translated, "output_translated", format="srt")
```

### Using the SubtitleApp Class

```python
from app import SubtitleApp

app = SubtitleApp()
results = app.process_video(
    video_path="your_video.mp4",
    translate=True  # Enable translation
)
```

### Batch Processing

```python
import glob
from app import SubtitleApp

app = SubtitleApp()

# Process all MP4 files
for video in glob.glob("*.mp4"):
    print(f"Processing {video}...")
    app.process_video(video, translate=False)
```

---

## 🧠 Training the Translation Model

### 1. Prepare Training Data

Create a JSON file with parallel sentences at `data/processed/train_data.json`:

```json
[
  {"source": "Hello world", "target": "Hola mundo"},
  {"source": "Good morning", "target": "Buenos días"},
  {"source": "How are you?", "target": "¿Cómo estás?"}
]
```

**Get Training Data:**
- Download parallel corpora from [OPUS](https://opus.nlpl.eu/)
- Use [Tatoeba](https://tatoeba.org/) for sentence pairs
- Create your own dataset

### 2. Train the Model

```bash
python scripts/train_translator.py
```

Training time depends on dataset size (typically 10-30 minutes for 10K sentence pairs).

### 3. Model Architecture

- **Encoder:** Bidirectional LSTM
- **Decoder:** LSTM with attention
- **Embeddings:** 256 dimensions
- **Hidden:** 512 dimensions
- **Layers:** 2 LSTM layers

The trained model will be saved to `models/translation/`.

---

## 📦 Building as Executable (.exe)

### Quick Build

```bash
pip install pyinstaller
pyinstaller --onefile --name SubtitleGenerator --add-data "src:src" --add-data "config.py:." --add-data "models:models" --hidden-import=whisper --hidden-import=torch app.py
```

### Using Spec File

Create `subtitle_generator.spec`:

```python
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('src', 'src'),
        ('models', 'models'),
        ('config.py', '.'),
    ],
    hiddenimports=[
        'whisper',
        'torch',
        'moviepy',
        'pydub',
        'numpy',
        'scipy',
        'tiktoken',
        'regex',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SubtitleGenerator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon='icon.ico'  # Optional
)
```

Then build:

```bash
pyinstaller subtitle_generator.spec
```

### Distribution Package

```
SubtitleGenerator/
├── SubtitleGenerator.exe       # Main executable
├── ffmpeg.exe                  # Required
├── ffprobe.exe                 # Required
├── models/                     # Model files
│   └── translation/
│       ├── model.pt
│       └── vocab.json
├── output/                     # Empty folder
├── temp/                       # Empty folder
└── README.txt                  # Usage instructions
```

### Optimizations

**Reduce File Size:**
- Use smaller Whisper model (tiny or base)
- Install CPU-only PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Enable UPX compression

**Pre-download Models:**
```python
import whisper
import torch

whisper.load_model("tiny")  # Downloads once
model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| **FFmpeg not found** | Install FFmpeg and ensure it's in your PATH |
| **Out of memory** | Use a smaller Whisper model (tiny or base) |
| **CUDA error** | Set `WHISPER_DEVICE = "cpu"` in `config.py` |
| **Translation returns original text** | Train the translation model first |
| **Poor transcription quality** | Use larger Whisper model or improve audio quality |

---

## 🎯 Performance Tips

1. **Use GPU** - If you have CUDA, set `WHISPER_DEVICE = "cuda"` in config
2. **Smaller Models** - Use "tiny" for quick testing, "base" for production
3. **VAD Tuning** - Adjust `VAD_THRESHOLD` (0.3-0.7) based on audio quality
4. **Batch Processing** - Process multiple videos in sequence to reuse loaded models

### Benchmark (1-minute video, Intel i7, 16GB RAM)

| Model | Time |
|-------|------|
| tiny | ~5 seconds |
| base | ~15 seconds |
| small | ~30 seconds |
| medium | ~60 seconds |
| large | ~2-3 minutes |

---

## 🔧 Dependencies

### Core
- **moviepy** - Video/audio processing
- **openai-whisper** - Speech-to-text transcription
- **torch** - Neural network framework
- **pydub** - Audio manipulation
- **silero-vad** - Voice activity detection

### System
- **FFmpeg** - Required for audio processing
- **Python 3.8+** - Development

---

## 📊 Supported Formats

### Video Input
- MP4 (recommended)
- AVI, MKV, MOV
- WebM, FLV

### Subtitle Output
- SRT (SubRip)
- VTT (WebVTT)

---

## 🎯 Roadmap

- [ ] GUI interface
- [ ] Batch processing CLI
- [ ] Multiple language support
- [ ] Real-time subtitling
- [ ] Custom model fine-tuning interface
- [ ] Subtitle editing capabilities

---

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

---

## 📝 License

MIT License - feel free to use for personal or commercial projects.

---

## 📞 Support

For issues or questions, please open an issue on GitHub.

---

**Note:** This application runs completely offline. Initial setup requires internet to download Whisper models and dependencies, but afterward, it works without any cloud connections.
