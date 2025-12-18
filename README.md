# Subtitle Generator and Translator

A production-ready, offline subtitle generation and translation application using Whisper for speech-to-text, Silero VAD for voice activity detection, and a custom-trained neural translation model.

## 🎯 Features

- **Audio Extraction**: Extract audio from video files
- **Voice Activity Detection**: Detect speech segments using Silero VAD
- **Speech-to-Text**: Transcribe audio using OpenAI Whisper
- **Custom Translation**: Translate subtitles using a custom-trained neural model (no cloud APIs)
- **Subtitle Generation**: Generate SRT and VTT subtitle files
- **Offline Operation**: Runs completely locally with no internet connection required
- **Production Structure**: Clean, modular, and maintainable codebase

## 📁 Project Structure

```
Subtitle-Generator/
├── app.py                      # Main application entry point
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── audio_processor.py      # Audio extraction and segmentation
│   ├── vad.py                  # Voice activity detection
│   ├── transcriber.py          # Whisper transcription
│   ├── translator.py           # Custom translation model
│   └── subtitle_generator.py  # SRT/VTT generation
│
├── models/                     # Trained models
│   └── translation/            # Translation model files
│       ├── model.pt
│       └── vocab.json
│
├── data/                       # Training data
│   ├── raw/                    # Raw translation datasets
│   └── processed/              # Processed training data
│       └── train_data.json
│
├── output/                     # Generated subtitle files
├── temp/                       # Temporary files
│   └── voice/                  # Segmented audio files
│
└── scripts/                    # Utility scripts
    └── train_translator.py     # Train translation model
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio processing)

### Install FFmpeg

**Linux:**
```bash
sudo apt-get install ffmpeg
```

**MacOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

## 🎬 Usage

### Basic Usage

1. Place your video file in the project directory
2. Update the `video_path` in [app.py](app.py) or run directly:

```bash
python app.py
```

### Advanced Usage

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

### 2. Train the Model

```bash
python scripts/train_translator.py
```

The trained model will be saved to `models/translation/`.

### 3. Configuration

Edit [config.py](config.py) to customize:
- Whisper model size (tiny, base, small, medium, large)
- VAD parameters
- Source and target languages
- Subtitle format (SRT or VTT)

## 📦 Building as .exe (Windows)

### Using PyInstaller

1. Install PyInstaller:
```bash
pip install pyinstaller
```

2. Create executable:
```bash
pyinstaller --onefile --name SubtitleGenerator app.py
```

3. The executable will be in the `dist/` folder

### Notes for .exe Distribution

- Include FFmpeg binary in the same directory as the .exe
- Pre-download Whisper models to avoid internet requirement:
  ```python
  import whisper
  whisper.load_model("tiny")  # Downloads once
  ```
- Include the `models/` folder with your translation model

## ⚙️ Configuration Options

Edit [config.py](config.py):

```python
# Whisper settings
WHISPER_MODEL_SIZE = "tiny"  # tiny, base, small, medium, large

# Languages
SOURCE_LANGUAGE = "en"
TARGET_LANGUAGE = "es"

# Subtitle format
SUBTITLE_FORMAT = "srt"  # srt or vtt

# VAD sensitivity
VAD_THRESHOLD = 0.5  # 0.0 to 1.0
```

## 🔧 Dependencies

- **moviepy**: Video/audio processing
- **whisper**: Speech-to-text transcription
- **torch**: Neural network framework
- **pydub**: Audio manipulation
- **silero-vad**: Voice activity detection

## 📝 License

MIT License - feel free to use for personal or commercial projects.

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📞 Support

For issues or questions, please open an issue on GitHub.

## 🎯 Roadmap

- [ ] GUI interface
- [ ] Batch processing
- [ ] Multiple language support
- [ ] Real-time subtitling
- [ ] Custom model fine-tuning interface
- [ ] Subtitle editing capabilities

---

**Note**: This application runs completely offline. Initial setup requires internet to download Whisper models and dependencies, but afterward it works without any cloud connections.
