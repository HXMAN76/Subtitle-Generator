# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Install FFmpeg** (if not already installed):
- **Linux**: `sudo apt-get install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### 2. Basic Usage

Place your video file in the project directory and run:

```bash
python app.py
```

**Edit the video path** in [app.py](app.py#L90) before running:
```python
video_path = "./your_video.mp4"  # Change this!
```

### 3. Output

Subtitles will be generated in the `output/` folder:
- `video_name_original.srt` - Original transcription
- `video_name_es.srt` - Translated version (if translation model is trained)

## 📝 Configuration

Edit [config.py](config.py) to customize:

### Change Whisper Model Size
```python
WHISPER_MODEL_SIZE = "tiny"  # Options: tiny, base, small, medium, large
```

**Model Comparison:**
- `tiny`: Fastest, lowest accuracy (~10x faster than base)
- `base`: Good balance of speed and accuracy
- `small`: Better accuracy, slower
- `medium`: High accuracy, requires more RAM
- `large`: Best accuracy, slowest, requires GPU

### Change Languages
```python
SOURCE_LANGUAGE = "en"  # Source language (e.g., "en", "es", "fr")
TARGET_LANGUAGE = "es"  # Target language for translation
```

### Change Subtitle Format
```python
SUBTITLE_FORMAT = "srt"  # Options: "srt" or "vtt"
```

## 🧠 Training Translation Model (Optional)

### Step 1: Prepare Data

Create `data/processed/train_data.json` with parallel sentences:
```json
[
  {"source": "Hello", "target": "Hola"},
  {"source": "Goodbye", "target": "Adiós"}
]
```

**Get Training Data:**
- Download parallel corpora from [OPUS](https://opus.nlpl.eu/)
- Use [Tatoeba](https://tatoeba.org/) for sentence pairs
- Create your own dataset

### Step 2: Train Model

```bash
python scripts/train_translator.py
```

Training time depends on dataset size (typically 10-30 minutes for 10K sentence pairs).

### Step 3: Use Translation

Set `translate=True` in [app.py](app.py#L105):
```python
results = app.process_video(
    video_path=video_path,
    translate=True  # Enable translation
)
```

## 🎯 Usage Examples

### Example 1: Generate Subtitles Only
```python
from src.audio_processor import AudioProcessor
from src.vad import VoiceActivityDetector
from src.transcriber import Transcriber
from src.subtitle_generator import SubtitleGenerator

# Initialize
audio_proc = AudioProcessor()
vad = VoiceActivityDetector()
transcriber = Transcriber()
subtitle_gen = SubtitleGenerator()

# Process
audio = audio_proc.convert_video_to_audio("video.mp4")
timestamps = vad.detect_speech(audio)
segments = audio_proc.segment_audio(audio, timestamps)
transcriptions = transcriber.transcribe_segments(segments)

# Generate subtitles
subtitle_gen.generate_subtitles(transcriptions, "output", format="srt")
```

### Example 2: Batch Processing
```python
import glob
from app import SubtitleApp

app = SubtitleApp()

# Process all MP4 files
for video in glob.glob("*.mp4"):
    print(f"Processing {video}...")
    app.process_video(video, translate=False)
```

### Example 3: Custom Configuration
```python
from src.transcriber import Transcriber
import config

# Override config temporarily
original_model = config.WHISPER_MODEL_SIZE
config.WHISPER_MODEL_SIZE = "medium"

transcriber = Transcriber()  # Uses medium model

# Restore original
config.WHISPER_MODEL_SIZE = original_model
```

## 🔧 Troubleshooting

### Issue: "FFmpeg not found"
**Solution**: Install FFmpeg and ensure it's in your PATH

### Issue: "Out of memory"
**Solution**: Use a smaller Whisper model (tiny or base)

### Issue: "CUDA error" 
**Solution**: Set `WHISPER_DEVICE = "cpu"` in [config.py](config.py#L13)

### Issue: Translation returns original text
**Solution**: Train the translation model first using `scripts/train_translator.py`

### Issue: Poor transcription quality
**Solution**: 
- Use a larger Whisper model (small, medium, or large)
- Ensure audio quality is good
- Adjust VAD threshold in [config.py](config.py#L16)

## 📊 Performance Tips

1. **Use GPU**: If you have CUDA, set `WHISPER_DEVICE = "cuda"` in config
2. **Smaller Models**: Use "tiny" for quick testing, "base" for production
3. **VAD Tuning**: Adjust `VAD_THRESHOLD` (0.3-0.7) based on audio quality
4. **Batch Processing**: Process multiple videos in sequence to reuse loaded models

## 🎬 Video Format Support

Supported formats:
- MP4 (recommended)
- AVI
- MKV
- MOV
- WebM
- FLV

## 📁 Project Structure Overview

```
src/
├── audio_processor.py   → Extracts and segments audio
├── vad.py              → Detects speech in audio
├── transcriber.py      → Converts speech to text
├── translator.py       → Translates text
└── subtitle_generator.py → Creates SRT/VTT files
```

## 🚀 Building for Distribution

See [BUILD_EXE.md](BUILD_EXE.md) for detailed instructions on creating a standalone .exe

**Quick build:**
```bash
pip install pyinstaller
pyinstaller --onefile --name SubtitleGenerator app.py
```

## 💡 Next Steps

1. ✅ Run `python app.py` with a test video
2. ✅ Adjust settings in [config.py](config.py) to optimize performance
3. ✅ Prepare translation training data if needed
4. ✅ Build as .exe for distribution (see [BUILD_EXE.md](BUILD_EXE.md))

## 📚 Additional Resources

- [README.md](README.md) - Full documentation
- [BUILD_EXE.md](BUILD_EXE.md) - Building executable guide
- [config.py](config.py) - All configuration options
- [scripts/train_translator.py](scripts/train_translator.py) - Training script

---

**Need help?** Check the README or open an issue!
