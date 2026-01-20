# 🎬 Subtitle Generator & Translator

A production-ready, offline subtitle generation and translation system with **REST API backend**. Uses **faster-whisper** for high-speed transcription and **custom-trained Transformer NMT models** for neural machine translation to **11 Indic languages** with **lazy model loading**.

**API Version**: 2.0.0 | **NMT Models**: 60.52M params each | **Languages**: as, bn, gu, hi, kn, ml, mr, or, pa, ta, te


## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎙️ **Speech-to-Text** | High-speed transcription using faster-whisper (3-4x faster than OpenAI Whisper) |
| 🌐 **Neural Translation** | Custom-trained 60M parameter Transformer for 11 Indic languages |
| 🇮🇳 **Multi-Language** | Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia, Assamese |
| 🌐 **REST API** | FastAPI backend with Swagger docs, background jobs, file uploads |
| 📝 **Subtitle Generation** | SRT and VTT format output |
| 🔌 **Offline Operation** | Runs completely locally - no cloud APIs needed |
| ⚡ **Full Audio Mode** | Processes entire audio in one pass for maximum speed |
| 🎯 **Auto GPU/CPU** | Automatically uses CUDA if available, falls back to CPU |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SUBTITLE GENERATOR v2.0                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      FastAPI Backend (api.py)                       │    │
│  │                                                                     │    │
│  │   GET  /languages ──► Available target languages                    │    │
│  │   POST /upload?target_lang=hi ──► Background Job                    │    │
│  │   POST /translate?target_lang=as ──► Instant Response               │    │
│  │   GET  /download/{id}/translated ──► SRT/VTT file                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     Processing Pipeline                             │    │ 
│  │                                                                     │    │
│  │  ┌─────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────┐  │    │
│  │  │  Video  │───►│ Audio Extract│───►│ Transcribe  │───►│Subtitles│  │    │
│  │  │  Input  │    │   (FFmpeg)   │    │(faster-whisper)│ │  (SRT)  │  │    │
│  │  └─────────┘    └──────────────┘    └──────┬──────┘    └─────────┘  │    │
│  │                                            │                        │    │
│  │                                            ▼                        │    │
│  │               ┌────────────────────────────────────────────────┐    │    │
│  │               │    Multi-Language Translator (Lazy Loading)    │    │    │
│  │               │                                                │    │    │
│  │               │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │    │    │
│  │               │  │ as  │ │ bn  │ │ gu  │ │ hi  │ │ ... │       │    │    │
│  │               │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘       │    │    │
│  │               │       (models loaded on-demand)                 │    │    │
│  │               └────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Subtitle-Generator/
├── api.py                      # FastAPI REST backend (v2.0.0)
├── app.py                      # CLI application (3-step pipeline)
├── config.py                   # Configuration settings
├── requirements.txt            # Dependencies
│
├── src/                        # Core modules
│   ├── audio_processor.py      # Video → Audio extraction
│   ├── transcriber.py          # faster-whisper transcription
│   ├── translator.py           # Multi-language NMT wrapper (lazy loading)
│   ├── subtitle_generator.py   # SRT/VTT generation
│   └── nmt/                    # Neural Machine Translation
│       ├── model/              # Transformer architecture
│       ├── training/           # Training pipeline
│       ├── inference/          # Translation inference
│       └── languages.py        # Language definitions
│
├── scripts/                    # CLI tools
│   ├── train_pipeline.sh      # Full training pipeline
│   ├── train_nmt.py           # Train translation model
│   ├── copy_models.sh         # Copy trained models
│   └── download_dataset.py    # Download training data
│
├── models/translation/         # Trained models (lazy loaded)
│   ├── nmt_spm.model          # Shared SentencePiece tokenizer
│   ├── nmt_spm.vocab          # Vocabulary file
│   ├── as/best.pt             # Assamese model (60M params)
│   ├── bn/best.pt             # Bengali model
│   ├── gu/best.pt             # Gujarati model
│   ├── hi/best.pt             # Hindi model
│   └── .../best.pt            # Other language models
│
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── data/                       # Training data
├── output/                     # Generated subtitles
└── temp/                       # Temporary files
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **FFmpeg** (for audio processing)
- **CUDA** (optional, for GPU acceleration)

### Installation

```bash
# Clone repository
git clone https://github.com/your-repo/Subtitle-Generator.git
cd Subtitle-Generator

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🌐 REST API Usage

### Start the Server

```bash
# Development (with auto-reload)
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Interactive Docs

Open in browser: **http://localhost:8000/docs**

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info |
| `GET` | `/health` | Health check & available languages |
| `GET` | `/languages` | **List supported/available languages** |
| `GET` | `/docs` | **Swagger UI** (interactive docs) |
| `POST` | `/upload?target_lang=hi` | Upload video → Start processing |
| `GET` | `/jobs/{id}` | Check job status & progress |
| `GET` | `/jobs` | List all jobs |
| `GET` | `/download/{id}/original` | Download original subtitles |
| `GET` | `/download/{id}/translated` | Download translated subtitles |
| `POST` | `/translate` | Translate single text |
| `POST` | `/translate/batch` | Translate multiple texts |
| `DELETE` | `/jobs/{id}` | Delete job & files |

### Example: Upload Video

```bash
# Upload with Hindi subtitles (default)
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_video.mp4"

# Upload with Assamese subtitles
curl -X POST "http://localhost:8000/upload?translate=true&target_lang=as" \
  -F "file=@your_video.mp4"

# Upload with Bengali subtitles
curl -X POST "http://localhost:8000/upload?translate=true&target_lang=bn" \
  -F "file=@your_video.mp4"

# Response: {"job_id": "abc123", "status_url": "/jobs/abc123"}
```

### Example: Check Job Status

```bash
curl http://localhost:8000/jobs/abc123

# Response: {"status": "completed", "progress": 1.0, ...}
```

### Example: Translate Text

```bash
# Translate to Hindi (default)
curl -X POST "http://localhost:8000/translate" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'

# Translate to Tamil
curl -X POST "http://localhost:8000/translate" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "target_lang": "ta"}'

# Check available languages
curl http://localhost:8000/languages
```

---

## 💻 CLI Usage

### Process a Video

```bash
# Edit video_path in app.py first
python app.py
```

### Interactive Translation

```bash
python scripts/translate.py --checkpoint models/translation/best.pt --interactive
```

---

## ⚙️ Configuration

Edit `config.py`:

```python
# Whisper settings
WHISPER_MODEL_SIZE = "tiny"   # tiny, base, small, medium, large-v3
WHISPER_DEVICE = "cuda"       # Auto-detected (cuda/cpu)

# Translation (Multi-language)
# Supported: as, bn, gu, hi, kn, ml, mr, or, pa, ta, te
SOURCE_LANGUAGE = "en"
TARGET_LANGUAGE = "hi"        # Default target language

# Subtitle format
SUBTITLE_FORMAT = "srt"       # srt, vtt
```

### Model Comparison

| Model | Speed | Accuracy | VRAM | Use Case |
|-------|-------|----------|------|----------|
| `tiny` | ⚡⚡⚡⚡⚡ | 70% | 1GB | Testing |
| `base` | ⚡⚡⚡⚡ | 80% | 1GB | General |
| `small` | ⚡⚡⚡ | 88% | 2GB | **Recommended** |
| `medium` | ⚡⚡ | 92% | 5GB | Quality |
| `large-v3` | ⚡ | 95% | 10GB | Professional |

---

## 🧠 Translation Model

### Supported Languages

| Code | Language | Dataset Size |
|------|----------|-------------|
| `hi` | Hindi | 8.6M pairs |
| `ta` | Tamil | 5.3M pairs |
| `te` | Telugu | 4.8M pairs |
| `bn` | Bengali | 8.5M pairs |
| `mr` | Marathi | 3.6M pairs |
| `gu` | Gujarati | 3.1M pairs |
| `kn` | Kannada | 4.0M pairs |
| `ml` | Malayalam | 5.8M pairs |
| `pa` | Punjabi | 2.4M pairs |
| `or` | Odia | 1.0M pairs |
| `as` | Assamese | 140K pairs |

### Architecture

- **Type**: Transformer (Encoder-Decoder)
- **Parameters**: 60.52 Million
- **Layers**: 6 encoder + 6 decoder
- **Attention Heads**: 8
- **Hidden Dim**: 512
- **Tokenizer**: SentencePiece (32K vocab)
- **Dataset**: AI4Bharat Samanantar (49.6M pairs)

### Train Your Own Model

```bash
# Download dataset for a specific language
python scripts/download_dataset.py --lang hi    # Hindi
python scripts/download_dataset.py --lang ta    # Tamil
python scripts/download_dataset.py --all-langs  # All languages

# Create combined tokenizer corpus
python scripts/download_dataset.py --lang hi ta te --create-corpus

# Train model for Hindi
python scripts/train_nmt.py --target-lang hi --streaming

# Train for Tamil with small config
python scripts/train_nmt.py --target-lang ta --config small

# Evaluate
python scripts/evaluate_nmt.py --checkpoint models/translation/best.pt --samples 10
```

---

## 📊 Performance

### Time Estimates (2-hour video)

| Step | Time (GPU) | Time (CPU) |
|------|------------|------------|
| Audio Extraction | 30 sec | 30 sec |
| Transcription | 15-25 min | 60-90 min |
| Translation | 5-10 min | 15-20 min |
| **Total** | **25-40 min** | **90-120 min** |

### Optimizations Applied

- ✅ **faster-whisper**: 3-4x faster than OpenAI Whisper
- ✅ **Full Audio Mode**: Single-pass processing
- ✅ **Batch Translation**: Efficient GPU utilization
- ✅ **Background Jobs**: Non-blocking API requests
- ✅ **Lazy Loading**: Models load on-demand (memory efficient)
- ✅ **Shared Tokenizer**: One tokenizer for all 11 languages
- ✅ **Model Caching**: Loaded models stay in memory

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| FFmpeg not found | Install FFmpeg and add to PATH |
| CUDA out of memory | Use smaller Whisper model (`tiny` or `base`) |
| Translation returns original | Ensure `models/translation/{lang}/best.pt` exists |
| Slow transcription | Check `WHISPER_DEVICE` is `cuda` |
| API port in use | Change port: `uvicorn api:app --port 8001` |
| Language not available | Check `/languages` endpoint for available models |

---

## 📦 Dependencies

### Core
- **faster-whisper** - CTranslate2-optimized Whisper
- **torch** - Neural network framework
- **sentencepiece** - Tokenization
- **moviepy** - Video processing

### API
- **fastapi** - REST API framework
- **uvicorn** - ASGI server
- **python-multipart** - File uploads

### System
- **FFmpeg** - Audio extraction
- **CUDA** (optional) - GPU acceleration

---

## 🎯 Roadmap

- [x] faster-whisper integration
- [x] Full audio mode
- [x] Custom NMT model
- [x] REST API backend
- [x] Multiple language pairs (11 Indic languages)
- [x] Multi-language lazy loading (v2.0.0)
- [x] Per-language model files
- [ ] Music detection (`[♪ Music ♪]`)
- [ ] Web UI frontend
- [ ] Docker deployment

---

## 📝 License

MIT License - free for personal and commercial use.

---

## 🙏 Acknowledgments

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - High-speed transcription
- [FastAPI](https://fastapi.tiangolo.com/) - Modern API framework
- [AI4Bharat Samanantar](https://huggingface.co/datasets/ai4bharat/samanantar) - Multi-language training data
- [SentencePiece](https://github.com/google/sentencepiece) - Tokenization
