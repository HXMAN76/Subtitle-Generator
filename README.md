# 🎬 Subtitle Generator & Translator

A production-ready, offline subtitle generation and translation system with **REST API backend**. Uses **faster-whisper** for high-speed transcription and **custom-trained XLarge Transformer NMT models** for neural machine translation to **11 Indic languages** with **per-language tokenizers** and **lazy model loading**.

**API Version**: 2.1.0 | **NMT Models**: XLarge (~385M params) | **Languages**: as, bn, gu, hi, kn, ml, mr, or, pa, ta, te

> ⚠️ **Training Status**: XLarge models are currently being trained. Results pending.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎙️ **Speech-to-Text** | High-speed transcription using faster-whisper (3-4x faster than OpenAI Whisper) |
| 🌐 **XLarge NMT** | Custom-trained 385M parameter Transformer for state-of-the-art translation |
| 🇮🇳 **11 Indic Languages** | Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia, Assamese |
| 🔤 **Per-Language Tokenizers** | Optimized tokenizer for each language (48K vocab for Dravidian) |
| 🌐 **REST API** | FastAPI backend with Swagger docs, background jobs, file uploads |
| 📝 **Subtitle Generation** | SRT and VTT format output |
| 🔌 **Offline Operation** | Runs completely locally - no cloud APIs needed |
| ⚡ **H100 Optimized** | Training optimized for H100/A100, inference on RTX 6000 Ada |

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
├── api.py                      # FastAPI REST backend
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
│       ├── config.py           # Model configs (base/large/xlarge)
│       └── languages.py        # Language definitions
│
├── scripts/                    # CLI tools
│   ├── train_tokenizer.py     # Train per-language tokenizers
│   ├── train_nmt.py           # Train translation model
│   ├── train_pipeline.sh      # Full training pipeline
│   └── evaluate_nmt.py        # Evaluate model BLEU scores
│
├── models/translation/         # Trained models (lazy loaded)
│   ├── hi/                     # Hindi model
│   │   ├── tokenizer.model     # Per-language tokenizer (32K vocab)
│   │   └── best.pt             # XLarge model (~385M params)
│   ├── ta/                     # Tamil model  
│   │   ├── tokenizer.model     # Per-language tokenizer (48K vocab, Dravidian)
│   │   └── best.pt
│   └── .../                    # Other language models
│
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── data/                       # Training data
└── output/                     # Generated subtitles
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

### Architecture: XLarge Transformer

> ⚠️ **Training Status**: Models are currently being trained. BLEU scores pending.

| Parameter | Value |
|-----------|-------|
| **Type** | Transformer (Encoder-Decoder) |
| **Parameters** | ~385 Million |
| **Encoder Layers** | 12 |
| **Decoder Layers** | 12 |
| **Attention Heads** | 16 |
| **Hidden Dimension** | 1024 |
| **Feed-Forward** | 4096 |
| **Tokenizer** | Per-language SentencePiece |
| **Vocab Size** | 32K (Indo-Aryan) / 48K (Dravidian) |
| **Dataset** | AI4Bharat Samanantar (49.6M pairs) |

### Supported Languages

| Code | Language | Family | Tokenizer Vocab |
|------|----------|--------|----------------|
| `hi` | Hindi | Indo-Aryan | 32K BPE |
| `bn` | Bengali | Indo-Aryan | 32K BPE |
| `mr` | Marathi | Indo-Aryan | 32K BPE |
| `gu` | Gujarati | Indo-Aryan | 32K BPE |
| `pa` | Punjabi | Indo-Aryan | 32K BPE |
| `or` | Odia | Indo-Aryan | 32K BPE |
| `as` | Assamese | Indo-Aryan | 32K BPE |
| `ta` | Tamil | **Dravidian** | **48K Unigram** |
| `te` | Telugu | **Dravidian** | **48K Unigram** |
| `kn` | Kannada | **Dravidian** | **48K Unigram** |
| `ml` | Malayalam | **Dravidian** | **48K Unigram** |

### Train Your Own Model

```bash
# 1. Train using the automated pipeline (Recommended)
# Handles data splitting, tokenizer training, and model training
bash scripts/train_pipeline.sh ta

# OR Manual Steps:

# 1. Train per-language tokenizer (optimized for each language)
python scripts/train_tokenizer.py --target-lang ta

# 2. Train XLarge model (~385M params)
python scripts/train_nmt.py --target-lang ta --config xlarge --streaming

# 3. Evaluate
python scripts/evaluate_nmt.py --language ta
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
- ✅ **Batch Translation**: Efficient GPU utilization
- ✅ **Background Jobs**: Non-blocking API requests
- ✅ **Lazy Loading**: Models load on-demand (memory efficient)
- ✅ **Per-Language Tokenizers**: Optimized vocabulary per language
- ✅ **XLarge Architecture**: 385M params for maximum quality
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
