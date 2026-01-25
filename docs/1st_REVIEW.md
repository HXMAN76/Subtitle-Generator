# NLP Literature Review: Subtitle Generation & Translation Systems

**Team A - Batch 9**  
**Domain**: Automated Subtitle Generation and Neural Machine Translation

---

## 📚 Comprehensive Literature Review

| Study (Author, Year) | Research Focus | Architecture / Methodology | Benchmarks & Evaluation | Our Project Comparison |
|---------------------|----------------|---------------------------|------------------------|----------------------|
| **Prabhakar et al. (2025)** | Automated Subtitling & Translation (Direct relevance) | **Whisper (OpenAI)** for Speech-to-Text + **Helsinki-NLP models** for Translation | • Comparison against standard Whisper models<br>• Accuracy validation on large-scale subtitle datasets | ✅ **Similar approach**: We use faster-whisper (optimized)<br>❌ **Different NMT**: We use custom 60M Transformer vs Helsinki models<br>✅ **Edge**: 11 Indic languages with offline deployment |
| **Papi et al. (2023)** | Simultaneous Translation & Subtitling | **Triangle Transformer**: One encoder, two decoders for simultaneous output | • BLEU, SubER, and Sigma scores<br>• Tested on MuST-Cinema corpus | ❌ **Different**: Sequential pipeline (ASR → NMT) vs simultaneous<br>✅ **Advantage**: Lower latency for post-processing<br>🔄 **Trade-off**: Not real-time but more accurate |
| **Rajaboina & Sariki (2025)** | Enhanced Contextual Understanding | **Hybrid BERT-CNN-LSTM**: BERT (context) + CNN (features) + LSTM (sequence) | • Contextual accuracy (metrics not detailed) | ❌ **Different architecture**: Pure Transformer vs hybrid<br>✅ **Advantage**: Lighter model (60M vs BERT-based ~340M)<br>✅ **Efficiency**: Faster inference |
| **Anand et al. (2025)** | Real-Time Live Video Subtitling | **Hybrid Bidirectional LSTM + Transformer** | • Word Accuracy Rate<br>• Latency measurement (real-time focus) | ❌ **Not real-time**: Background processing pipeline<br>✅ **Better accuracy**: Full-audio mode vs streaming<br>🎯 **Use case**: Post-production vs live streaming |
| **Poncelet & Hamme (2025)** | Broadcast Media Transcription | **MultiTransformer Decoder**: End-to-end with cascaded encoders | • ASR and subtitle generation accuracy | ✅ **Similar**: Transformer-based architecture<br>❌ **Different**: Cascaded encoders vs standard 6-layer<br>✅ **Advantage**: Simpler, proven architecture |
| **Yu et al. (2025)** | End-to-End Extraction (Video-based) | **Vision-Language Models**: Vision encoder + InterleavedVT + LLM | • Tested on ViSa dataset (2.5M videos)<br>• Compared vs open-source tools and LVLMs | ❌ **Different modality**: Audio-only vs multimodal<br>✅ **Focused**: Specialized for audio subtitles<br>✅ **Efficiency**: No visual processing overhead |
| **Penyameen et al. (2025)** | Multilingual Video Transcription | **Whisper (OpenAI)** + FFmpeg + MoviePy | • System performance validation (metrics not specified) | ✅ **Very similar stack**: Whisper + FFmpeg + MoviePy<br>✅ **Enhancement**: faster-whisper (3-4x speedup)<br>✅ **Addition**: Custom NMT for Indic languages |
| **Google Translate WMT 2025** | LLM-based NMT Refinement | **Gemma 3 LLM** with fine-tuning and reinforcement learning | • WMT benchmarks<br>• Fluent vs literal style control | ❌ **Model size**: 60M params vs Gemma 3 (7B+)<br>✅ **Deployment**: Offline-capable vs cloud-only<br>❌ **Scope**: Task-specific vs general-purpose |
| **VNJPTranslate (2025)** | Low-Resource Language Pairs (Vietnamese-Japanese) | **LLMs with QLoRA** for efficient fine-tuning + synthetic data | • Low-resource pair performance | ✅ **Similar challenge**: Low-resource Indic languages<br>✅ **Different approach**: From-scratch training vs fine-tuning<br>✅ **Scale**: 11 languages vs 1 pair |
| **Sony AI (ACL 2025)** | Domain-Adaptive Translation | **Graph Neural Networks** for idiomatic translation + **Multi-Armed Bandit** for model selection | • Domain-specific accuracy<br>• African low-resource languages | ❌ **No dynamic selection**: Single model per language<br>✅ **Simpler**: Standard Transformer<br>🔄 **Future**: Could add model selection |
| **Multimodal NMT (2025)** | Image Caption Translation | **CNN + RNN with attention** for visual + textual inputs | • Under-resourced language performance | ❌ **Audio-only**: No visual modality<br>✅ **Focused**: Specialized for speech<br>✅ **Efficiency**: Lower computational cost |
| **AI-Powered Subtitle Management (Feb 2025)** | End-to-End Subtitle System | **ASR + MT + Segmentation + Formatting** integrated system | • Multi-language accuracy<br>• Synchronization metrics | ✅ **Complete pipeline**: Similar integrated approach<br>✅ **Enhancement**: Custom NMT vs generic MT<br>✅ **REST API**: Production-ready with FastAPI |

---

## 🎯 Our Project: Technical Specifications

### System Overview

**Name**: Multi-Language Subtitle Generator & Translator v2.0.0  
**Type**: Offline, Production-Ready REST API System

### Core Components

| Component | Technology | Specifications |
|-----------|-----------|----------------|
| **Speech Recognition** | faster-whisper (CTranslate2) | • 3-4x faster than OpenAI Whisper<br>• GPU/CPU auto-detection<br>• Models: tiny to large-v3 |
| **Translation** | Custom Transformer NMT | • 60.52M parameters per language<br>• 11 language-specific models<br>• Lazy loading architecture |
| **Tokenizer** | SentencePiece (Shared) | • 32K vocabulary<br>• 12 language tags<br>• Single tokenizer for all languages |
| **API** | FastAPI | • Background job processing<br>• Swagger/OpenAPI docs<br>• Multi-language support endpoints |

### Supported Languages

**Source**: English (en)  
**Targets**: Assamese (as), Bengali (bn), Gujarati (gu), Hindi (hi), Kannada (kn), Malayalam (ml), Marathi (mr), Odia (or), Punjabi (pa), Tamil (ta), Telugu (te)

### Performance Metrics

| Language | BLEU Score | Status | vs IndicTrans2 |
|----------|------------|--------|----------------|
| Hindi (hi) | 39.33 | ✅ Good | -8.87 |
| Gujarati (gu) | 39.76 | ✅ Competitive | -1.44 |
| Odia (or) | 63.89 | 🔍 Verify | +26.79 |
| Punjabi (pa) | 34.93 | ✅ Acceptable | -4.67 |
| Assamese (as) | 14.37 | ⚠️ Needs work | -14.93 |
| Kannada (kn) | 12.14 | ⚠️ Retrain | -21.06 |
| Malayalam (ml) | 15.97 | ⚠️ Retrain | -18.13 |
| Tamil (ta) | 15.44 | ⚠️ Retrain | -20.36 |

---

## 🔍 Research Gaps Our Project Addresses

### 1. **Offline Indic Language Translation** 🌐

**Gap**: Most commercial systems (Google Translate, Azure) require internet connectivity. Low-resource Indic languages have limited offline support.

**Our Solution**:
- ✅ Fully offline operation
- ✅ 11 Indic languages with custom-trained models
- ✅ Trained on 49.6M sentence pairs (Samanantar dataset)
- ✅ Deployable on local infrastructure

**Impact**: Enables subtitle generation in regions with limited internet or data privacy requirements.

---

### 2. **Memory-Efficient Multi-Language Systems** 💾

**Gap**: Loading multiple large translation models (1B+ params) simultaneously is memory-prohibitive for edge deployment.

**Our Solution**:
- ✅ **Lazy loading**: Models loaded on-demand
- ✅ **Shared tokenizer**: Single 5MB tokenizer for all 11 languages
- ✅ **Compact models**: 60M params per model (20x smaller than IndicTrans2)
- ✅ **Selective loading**: Load only required languages

**Impact**: Enables deployment on consumer-grade GPUs (8GB VRAM) with all 11 languages available.

---

### 3. **Production-Ready Subtitle Pipeline** 🎬

**Gap**: Academic research often focuses on individual components (ASR or NMT) without end-to-end integration.

**Our Solution**:
- ✅ **Complete pipeline**: Video → Audio → Transcription → Translation → SRT/VTT
- ✅ **REST API**: FastAPI with background job processing
- ✅ **Format support**: SRT and VTT subtitle formats
- ✅ **Progress tracking**: Real-time job status with `/jobs/{id}` endpoint

**Impact**: Can be directly deployed in production workflows without additional integration work.

---

### 4. **Optimized Inference Speed** ⚡

**Gap**: Standard Whisper and Transformer models have high inference latency for long videos.

**Our Solution**:
- ✅ **faster-whisper**: CTranslate2-optimized (3-4x speedup)
- ✅ **Full-audio mode**: Process entire audio in one pass (no chunking overhead)
- ✅ **Batch translation**: GPU-optimized subtitle translation
- ✅ **Async processing**: Non-blocking API with background tasks

**Impact**: Process 2-hour video in 25-40 minutes (GPU) vs 90-120 minutes with standard tools.

---

### 5. **Language-Specific Model Customization** 🔧

**Gap**: Multilingual models (mBART, NLLB) suffer from "curse of multilinguality" where performance degrades with more languages.

**Our Solution**:
- ✅ **Per-language models**: Separate 60M model for each target language
- ✅ **Specialized training**: Language-specific fine-tuning
- ✅ **Flexible updates**: Can retrain individual languages without affecting others
- ✅ **Dravidian strategy**: Planned specialized tokenizer for kn/ml/ta

**Impact**: Better performance for specific language pairs vs generic multilingual models.

---

### 6. **Transparent Evaluation & Benchmarking** 📊

**Gap**: Many commercial systems don't publish detailed performance metrics or comparisons.

**Our Solution**:
- ✅ **Open evaluation**: BLEU, METEOR, chrF metrics on test sets
- ✅ **Baseline comparisons**: vs IndicTrans, Google Translate (planned)
- ✅ **Reproducible**: All evaluation scripts and datasets documented
- ✅ **Gap analysis**: Identified weaknesses (Dravidian languages) with retraining plan

**Impact**: Transparent, research-grade quality assessment enabling continuous improvement.

---

### 7. **Dravidian Language Focus** 🎯

**Gap**: Dravidian languages (Kannada, Malayalam, Tamil, Telugu) are underrepresented in NMT research compared to Indo-Aryan languages.

**Our Solution**:
- ✅ **Explicit support**: 4 Dravidian languages included
- ✅ **Identified tokenization issue**: Planned separate Dravidian tokenizer
- ✅ **Retraining strategy**: Documented approach to improve BLEU by 10-15 points
- ✅ **Dataset**: Large-scale training data (4-5M pairs per language)

**Impact**: Addresses critical gap in South Indian language technology.

---

## 📊 Comparative Analysis

### Strengths vs Competitors

| Aspect | Our Project | Competitors |
|--------|-------------|-------------|
| **Offline Operation** | ✅ Full offline | ❌ Most require cloud |
| **Indic Languages** | ✅ 11 with custom models | ⚠️ Generic or limited |
| **Model Size** | ✅ 60M (efficient) | ❌ 1B+ params |
| **Deployment** | ✅ Consumer GPU (8GB) | ❌ High-end GPUs |
| **API** | ✅ REST ready | ⚠️ Varies |

### Challenges

| Aspect | Our Project | Industry |
|--------|-------------|----------|
| **Quality** | ⚠️ BLEU 39 (hi), 14-16 (Dravidian) | ✅ BLEU 48+ |
| **Real-Time** | ❌ Batch only | ✅ Some support live |
| **LLM** | ❌ Traditional Transformer | ✅ Gemma, GPT-based |
