# 🤖 Agentic Multimodal RAG System

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-purple.svg)](https://ollama.ai)
[![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-orange.svg)](https://github.com/facebookresearch/faiss)

**An enterprise-grade, self-correcting RAG system with multimodal support, agentic planning, and production-ready evaluation.**

[Quick Start](#-quick-start) • [Architecture](#-architecture) • [Features](#-key-features) • [Configuration](#-configuration) • [Evaluation](#-evaluation)

</div>

---

## 📖 Overview

This project implements an advanced **Agentic Multimodal Retrieval-Augmented Generation (RAG)** system that answers user queries by dynamically retrieving and reasoning over **text, image, audio, and video** data.

Unlike static RAG pipelines, this system uses an **agentic planner** to orchestrate retrieval strategies, verify answer faithfulness, and adaptively retry or abstain—ensuring grounded, explainable, and trustworthy outputs.

### What Makes This Different?

| Feature | Traditional RAG | This System |
|---------|----------------|-------------|
| Retrieval | Single-pass dense search | Hybrid (Dense + Sparse + Reranking) |
| Planning | Static pipeline | Agentic intent-based planning |
| Verification | None | NLI-based faithfulness + citation checks |
| Modalities | Text only | Text, Image, Audio, Video |
| Self-correction | None | Automatic retry on low confidence |
| Citations | Optional | Enforced with verification |

---

## ✨ Key Features

### 🧠 Agentic Control Plane
- **Intent Classification**: Automatically detects query type (factual, explanatory, comparative, visual, etc.)
- **Modality Selection**: Intelligently selects which data sources to query
- **Adaptive Retry**: Self-corrects when verification confidence is low
- **Confidence Scoring**: Transparent scoring for all responses

### 🔍 Advanced Retrieval
- **Hybrid Search**: Combines FAISS dense vectors (BGE-M3) + BM25 sparse retrieval
- **Cross-Encoder Reranking**: BAAI/bge-reranker for precise result ordering
- **Query Expansion**: HyDE and multi-perspective query generation
- **Parallel Retrieval**: 5.6x latency improvement with concurrent execution

### 📚 Multimodal Support
- **Text**: PDF, DOCX, TXT, Markdown with semantic chunking
- **Images**: CLIP embeddings + BLIP-2 captioning + OCR
- **Audio**: Whisper transcription to searchable text
- **Video**: Frame extraction + scene summarization

### ✅ Grounded Generation
- **Citation Enforcement**: All answers include `[source#chunk]` citations
- **NLI Verification**: DeBERTa-based faithfulness checking
- **Abstention**: System refuses to answer when evidence is insufficient

### ⚡ Performance Optimizations
- **Result Caching**: Fuzzy query matching with configurable TTL
- **GPU Acceleration**: CUDA support for embeddings, reranking, and generation
- **Memory Management**: Automatic CPU fallback on OOM

---

## 🏗️ Architecture

### High-Level Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                                  │
│                  chat.py (CLI) | ui/app.py (Web)                       │
└────────────────────────────┬───────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION ENGINE                                 │
│              orchestrator/execution_engine.py                          │
│                                                                         │
│   Query → Cache Check → Plan → Retrieve → Fuse → Generate → Verify    │
└────────────────────────────┬───────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ CONTROL PLANE │   │  DATA PLANE   │   │  EVALUATION   │
│               │   │               │   │               │
│ • Planner     │   │ • Retrieval   │   │ • RAGAS       │
│ • Verifier    │   │ • Fusion      │   │ • Metrics     │
│               │   │ • Generation  │   │ • SLO Checks  │
└───────────────┘   └───────────────┘   └───────────────┘
```

### Control Plane vs Data Plane

| Control Plane (Decision Making) | Data Plane (Execution) |
|--------------------------------|------------------------|
| Agentic Planner | Multimodal Ingestion |
| Intent Classification | Dense Retrieval (FAISS) |
| Modality Selection | Sparse Retrieval (BM25) |
| Retry Logic | Cross-Encoder Reranking |
| Verifier / Critic | Context Fusion |
| Abstention Rules | LLM Generation |

### Agentic Loop Flow

```
1. Query Received
       │
       ▼
2. ┌─────────────────────────────────┐
   │    AGENTIC PLANNER              │
   │  • Classify intent              │
   │  • Select modalities            │
   │  • Set retrieval parameters     │
   └─────────────┬───────────────────┘
                 │
       ▼─────────┴─────────▼
3. ┌─────────────┐  ┌─────────────┐
   │   DENSE     │  │   SPARSE    │  ← Parallel Execution
   │   (FAISS)   │  │   (BM25)    │
   └──────┬──────┘  └──────┬──────┘
          └────────┬───────┘
                   ▼
4.         ┌─────────────┐
           │  RERANKER   │
           │  (BGE)      │
           └──────┬──────┘
                  │
                  ▼
5.         ┌─────────────┐
           │  CONTEXT    │
           │  FUSION     │
           └──────┬──────┘
                  │
                  ▼
6.         ┌─────────────┐
           │  GROUNDED   │
           │  LLM GEN    │
           └──────┬──────┘
                  │
                  ▼
7.         ┌─────────────┐
           │  VERIFIER   │──── Low Confidence ────┐
           └──────┬──────┘                        │
                  │                               │
           High Confidence                  Retry (up to N)
                  │                               │
                  ▼                               │
8.         ┌─────────────┐                        │
           │   RESPONSE  │ ◄──────────────────────┘
           │ + Citations │
           └─────────────┘
```

---

## 📁 Project Structure

```
agentic-multimodal-rag/
│
├── config/
│   └── settings.yaml          # Centralized configuration
│
├── planner/                   # 🧠 Control Plane - Orchestration
│   ├── agentic_planner.py     # Intent classification & strategy
│   ├── local_agentic_planner.py
│   └── schemas.py             # Data models
│
├── ingestion/                 # 📥 Offline Data Processing
│   ├── text_ingest.py         # PDF, DOCX, TXT, MD
│   ├── image_ingest.py        # CLIP + BLIP-2 + OCR
│   ├── audio_ingest.py        # Whisper transcription
│   └── video_ingest.py        # Frame extraction
│
├── retrieval/                 # 🔍 Query-Time Retrieval
│   ├── dense_retriever.py     # FAISS semantic search
│   ├── sparse_retriever.py    # BM25 keyword search
│   ├── reranker.py            # Cross-encoder reranking
│   ├── parallel_retriever.py  # Concurrent execution
│   └── query_expansion.py     # HyDE, reformulation
│
├── fusion/                    # 🔗 Context Assembly
│   └── context_fusion.py      # Dedup, MMR, token budget
│
├── generation/                # 💬 Answer Generation
│   └── grounded_llm.py        # Citation-aware LLM
│
├── verification/              # ✅ Quality Assurance
│   └── verifier.py            # NLI faithfulness checks
│
├── evaluation/                # 📊 Metrics & Benchmarks
│   ├── metrics.py             # BLEU, ROUGE, BERTScore
│   ├── ragas.py               # RAGAS framework
│   └── slo.yaml               # Service Level Objectives
│
├── orchestrator/              # 🎯 Main Entry Point
│   └── execution_engine.py    # Pipeline coordinator
│
├── ui/                        # 🖥️ User Interfaces
│   └── app.py                 # Streamlit web app
│
├── wrappers/                  # 🔧 Model Wrappers (Multimodal)
│   ├── image_encoder.py       # CLIP image embeddings
│   ├── caption.py             # BLIP-2 image captioning
│   ├── ocr.py                 # Multilingual OCR (80+ languages)
│   └── multimodal_image_processor.py  # Combined processor
│
├── scripts/                   # 🛠️ Utilities
│   ├── embed_and_index.py     # Build FAISS index
│   ├── retrieve_demo.py       # Test retrieval
│   └── run_ragas_pipeline.py  # Evaluation pipeline
│
├── tests/                     # 🧪 Test Suite
│   ├── test_end_to_end.py
│   ├── test_retrieval_accuracy.py
│   └── comprehensive_test.py
│
├── chat.py                    # Interactive CLI chat
└── requirements.txt           # Dependencies
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Ollama** (for local LLM inference)
- **8GB+ RAM** (16GB recommended)
- **NVIDIA GPU** (optional, for faster inference)

### Installation

```powershell
# 1. Clone the repository
git clone https://github.com/your-org/agentic-multimodal-rag.git
cd agentic-multimodal-rag

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the local LLM model (one-time)
ollama pull llama3:8b
```

### Environment Variables (.env)

This repo supports loading secrets from a local `.env` file (never commit it).

```powershell
# Copy the template
copy .env.example .env

# Then edit .env and set at least one provider:
# - OPENROUTER_API_KEY (cloud)
# - or run Ollama locally (default fallback)
```

### First Run

```powershell
# Step 1: Add documents to data/raw/text/

# Step 2: Ingest and index your documents
python -m ingestion.text_ingest
python -m scripts.embed_and_index

# Step 3: Start chatting!
python chat.py
```

---

## �️ Multimodal Data Usage

This system supports **text, image, audio, and video** data. Each modality has its own ingestion pipeline.

### Data Directory Structure

```
data/
├── raw/
│   ├── text/              # PDF, DOCX, TXT, MD files
│   ├── image/             # JPG, PNG images
│   ├── audio/             # MP3, WAV, M4A, FLAC, OGG files
│   └── video/             # MP4, AVI, MKV, MOV files
├── processed/
│   ├── chunks.jsonl       # All processed chunks (all modalities)
│   ├── image_embeddings.npz  # CLIP embeddings for images
│   └── video_frames/      # Extracted video keyframes
└── index/
    ├── faiss_text.index   # Text vector index
    └── faiss_image.index  # Image vector index
```

### 📄 Text Ingestion

**Supported formats:** PDF, DOCX, TXT, Markdown

```powershell
# 1. Place documents in data/raw/text/
# 2. Run ingestion
python -m ingestion.text_ingest

# 3. Build index
python -m scripts.embed_and_index
```

**Output:** Semantic chunks with metadata in `data/processed/chunks.jsonl`

---

### 🖼️ Image Ingestion

**Supported formats:** JPG, PNG, BMP, TIFF, WebP

**Pipeline:** Image → CLIP Embedding → BLIP-2 Caption → OCR Text → Combined Content

```powershell
# 1. Place images in data/raw/image/
# Or extract images from PDFs:
python -m ingestion.extract_images

# 2. Process images (CLIP + BLIP-2 + OCR)
python -m ingestion.image_ingest

# 3. Build image index
python -m scripts.embed_and_index
```

**What happens:**
- **CLIP Encoder** (`wrappers/image_encoder.py`): Creates 768-dim semantic embeddings
- **BLIP-2 Captioning** (`wrappers/caption.py`): Generates descriptive text captions
- **Multilingual OCR** (`wrappers/ocr.py`): Extracts text from images (80+ languages)

**Output fields per image:**
```json
{
  "chunk_id": "img_001",
  "modality": "image",
  "source_file": "diagram.png",
  "caption": "A flowchart showing the RAG pipeline architecture",
  "ocr_text": "Query → Retrieval → Generation",
  "combined_content": "A flowchart showing... Query → Retrieval..."
}
```

---

### 🎵 Audio Ingestion

**Supported formats:** MP3, WAV, M4A, FLAC, OGG, WebM, WMA, AAC

**Pipeline:** Audio → Whisper Transcription → Timestamped Segments → Searchable Chunks

```powershell
# 1. Place audio files in data/raw/audio/

# 2. Transcribe with Whisper
python -m ingestion.audio_ingest

# 3. Index transcripts
python -m scripts.embed_and_index
```

**Features:**
- **OpenAI Whisper** (medium/large-v3): SOTA multilingual ASR
- **100+ Languages**: Including Hindi, Telugu, Tamil, Chinese, Japanese, etc.
- **Word-level timestamps**: For precise citation
- **Automatic language detection**

**Output fields per audio:**
```json
{
  "chunk_id": "audio_001",
  "modality": "audio",
  "source_file": "interview.mp3",
  "transcript": "Welcome to our discussion on transformer architecture...",
  "language_detected": "en",
  "duration_seconds": 1847.5,
  "segments": [{"start": 0.0, "end": 5.2, "text": "Welcome..."}]
}
```

---

### 🎬 Video Ingestion

**Supported formats:** MP4, AVI, MKV, MOV, WebM, FLV, WMV

**Pipeline:** Video → Keyframe Extraction → Frame Captioning → Audio Transcription → Combined Content

```powershell
# 1. Place videos in data/raw/video/

# 2. Process video (frames + audio)
python -m ingestion.video_ingest

# 3. Index all content
python -m scripts.embed_and_index
```

**Features:**
- **Intelligent keyframe extraction**: Scene change detection
- **BLIP-2 frame captioning**: Describes each keyframe
- **Whisper audio transcription**: Full audio track transcription
- **OCR on frames**: Extracts text from slides/diagrams
- **Timestamps**: HH:MM:SS format for precise retrieval

**Output fields per video chunk:**
```json
{
  "chunk_id": "video_001_frame_042",
  "modality": "video",
  "source_file": "lecture.mp4",
  "frame_path": "data/processed/video_frames/lecture_00042.jpg",
  "frame_caption": "A presenter explaining neural network layers",
  "frame_ocr": "Layer 1: Input → Layer 2: Hidden",
  "audio_transcript": "Now let's look at the hidden layer...",
  "timestamp_seconds": 126.0,
  "timestamp_formatted": "00:02:06"
}
```

---

### 🔄 Full Multimodal Workflow

```powershell
# Complete workflow for all modalities

# Step 1: Ingest all data types
python -m ingestion.text_ingest    # Text documents
python -m ingestion.image_ingest   # Images
python -m ingestion.audio_ingest   # Audio files
python -m ingestion.video_ingest   # Videos

# Step 2: Build unified index
python -m scripts.embed_and_index

# Step 3: Query across all modalities
python chat.py
```

**Example multimodal query:**
```
> What does the architecture diagram show and what did the speaker explain about it?
```

The system will retrieve relevant image captions, video frames, and audio transcripts to answer.

---

## �📋 Usage Guide

### Interactive Chat

```powershell
python chat.py
```

This launches an interactive CLI where you can ask questions about your indexed documents.

### Single Query (Orchestrator)

```powershell
python -m orchestrator.execution_engine
```

### Web Interface

```powershell
streamlit run ui/app.py
```

### Full Pipeline Commands

| Task | Command |
|------|---------|
| Activate environment | `.\venv\Scripts\Activate.ps1` |
| Start Ollama server | `ollama serve` |
| Ingest text documents | `python -m ingestion.text_ingest` |
| Ingest images | `python -m ingestion.image_ingest` |
| Build FAISS index | `python -m scripts.embed_and_index` |
| Test retrieval | `python -m scripts.retrieve_demo` |
| Interactive chat | `python chat.py` |
| Run evaluation | `python -m evaluation.eval_retrieval` |
| RAGAS evaluation | `python scripts/run_ragas_pipeline.py --input evaluation/ragas_dev.jsonl --output artifacts/predictions.jsonl` |

---

## ⚙️ Configuration

All settings are centralized in `config/settings.yaml`:

### Core Settings

```yaml
agent:
  confidence_threshold: 0.7    # Min confidence to accept answer
  max_attempts: 2              # Retry attempts on low confidence
  phase1_enabled: true         # Enable optimizations

models:
  embedding_model: "BAAI/bge-m3"           # Text embeddings (1024-dim)
  image_embedding_model: "openai/clip-vit-large-patch14"  # Image (768-dim)

planner:
  mode: agentic                # agentic | fast (rule-based)
  local_model: "gemma2:2b"     # Planning model
  use_gpu: true

retrieval:
  dense_k: 100       # Dense retrieval candidates
  sparse_k: 100      # Sparse retrieval candidates
  rerank_k: 50       # After reranking

generation:
  provider: "auto"   # auto | openrouter | ollama
  max_generation_tokens: 1024
  temperature: 0.3

verification:
  enabled: true
  faithfulness_threshold: 0.65
```

### LLM Provider Options

| Provider | Setup | Use Case |
|----------|-------|----------|
| **Ollama** (default) | `ollama pull llama3:8b` | Local, private, free |
| **OpenRouter** | Set `OPENROUTER_API_KEY` in `.env` | Cloud, higher quality |

---

## 📊 Evaluation

### RAGAS Framework

This repository includes a reproducible RAGAS evaluation pipeline:

```powershell
# Run full evaluation pipeline
python scripts/run_ragas_pipeline.py \
  --input evaluation/ragas_dev.jsonl \
  --output artifacts/predictions.jsonl

# Convert to CSV for analysis
python scripts/jsonl_to_csv.py \
  --input artifacts/predictions.jsonl \
  --output artifacts/predictions.csv
```

### Metrics Computed

| Category | Metrics |
|----------|---------|
| **Retrieval** | Recall@1, Recall@5, Recall@10, MRR, nDCG |
| **Generation** | BLEU, ROUGE-1, ROUGE-2, ROUGE-L |
| **Grounding** | Faithfulness, Citation Coverage, BERTScore |
| **RAGAS Official** | Answer Relevancy, Context Precision, Context Recall |

### Service Level Objectives (SLOs)

Defined in `evaluation/slo.yaml`:

```yaml
retrieval:
  min_recall_hit_at_5: 0.80
  min_mrr: 0.50
  min_ndcg_at_10: 0.60

latency:
  max_pipeline_p95_ms: 8000
  max_retrieval_p95_ms: 500
```

### Current Performance

| Metric | Score |
|--------|-------|
| Recall@1 | 0.80 |
| Recall@5 | 1.00 |
| Recall@10 | 1.00 |
| MRR | 0.84 |
| ROUGE-1 | 0.35 |
| Avg Latency | ~50s end-to-end |

---

## 🧪 Testing

```powershell
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_retrieval_accuracy.py -v
pytest tests/test_end_to_end.py -v
pytest tests/comprehensive_test.py -v
```

---

## 🛠️ Development

### Adding New Documents

1. Place documents in `data/raw/text/` (or appropriate modality folder)
2. Run ingestion: `python -m ingestion.text_ingest`
3. Rebuild index: `python -m scripts.embed_and_index`

### Adding New Modalities

1. Create ingestion module in `ingestion/`
2. Update chunking logic in `semantic_chunker.py`
3. Configure index in `config/settings.yaml`

### Extending the Planner

Modify intent classification in `planner/local_agentic_planner.py`:

```python
INTENT_TYPES = [
    "factual",
    "explanatory",
    "comparative",
    "visual",
    "temporal",
    "procedural",
    "aggregation",
    "unknown"
]
```

---

## 📚 Additional Documentation

| Document | Description |
|----------|-------------|
| [QUICK_START.md](QUICK_START.md) | 5-minute setup guide |
| [HOW_TO_RUN.md](HOW_TO_RUN.md) | Step-by-step commands |
| [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md) | Detailed architecture diagrams |
| [RAGAS_FRAMEWORK_REPORT.md](RAGAS_FRAMEWORK_REPORT.md) | Evaluation methodology |
| [FAANG_PROJECT_REPORT.md](FAANG_PROJECT_REPORT.md) | Engineering deep-dive |
| [RUN_END_TO_END.md](RUN_END_TO_END.md) | Complete workflow guide |

---

## 🔧 Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| "Connection refused" | Ensure `ollama serve` is running |
| "Model not found" | Run `ollama pull llama3:8b` |
| CUDA out of memory | Set `use_gpu: false` in config or use smaller batch sizes |
| Slow generation | Enable caching in `config/settings.yaml` |
| Missing citations | Ensure documents are properly indexed |

### Performance Tips

- **Enable caching**: Set `cache.enabled: true` for repeated queries
- **Use GPU**: Ensure CUDA is available for embeddings and reranking
- **Reduce chunk size**: Lower `dense_k` and `sparse_k` for faster retrieval
- **Use fast planner**: Set `planner.mode: fast` for rule-based planning

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **FAISS** by Meta AI for vector similarity search
- **Sentence Transformers** for embedding models
- **Ollama** for local LLM inference
- **HuggingFace** for model hosting and transformers library
- **RAGAS** for evaluation framework

---

<div align="center">

**Built with ❤️ for reliable, grounded AI**

[Report Bug](https://github.com/your-org/agentic-multimodal-rag/issues) • [Request Feature](https://github.com/your-org/agentic-multimodal-rag/issues)

</div>
