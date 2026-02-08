# Models Reference Guide

This document lists all AI/ML models used in the Agentic Multimodal RAG system, their purpose, and file locations.

---

## 📊 Summary Table

| Model | Purpose | Source | Auto-Pull | Required |
|-------|---------|--------|-----------|----------|
| `BAAI/bge-m3` | Text embedding | HuggingFace | ✅ | ✅ Yes |
| `openai/clip-vit-large-patch14` | Image embedding | HuggingFace | ✅ | For images |
| `gemma2:2b` | Agentic planner | Ollama | ✅ | Optional |
| `gemma3:4b` | Fallback planner | Ollama | ✅ | Optional |
| `llama3:8b` | Answer generation (local) | Ollama | ❌ | If no API |
| `nvidia/nemotron-3-nano-30b-a3b:free` | Answer generation (API) | OpenRouter | N/A | If API key set |
| `BAAI/bge-reranker-base` | Reranking | HuggingFace | ✅ | ✅ Yes |
| `microsoft/deberta-v3-base` | Faithfulness verification | HuggingFace | ✅ | Optional |
| `Salesforce/blip2-flan-t5-xl` | Image captioning | HuggingFace | ✅ | For images |

---

## 🔍 Detailed Model Reference

### 1. Text Embedding: `BAAI/bge-m3`
**Purpose:** Converts text chunks into dense vector embeddings (1024-dim) for semantic search.

**Files:**
- [config/settings.yaml](config/settings.yaml) - `models.embedding_model`
- [utils/sota_embedder.py](utils/sota_embedder.py) - Lines 5, 64-65, 76, 383
- [utils/dual_embedder.py](utils/dual_embedder.py) - Lines 5, 7, 39, 48
- [utils/model_loader.py](utils/model_loader.py) - Line 89
- [fusion/context_fusion.py](fusion/context_fusion.py) - Tokenizer

**Specs:**
- Dimension: 1024
- Max context: 8192 tokens
- Multilingual: Yes

---

### 2. Image Embedding: `openai/clip-vit-large-patch14`
**Purpose:** Converts images into dense vector embeddings (768-dim) for multimodal search.

**Files:**
- [config/settings.yaml](config/settings.yaml) - `models.image_embedding_model`, `image_processing.image_encoder`
- [wrappers/image_encoder.py](wrappers/image_encoder.py) - Line 39
- [wrappers/multimodal_image_processor.py](wrappers/multimodal_image_processor.py) - Lines 107, 253
- [utils/dual_embedder.py](utils/dual_embedder.py) - Line 49
- [utils/model_loader.py](utils/model_loader.py) - Line 64

**Specs:**
- Dimension: 768
- Type: Vision Transformer (ViT)

---

### 3. Agentic Planner: `gemma2:2b` (Primary) / `gemma3:4b` (Fallback)
**Purpose:** Classifies query intent, selects modalities, and determines retrieval strategy.

**Files:**
- [config/settings.yaml](config/settings.yaml) - `planner.local_model`, `planner.fallback_model`
- [planner/agentic_planner.py](planner/agentic_planner.py) - Lines 4-7, 51-67, 137-174, 259-264, 297, 333, 398-402

**Auto-Pull:** ✅ Yes (added in recent update)

**Fallback:** If unavailable, uses rule-based planning automatically.

---

### 4. Answer Generation (Local): `llama3:8b`
**Purpose:** Generates grounded answers with citations using retrieved context.

**Files:**
- [config/settings.yaml](config/settings.yaml) - `planner.generator.local_model`, `models.generator_model`
- [generation/grounded_llm.py](generation/grounded_llm.py) - Lines 79, 82, 161, 165, 171, 176, 178, 182
- [retrieval/query_expansion.py](retrieval/query_expansion.py) - Line 76
- [scripts/run_ragas_pipeline.py](scripts/run_ragas_pipeline.py) - Line 139
- [scripts/run_full_eval.py](scripts/run_full_eval.py) - Line 106

**Specs:**
- Context: 8k tokens
- Optimized for instruction-following

---

### 5. Answer Generation (API): `nvidia/nemotron-3-nano-30b-a3b:free`
**Purpose:** Cloud-based answer generation via OpenRouter API.

**Files:**
- [config/settings.yaml](config/settings.yaml) - `generation.openrouter.model`
- [generation/grounded_llm.py](generation/grounded_llm.py) - OpenRouter client initialization

**Requirements:**
- Set `OPENROUTER_API_KEY` or `OP_TOKEN` environment variable
- Free tier available

---

### 6. Reranker: `BAAI/bge-reranker-base`
**Purpose:** Re-scores retrieved chunks for relevance, improving precision.

**Files:**
- [config/settings.yaml](config/settings.yaml) - `reranker.model`, `reranker.fallback_model`
- [retrieval/reranker.py](retrieval/reranker.py) - Lines 55-56

**Specs:**
- Fast inference
- GPU-accelerated

---

### 7. Faithfulness Verification: `microsoft/deberta-v3-base`
**Purpose:** Verifies that generated answers are grounded in retrieved context (NLI-based).

**Files:**
- [config/settings.yaml](config/settings.yaml) - `verification.faithfulness_model`
- [verification/verifier.py](verification/verifier.py) - Line 53

**Specs:**
- Runs on CPU by default (to free GPU for other models)
- Threshold: 0.65

---

### 8. Image Captioning: `Salesforce/blip2-flan-t5-xl`
**Purpose:** Generates text descriptions of images for indexing and retrieval.

**Files:**
- [config/settings.yaml](config/settings.yaml) - `image_processing.image_captioning_model`
- [wrappers/caption.py](wrappers/caption.py) - Lines 36, 44, 47-48

**Specs:**
- Max caption length: 256 tokens
- Memory-intensive (uses sequential loading to manage VRAM)

---

## ⚙️ Configuration

All models are configured in [config/settings.yaml](config/settings.yaml):

```yaml
models:
  embedding_model: "BAAI/bge-m3"
  image_embedding_model: "openai/clip-vit-large-patch14"

planner:
  local_model: "gemma2:2b"
  fallback_model: "gemma3:4b"
  generator:
    local_model: "llama3:8b"

reranker:
  model: "BAAI/bge-reranker-base"

verification:
  faithfulness_model: "microsoft/deberta-v3-base"

image_processing:
  image_captioning_model: "Salesforce/blip2-flan-t5-xl"
  image_encoder: "openai/clip-vit-large-patch14"

generation:
  openrouter:
    model: "nvidia/nemotron-3-nano-30b-a3b:free"
```

---

## 🚀 Quick Start

### Using API for Generation (Recommended)
```bash
# Set your API key
set OPENROUTER_API_KEY=your_key_here

# Run chat (planner will auto-pull gemma2:2b)
python chat.py
```

### Using Local Ollama for Everything
```bash
# Pull required models
ollama pull gemma2:2b    # Planner
ollama pull llama3:8b    # Generation

# Run chat
python chat.py
```

---

## 📦 Model Sizes

| Model | Size | VRAM Required |
|-------|------|---------------|
| `gemma2:2b` | ~1.6 GB | ~2 GB |
| `llama3:8b` | ~4.7 GB | ~6 GB |
| `bge-m3` | ~1.2 GB | ~2 GB |
| `bge-reranker-base` | ~400 MB | ~1 GB |
| `clip-vit-large-patch14` | ~1.7 GB | ~2 GB |
| `blip2-flan-t5-xl` | ~12 GB | ~14 GB |
| `deberta-v3-base` | ~400 MB | ~1 GB (CPU) |

---

## 🔄 Pipeline Flow

```
User Query
    ↓
[1] Planner (gemma2:2b) → Intent, Modality, Strategy
    ↓
[2] Embedding (bge-m3 / clip-vit) → Query vectors
    ↓
[3] Retrieval (FAISS + BM25) → Candidate chunks
    ↓
[4] Reranker (bge-reranker-base) → Top-k relevant
    ↓
[5] Generation (llama3:8b / OpenRouter) → Grounded answer
    ↓
[6] Verification (deberta-v3) → Faithfulness check
    ↓
Final Answer with Citations
```
