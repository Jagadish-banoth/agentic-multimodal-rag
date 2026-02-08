# 🚀 END-TO-END MULTIMODAL RAG - COMPLETE WORKFLOW
**Step-by-step commands to run the full pipeline with all modalities**

---

## ✅ PREREQUISITES (One-time setup)

```powershell
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Verify Python dependencies
pip install -r requirements.txt

# 3. Pull local LLM model (Ollama - required)
ollama pull llama3:8b

# 4. Optional: Start Redis for caching (performance boost)
# Windows: Download from https://github.com/microsoftarchive/redis/releases
# Or use Docker:
docker run -d -p 6379:6379 --name redis redis:alpine

# OR use WSL:
wsl -d Ubuntu
sudo service redis-server start

# 5. Verify Ollama is running
ollama list
# Should show llama3:8b
```

---

## 📁 STEP 1: PREPARE YOUR DATA

```powershell
# Check current data directory structure
Get-ChildItem -Path data\raw -Recurse

# Expected structure:
# data/raw/
#   ├── text/           # Put PDFs, DOCX, TXT, MD files here
#   ├── image/          # Put JPG, PNG images here
#   ├── audio/          # Put MP3, WAV, M4A files here
#   └── video/          # Put MP4, AVI, MOV files here
```

### Add Sample Data (if you don't have any)

```powershell
# Create sample text document
@"
# Transformer Architecture

The Transformer is a deep learning model introduced in 2017 by Vaswani et al.

## Key Components:
1. Self-attention mechanism
2. Multi-head attention
3. Position-wise feed-forward networks
4. Positional encoding

The model achieves state-of-the-art results on machine translation tasks.
"@ | Out-File -FilePath "data\raw\text\transformer.md" -Encoding utf8

# Verify file created
Get-Content data\raw\text\transformer.md
```

---

## 📥 STEP 2: INGEST ALL MODALITIES

### 2.1 Text Ingestion (PDF, DOCX, TXT, MD)

```powershell
# Ingest all text documents
python -m ingestion.text_ingest

# Expected output:
# - Processed chunks saved to: data/processed/chunks.jsonl
# - Metadata saved to: data/processed/metadata.json
```

**What it does:**
- Loads documents from `data/raw/text/`
- Chunks text (256 tokens, 64 overlap)
- Saves to `data/processed/chunks.jsonl`

### 2.2 Image Ingestion (JPG, PNG) [Optional]

```powershell
# Extract images from PDFs or use standalone images
python -m ingestion.extract_images

# Then process images with CLIP + BLIP2 + OCR
python -m ingestion.image_ingest

# Expected output:
# - Image embeddings and captions
# - OCR text extracted
# - Saved to: data/processed/images.jsonl
```

**What it does:**
- CLIP embeddings (768-dim)
- BLIP2 captions
- Tesseract OCR
- Combines caption + OCR text

### 2.3 Audio Ingestion (MP3, WAV) [Optional]

```powershell
# Transcribe audio with Whisper
python -m ingestion.audio_ingest

# Expected output:
# - Transcripts saved to: data/processed/audio.jsonl
# - Timestamped segments
```

**What it does:**
- Whisper ASR (large-v3)
- Timestamped transcription
- Speaker diarization (optional)

### 2.4 Video Ingestion (MP4, AVI) [Optional]

```powershell
# Extract frames + audio from videos
python -m ingestion.video_ingest_fast

# Expected output:
# - Keyframes extracted
# - Audio transcribed
# - Frame captions generated
# - Saved to: data/processed/video.jsonl
```

**What it does:**
- Keyframe extraction (1 fps)
- Whisper audio transcription
- BLIP2 frame captioning
- Timeline metadata

---

## 🔧 STEP 3: BUILD INDICES

### 3.1 Generate Embeddings and Build FAISS Index

```powershell
# This builds BOTH text and image indices (dual-index mode)
python -m scripts.embed_and_index

# Expected output:
# ✓ Loaded embedding model: BAAI/bge-m3 (1024-dim)
# ✓ Loaded image model: openai/clip-vit-large-patch14 (768-dim)
# ✓ Embedded 150 text chunks
# ✓ Embedded 45 images
# ✓ Built text index: data/index/faiss_text.index
# ✓ Built image index: data/index/faiss_image.index
# ✓ Saved metadata: data/index/meta.jsonl
```

**What it creates:**
- `data/index/faiss_text.index` (BGE-m3, 1024-dim, HNSW)
- `data/index/faiss_image.index` (CLIP, 768-dim, HNSW)
- `data/index/meta.jsonl` (chunk metadata)
- `data/index/bm25.pkl` (sparse index for keyword search)

### 3.2 Verify Indices

```powershell
# Check index files exist
Get-ChildItem data\index

# Expected files:
# faiss_text.index
# faiss_image.index
# meta.jsonl
# bm25.pkl
```

---

## 🧪 STEP 4: TEST RETRIEVAL (Before Full Pipeline)

### 4.1 Test Dense Retrieval

```powershell
# Interactive retrieval demo
python -m scripts.retrieve_demo

# Try queries like:
# - "What is transformer architecture?"
# - "Explain self-attention"
# - "How does BERT work?"
```

**Expected output:**
```
Query: What is transformer architecture?

Top 5 Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[1] Score: 0.847 | transformer.md | Page 1
The Transformer is a deep learning model introduced in 2017...

[2] Score: 0.782 | transformer.md | Page 1
Key Components: 1. Self-attention mechanism...
```

### 4.2 Test Hybrid Retrieval (Dense + Sparse)

```powershell
# Quick retrieval test
python -m scripts.test_retrieval_quick

# Tests:
# - Dense retrieval (FAISS)
# - Sparse retrieval (BM25)
# - Hybrid fusion (RRF)
# - Reranking (Jina)
```

---

## 🎯 STEP 5: RUN FULL PIPELINE (All Modalities)

### 5.1 Single Query Test (Orchestrator)

```powershell
# Run a single query through full pipeline
python -m orchestrator.execution_engine

# Uses default query: "What is self-attention in transformers?"
```

**Pipeline flow:**
```
Query
  → Planner (intent: factual_qa, modalities: text)
  → Retrieval (dense: 20, sparse: 20, rerank: 12)
  → Fusion (12 chunks, 1200 tokens)
  → Generation (llama3:8b)
  → Verification (NLI faithfulness check)
  → Response (answer + sources + confidence)
```

**Expected timing:**
- Planning: 0.5-1.0s
- Retrieval: 0.1-0.3s
- Rerank: 0.2-0.5s (GPU) / 1-2s (CPU)
- Fusion: 0.05s
- Generation: 2-5s
- Verification: 0.3-0.8s (NLI model)
- **Total: 3-8s**

### 5.2 Interactive Chat (Multi-turn)

```powershell
# Interactive chat interface
python chat.py

# Example conversation:
# You> What is a transformer?
# Assistant> [Full answer with sources]
#
# You> How does attention work?
# Assistant> [Retrieves relevant context, generates answer]
#
# Type 'exit' to quit
```

### 5.3 Test Multimodal Queries

```powershell
# If you have images indexed:
python chat.py

# Try queries like:
# - "What's in Figure 3?"
# - "Show me images of neural networks"
# - "What does the diagram explain?"
# - "Find images with text 'Attention'"
```

---

## 📊 STEP 6: EVALUATE PERFORMANCE

### 6.1 Test Cache Performance

```powershell
# Start Redis if not running
docker start redis
# OR: wsl sudo service redis-server start

# Run cache test
python scripts\test_cache.py

# Expected output:
# ✓ Cache initialized
# Pass 1: Cache misses expected
# Pass 2: Cache hits expected
# ✓ HIT: What is machine learning?
# Hit rate: 0.85
```

### 6.2 Run Retrieval Evaluation

```powershell
# Evaluate retrieval quality on dev queries
python -m evaluation.eval_retrieval

# Metrics:
# - Precision@5
# - Recall@10
# - MRR (Mean Reciprocal Rank)
# - NDCG
```

### 6.3 Run RAGAS Evaluation (Answer Quality)

```powershell
# Full pipeline evaluation
python scripts\run_ragas_pipeline.py --input evaluation\ragas_dev.jsonl --output artifacts\predictions.jsonl

# Evaluate answers
python scripts\evaluate_ragas.py --input artifacts\predictions.jsonl --output artifacts\eval_results.jsonl

# Convert to CSV for analysis
python scripts\convert_ragas_results.py --input artifacts\eval_results.jsonl --output artifacts\eval_results.csv

# View results
Import-Csv artifacts\eval_results.csv | Format-Table
```

---

## 🔍 STEP 7: MONITOR & DEBUG

### 7.1 Check Logs

```powershell
# View recent logs
Get-Content logs\text_ingest.log -Tail 50
```

### 7.2 Verify Pipeline Components

```powershell
# Test planner
python -c "from planner.local_agentic_planner import LocalAgenticPlanner; import yaml; cfg = yaml.safe_load(open('config/settings.yaml')); p = LocalAgenticPlanner(cfg); print(p.plan('What is AI?'))"

# Test retriever
python -c "from retrieval.dense_retriever import DenseRetriever; import yaml; cfg = yaml.safe_load(open('config/settings.yaml')); r = DenseRetriever(cfg); print(r.retrieve('AI', top_k=3))"

# Test verifier
python -c "from verification.verifier import Verifier; import yaml; cfg = yaml.safe_load(open('config/settings.yaml')); v = Verifier(cfg); print('NLI model:', v.nli_model)"
```

### 7.3 Check System Resources

```powershell
# GPU usage (if CUDA available)
nvidia-smi

# Memory usage
Get-Process python | Select-Object CPU, WorkingSet64, ProcessName

# Disk space
Get-PSDrive C | Select-Object Used, Free
```

---

## 🎨 STEP 8: TEST SPECIFIC MODALITIES

### Text-only Query

```powershell
python chat.py
# You> Summarize the transformer architecture
# Expected: Retrieves text chunks, generates summary
```

### Image Query (if images indexed)

```powershell
python chat.py
# You> What's shown in the neural network diagram?
# Expected: Retrieves image + caption + OCR, describes content
```

### Multimodal Query

```powershell
python chat.py
# You> Find all mentions of attention mechanism in text and images
# Expected: Hybrid search across text + image indices
```

---

## 🐛 TROUBLESHOOTING

### Issue: "No FAISS index found"

```powershell
# Rebuild indices
python -m scripts.embed_and_index
```

### Issue: "Ollama connection failed"

```powershell
# Check Ollama is running
ollama list

# If not installed:
# Download from: https://ollama.ai
# Then: ollama pull llama3:8b
```

### Issue: "Redis connection failed"

```powershell
# Option 1: Disable cache temporarily
# Edit config/settings.yaml:
# cache:
#   enabled: false

# Option 2: Start Redis
docker run -d -p 6379:6379 redis:alpine

# Option 3: Install Redis for Windows
# https://github.com/microsoftarchive/redis/releases
```

### Issue: "Out of memory during indexing"

```powershell
# Process in smaller batches
# Edit scripts/embed_and_index.py:
# BATCH_SIZE = 16  # Reduce from 32
```

### Issue: "Slow reranking"

```powershell
# Use GPU if available
nvidia-smi

# Or disable reranking temporarily
# Edit config/settings.yaml:
# reranker:
#   model: null  # Disables reranking
```

---

## 📈 PERFORMANCE BENCHMARKS

### Expected Performance (without cache)

| Stage | Latency | Notes |
|-------|---------|-------|
| Planning | 500-1000ms | LLM call |
| Retrieval | 50-200ms | FAISS search |
| Reranking | 200-1000ms | Depends on GPU |
| Fusion | 20-50ms | Token counting |
| Generation | 2-5s | LLM generation |
| Verification | 300-800ms | NLI model |
| **Total** | **3-8s** | First query |

### With Cache (80% hit rate)

| Stage | Latency | Notes |
|-------|---------|-------|
| Embedding cache hit | 2-5ms | Redis lookup |
| Query result cache hit | 5-10ms | Full answer cached |
| **Total (cached)** | **<50ms** | 100x faster |

---

## ✅ SUCCESS CRITERIA

Your pipeline is working correctly if:

1. ✅ Ingestion completes without errors
2. ✅ Index files exist in `data/index/`
3. ✅ Retrieval demo returns relevant results
4. ✅ Chat generates coherent answers with sources
5. ✅ Verification confidence > 0.7
6. ✅ Cache hit rate increases over time
7. ✅ No errors in logs

---

## 🚀 NEXT STEPS

After successful end-to-end run:

1. **Add more documents** to `data/raw/`
2. **Re-run ingestion** and rebuild indices
3. **Test query result cache** (warm up with common queries)
4. **Implement OpenTelemetry** for monitoring
5. **Tune hyperparameters** in `config/settings.yaml`
6. **Run RAGAS evaluation** to measure accuracy
7. **Deploy to production** (Docker + K8s)

---

## 📞 QUICK REFERENCE

### Full End-to-End Command Sequence

```powershell
# 1. Activate environment
.\venv\Scripts\Activate.ps1

# 2. Verify Ollama
ollama list

# 3. Ingest data (adjust for your modalities)
python -m ingestion.text_ingest
# python -m ingestion.image_ingest  # If you have images
# python -m ingestion.audio_ingest  # If you have audio
# python -m ingestion.video_ingest_fast  # If you have video

# 4. Build indices
python -m scripts.embed_and_index

# 5. Test retrieval
python -m scripts.retrieve_demo

# 6. Run full pipeline
python chat.py

# 7. Evaluate
python scripts\run_ragas_pipeline.py --input evaluation\ragas_dev.jsonl --output artifacts\predictions.jsonl --mock
```

**Total time:** 5-15 minutes (depending on data size)

---

**Last Updated:** January 21, 2026  
**Pipeline Status:** ✅ Production-Ready
