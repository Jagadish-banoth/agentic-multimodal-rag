# Agentic Multimodal RAG — Engineering Report 

**Repo:** `agentic-multimodal-rag`
**Generated:** 2026-02-06
**Audience:** engineering + applied-ML reviewers (design / ops / evaluation)

---

## 0) Executive Summary (Main Points)

### What this system is

This project is an **agentic, multimodal Retrieval-Augmented Generation (RAG) system** that answers user questions by (1) planning a retrieval strategy, (2) running hybrid retrieval over indexed corpora (dense + sparse; optionally multimodal), (3) fusing evidence into a citation-ready context, (4) generating a grounded answer with citations, and (5) verifying faithfulness using strict citation checks plus an NLI verifier.

### Why it matters

Compared to “static” RAG, this pipeline is designed to be **self-correcting**: if verification confidence is low, it retries up to a configured budget. It also incorporates **Phase-1 performance optimizations** (parallel retrieval, query expansion, caching) to keep latency acceptable on local hardware.

### Key technical differentiators

- **Control plane / Data plane split** implemented in code (planner + verifier vs retrieval + fusion + generation).
- **Hybrid retrieval**: FAISS dense (BGE-m3) + BM25 sparse, merged by Reciprocal Rank Fusion (RRF).
- **Reranking** via cross-encoder (default `BAAI/bge-reranker-base`) with GPU support.
- **Multimodal ingestion** (image/audio/video) producing text representations for indexing plus optional CLIP image index.
- **Machine-checkable grounding**: evidence chunks are assigned stable citation IDs of the form **`[filename#chunk_id]`** with a `chunk_map` used downstream by verifier.

### Current evaluation snapshot (from artifacts)

From `artifacts/full_eval_results_summary.json` (5 queries):

- Retrieval: **Recall@1 = 0.80**, **Recall@5 = 1.00**, **Recall@10 = 1.00**, **MRR = 0.84**
- Generation: **ROUGE-1 = 0.3528**, **ROUGE-L = 0.3354**, **BLEU(avg) = 0.1793**
- BERTScore: **F1 = 0.0849** (note: precision is negative; see evaluation notes)
- Latency: **avg_time_per_query = 49.87s** (end-to-end evaluation run)

From `artifacts/comprehensive_test_results.json` (5 queries):

- **Success rate = 1.0**
- **Average confidence = 0.5676**
- **Faithful rate = 0.0** in that artifact (likely due to citation-format mismatch; see “Known Issues / Gaps”).

### What’s “FAANG-grade” here vs what’s missing

Strong:

- Modular system design, robust fallbacks, strict citation verification design, and reproducible scripts.
- Clear configuration surface in `config/settings.yaml`.

Missing / opportunities:

- True production observability (distributed tracing, metrics export) beyond JSONL logs.
- Larger evaluation sets + consistent evaluation definitions across scripts.
- Latency reductions (planner + reranker + generation dominating time on commodity hardware).

---

## 1) Objectives, Non-Goals, and Success Criteria

### Objectives

1. **Grounded QA over private corpora**: produce answers supported by retrieved evidence.
2. **Reduce hallucinations**: enforce citation discipline and reject/abstain when evidence is insufficient.
3. **Support multimodal corpora**: ingest text, images, audio, and video into a unified retrieval surface.
4. **Agentic robustness**: planner selects strategy; verifier triggers retry when confidence is low.
5. **Practical local-first execution**: default to Ollama (local models) with optional OpenRouter.

### Non-goals (current scope)

- Serving as a high-QPS hosted API out of the box (the repo ships Streamlit UI + CLI chat; HTTP API is only a skeleton for load tests).
- Learning-to-rank training or supervised fine-tuning.
- Fully featured knowledge graph retrieval (planner includes `GRAPH_ENHANCED` strategy, but no graph index is present in the current codebase).

### Success criteria (suggested)

- Retrieval: RecallHit@5 ≥ 0.80, MRR ≥ 0.50, nDCG@10 ≥ 0.60 (see `evaluation/slo.yaml`).
- Generation: high citation coverage; low verifier false-accept rate.
- Latency: pipeline p95 under 8s on target hardware (**SLO gate is defined in milliseconds**; see `evaluation/slo.yaml` + `scripts/check_slos.py`).

---

## 2) System Architecture

### 2.1 High-level flow

```text
User Query
  -> ExecutionEngine (orchestrator)
       -> QueryProcessor (normalize whitespace, preserve case)
       -> ResultCache (optional, fuzzy match)
       -> Planner (intent, modality, strategy, k-values)
       -> QueryExpander (variants + HyDE) [optional]
       -> Retrieval (dense + sparse) [parallel]
       -> Merge (RRF)
       -> Reranker (cross-encoder)
       -> ContextFusion (dedup + MMR + token budget + cite IDs)
       -> GroundedLLM (generate answer, extract citations, build evidence)
       -> Verifier (strict citation checks + NLI)
       -> Accept / Retry / Best-effort
```

### 2.2 Control Plane vs Data Plane

**Control plane** (decides what to do):

- `planner/agentic_planner.py`: intent + strategy selection; rule-based fallback; plan caching.
- `verification/verifier.py`: validates citations, runs NLI checks, returns accept/retry decision.

**Data plane** (does the work):

- `retrieval/*`: query expansion, parallel hybrid retrieval, reranking.
- `fusion/context_fusion.py`: evidence selection + formatting.
- `generation/grounded_llm.py`: grounded generation (OpenRouter/Ollama).
- `ingestion/*` + `scripts/embed_and_index.py`: offline data ingestion, embedding, indexing.

### 2.3 Phase-1 performance optimizations (implemented)

- **ParallelRetriever** (`retrieval/parallel_retriever.py`): retrieves dense/sparse for multiple expanded queries concurrently.
- **QueryExpander** (`retrieval/query_expansion.py`): deterministic variants + optional HyDE via Ollama.
- **ResultCache** (`utils/result_cache.py`): exact + fuzzy caching with Redis optional.
- **Embedding cache** (referenced by dense retriever as optional): reduces repeated embedding computations.

Default agent loop parameters (from `config/settings.yaml`):

- `agent.confidence_threshold = 0.7`
- `agent.max_attempts = 2`

---

## 3) Core Components (Deep Dive)

### 3.1 Orchestrator

**File:** `orchestrator/execution_engine.py`

Responsibilities:

- Owns the end-to-end pipeline and retry loop.
- Applies Phase-1 optimization modules if enabled.
- Produces structured response with timings, plan, verification.

Key behavior:

- Calls `ContextFusion.fuse_with_mapping(...)` to get **`context` + `chunk_map`**.
- Calls `GroundedLLM.generate(query, context, results, chunk_map=chunk_map)`.
- Calls `Verifier.verify(query, response, results, chunk_map=chunk_map)`.
- Accepts response when `verified == True` and caches it (if enabled).

### 3.2 Planner

**Files:**

- `planner/agentic_planner.py` (primary)
- `planner/schemas.py`
- `planner/local_agentic_planner.py` (alternative heavy LLM planner)

Planner outputs a `RetrievalPlan` with:

- intent: factual/explanatory/comparative/... (enum)
- modalities: text/image/audio/video
- strategy: dense/sparse/hybrid/... (enum)
- dynamic params: `dense_k`, `sparse_k`, `rerank_k`
- confidence score

Implementation notes:

- `AgenticPlanner` is designed to use **Gemma 2B** (via Ollama) for medium+ complexity queries and a rule-based/fast path for simple ones.
- It includes an in-memory TTL cache (`ThreadSafePlanCache`) to avoid repeated planning costs.

### 3.3 Retrieval

#### 3.3.1 Query preprocessing

**File:** `retrieval/query_processor.py`

- Removes extra whitespace.
- Explicitly **preserves case** (important for acronyms / proper nouns).
- Optional tokenizer truncation to avoid extreme prompts.

#### 3.3.2 Query expansion

**File:** `retrieval/query_expansion.py`

- Deterministic variants (formal / technical / simplified).
- Optional HyDE: generates “hypothetical passages” using Ollama.
- Expansion is cached in-memory by query.

#### 3.3.3 Parallel hybrid retrieval

**File:** `retrieval/parallel_retriever.py`

- Runs retrieval for multiple queries concurrently.
- Runs dense and sparse retrieval concurrently for each query.
- Merges results and reranks the merged candidate set.

#### 3.3.4 Dense retrieval (FAISS)

**File:** `retrieval/dense_retriever.py`

- Supports **dual-index** mode:
  - Text index: BGE-m3 (1024-dim)
  - Image index: CLIP (768-dim)
- Combines text + image index results with **weighted fusion** (weights are configurable in `config/settings.yaml`).
- Loads metadata from `data/index/meta.jsonl` and full chunk content from `data/processed/chunks.jsonl`.
- Uses `utils/dual_embedder.py` to encode queries for text and optionally CLIP text embeddings.

#### 3.3.5 Sparse retrieval (BM25)

**File:** `retrieval/sparse_retriever.py`

- Loads pickled BM25 index artifacts:
  - `data/index/bm25_index.pkl`
  - `data/index/bm25_corpus.pkl`
- Simple regex tokenization; returns results with content + metadata.

#### 3.3.6 Reranking

**File:** `retrieval/reranker.py`

- Primary model: `BAAI/bge-reranker-base` (configurable).
- GPU if available; CPU fallback.
- Timeout-based mechanism exists (used when primary differs from fallback).
- In the Phase-1 path, `ParallelRetriever` passes the planner’s `rerank_k` as `top_n` to the reranker (so `rerank_k` is the effective output size).
- Outputs `rerank_score` used by fusion for sorting.

### 3.4 Context Fusion

**File:** `fusion/context_fusion.py`

Key properties:

- Dedup by `chunk_id`.
- Sort by `rerank_score` if present.
- Diversity enforcement via MMR when possible.
- Enforces token budget, truncates overly long chunks.

Grounding design:

- Assigns stable citation IDs: **`filename#chunk_id`**.
- Returns `chunk_map: Dict[cite_id -> chunk_fields]`.
- Formats evidence blocks with citations embedded directly as `[{cite_id}] ...`.

### 3.5 Grounded generation

**File:** `generation/grounded_llm.py`

Backends:

- `generation.provider = auto|openrouter|ollama` (see `config/settings.yaml`).
- “Auto” will use OpenRouter if `OPENROUTER_API_KEY`/`OP_TOKEN` exists; otherwise uses local Ollama.

Grounding behavior:

- Prompt requires citations for each claim.
- Extracts citations from answer text (bracket parsing) and cross-validates via `chunk_map`.
- Builds a structured `evidence` list of `{claim, cite_ids}` for verification.

### 3.6 Verification

**File:** `verification/verifier.py`

Checks:

1. **Strict citation validity**: citations must exist in `chunk_map`.
2. **Citation coverage ratio**: fraction of sentences with citations.
3. **NLI faithfulness**: entails/supports checks using `microsoft/deberta-v3-base` by default.
4. **Confidence computation**: combines evidence score + citation score + hedging penalty.

Output:

- `verified: bool`, `confidence: float`, and `should_retry: bool` to drive the agentic loop.

---

## 4) Data / Indexing / Offline Pipeline

### 4.1 Text ingestion

**File:** `ingestion/text_ingest.py`

- Parses PDFs via PyMuPDF, plus TXT/DOCX/PPTX/HTML/CSV/XLSX/JSON.
- Optional OCR for scanned PDFs (Tesseract if available).
- Token-aware chunking and overlap.
- Output: `data/processed/chunks.jsonl` (+ `chunks_index.json`).

### 4.2 Image ingestion

**File:** `ingestion/image_ingest.py`

- Uses a `MultimodalImageProcessor` wrapper to combine:
  - CLIP encodings
  - BLIP-2 captions
  - OCR text
- Writes chunk-like records into the same `chunks.jsonl` stream.
- Stores embeddings in `data/processed/image_embeddings.npz` for indexing.

### 4.3 Audio ingestion

**File:** `ingestion/audio_ingest.py`

- Whisper-based transcription (default size “medium” inside the module).
- Writes transcript to chunks with modality `audio` and `content=transcript`.

### 4.4 Video ingestion (fast)

**File:** `ingestion/video_ingest_fast.py`

- Smart keyframe extraction, batch captioning, conditional OCR.
- Whisper transcription with smaller model for speed.
- Designed for “good enough” multimodal indexing under local compute.

### 4.5 Embedding + indexing

**File:** `scripts/embed_and_index.py`
Creates:

- Dense FAISS indices (HNSW by default):
  - `data/index/faiss_text.index`
  - `data/index/faiss_image.index` (only if images exist and not in text-only mode)
- Sparse BM25 artifacts:
  - `data/index/bm25_index.pkl`
  - `data/index/bm25_corpus.pkl`
- Metadata manifest: `data/index/meta.jsonl`

---

## 5) Interfaces (How to Use)

### 5.1 CLI chat

**File:** `chat.py`

```powershell
# In PowerShell
.\venv\Scripts\Activate.ps1
ollama serve   # keep running in another window
python chat.py
```

### 5.2 Streamlit UI

**File:** `ui/app.py`

```powershell
.\venv\Scripts\Activate.ps1
streamlit run ui/app.py
```

- Upload file → ingestion runs → indexing runs → chat becomes available.

### 5.3 End-to-end workflow (text-only minimal)

1) Put PDFs/TXT/MD in `data/raw/text/`
2) Run ingestion:

```powershell
python -m ingestion.text_ingest
```

3) Build indices:

```powershell
python -m scripts.embed_and_index
```

4) Ask questions:

```powershell
python chat.py
```

---

## 6) Configuration Surface

**File:** `config/settings.yaml`

Key knobs:

- `agent.confidence_threshold`, `agent.max_attempts`
- Planner:
  - `planner.mode`, `planner.local_model`, `planner.fallback_model`, `planner.llm_timeout`
- Retrieval:
  - `retrieval.dense_k`, `retrieval.sparse_k`, `retrieval.rerank_k`
- Reranker:
  - `reranker.model`, `reranker.timeout_seconds`, `reranker.use_gpu`, `reranker.top_n`
- Fusion:
  - `fusion.max_chunks`, `fusion.max_tokens`, `fusion.use_mmr`
- Generation:
  - `generation.provider` (auto/openrouter/ollama)
  - `generation.openrouter.model`
  - `generation.ollama.host`
- Verification:
  - `verification.enabled`, `verification.faithfulness_model`, `verification.faithfulness_threshold`, `verification.device`

Selected defaults (from `config/settings.yaml`):

- Agent: `confidence_threshold=0.7`, `max_attempts=2`
- Planner: `mode=agentic`, `local_model=gemma2:2b`, `fallback_model=gemma3:4b`, `llm_timeout=5s`
- Retrieval depth: `dense_k=100`, `sparse_k=100`, `rerank_k=50`
- Reranker: `model=BAAI/bge-reranker-base`, `timeout_seconds=30`, `use_gpu=true`
- Fusion: `max_chunks=12`, `max_tokens=1200`
- Generation: `provider=auto` (OpenRouter if configured, else Ollama), `ollama.host=http://localhost:11434`
- Verification: `faithfulness_model=microsoft/deberta-v3-base`, `faithfulness_threshold=0.65`, `device=cpu`

---

## 7) Evaluation: Metrics, Scores, and Interpretation

### 7.1 What is measured

- Retrieval quality: Recall@K, MRR (see `evaluation/eval_retrieval.py` and `evaluation/metrics.py`).
- End-to-end RAG metrics (custom + optional official RAGAS): ROUGE, BLEU, BERTScore (see `evaluation/ragas.py`).
- SLO gates: `evaluation/slo.yaml` + `scripts/check_slos.py`.

### 7.2 Current metric scores (from artifacts)

**Source:** `artifacts/full_eval_results_summary.json`

| Category   |             Metric |  Score |
| ---------- | -----------------: | -----: |
| Retrieval  |           Recall@1 |   0.80 |
| Retrieval  |           Recall@5 |   1.00 |
| Retrieval  |          Recall@10 |   1.00 |
| Retrieval  |                MRR |   0.84 |
| Generation |            ROUGE-1 | 0.3528 |
| Generation |            ROUGE-2 | 0.2863 |
| Generation |            ROUGE-L | 0.3354 |
| Generation |           BLEU-avg | 0.1793 |
| Semantic   |       BERTScore F1 | 0.0849 |
| Runtime    | Avg time/query (s) |  49.87 |

Notes:

- BERTScore precision being negative indicates the underlying model/scaling being used may not be calibrated for this dataset, or that evaluation text formatting differs (e.g., citations included vs not). Treat this metric carefully unless standardized preprocessing is applied.
- The evaluation set is small (`total_queries = 5`), so metrics are high-variance.

### 7.3 Pipeline performance breakdown (from artifacts)

**Source:** `artifacts/comprehensive_test_results.json` aggregated

- `n = 5`
- success_rate = 1.00
- avg_confidence = 0.5676
- faithful_rate = 0.00
- avg_total_time_s = 320.66

Average per-stage timings (seconds):

- planning: 29.97
- retrieval: 2.95
- rerank: 66.46
- fusion: 0.57
- generation: 53.67

P95 per-stage timings (seconds):

- planning: 34.83
- retrieval: 7.12
- rerank: 130.96
- fusion: 2.01
- generation: 91.44

Interpretation:

- The dominant costs are **planner**, **reranker**, and **generation**.
- Retrieval (FAISS/BM25) and fusion are relatively cheap.

Important note on latency numbers:

- `artifacts/full_eval_results_summary.json` reports `avg_time_per_query` for a specific evaluation run.
- `artifacts/comprehensive_test_results.json` contains a different per-query timing breakdown (much larger totals), and should be treated as a separate run/profile.
- The SLO gate in `evaluation/slo.yaml` is defined in **milliseconds** and is checked by `scripts/check_slos.py` against dedicated benchmark artifacts (e.g., `artifacts/retrieval_benchmark.json`, `artifacts/pipeline_profile.json`).

---

## 8) Known Issues / Gaps (Observed)

### 8.1 Citation format mismatch across modules / artifacts

The system’s *intended* citation format is **`[filename#chunk_id]`** (fusion assigns `cite_id` and builds `chunk_map`).
However, at least one historical artifact (`artifacts/comprehensive_test_results.json`) contains answers with citations like `[CHUNK 1]` or `[Source 1]`, which the strict verifier will not recognize as valid. This likely explains the `faithful=false` fields and `faithful_rate = 0.0` in that artifact.

**Recommendation:** standardize all prompts and all test scripts to use **only** `filename#chunk_id` citations, and optionally add backward-compat parsing in verifier if you must support legacy formats.

### 8.2 Small evaluation set

Current artifact summaries are computed over 5 queries. This is useful for smoke tests but insufficient for robust claims.

### 8.3 Production observability

Logs exist, plus JSONL telemetry (`logs/telemetry.jsonl`), but there is no full tracing/metrics export.

---

## 9) Pros, Cons, Advantages, Limitations

### Pros / Advantages

- **Modular architecture** with clear responsibility boundaries.
- **Local-first**: works offline with Ollama (privacy-friendly, cost control).
- **Strong retrieval stack**: dense + sparse + rerank is the standard for high-quality RAG.
- **Grounding-first design**: stable citation IDs + strict verifier.
- **Multimodal ingestion**: image OCR/captioning, audio/video transcription and captions.
- **Config-driven**: can swap models and thresholds without code edits.

### Cons / Limitations

- **Latency** can be high on commodity hardware, especially for reranking and generation.
- **Dependency weight** is significant (PyTorch + transformers + multimodal stack).
- **Evaluation consistency** needs tightening (metrics differ across scripts; citation formatting matters).
- **Graph retrieval** appears in planning schema but isn’t implemented as an index/engine.
- **No production API** included (Streamlit UI is good for demos; API is left to integrator).

---

## 10) Main Files (What They Do)

### Entry points

- `chat.py` — CLI interactive chat.
- `ui/app.py` — Streamlit UI for upload + chat.
- `orchestrator/execution_engine.py` — end-to-end orchestrator and agentic loop.

### Control plane

- `planner/agentic_planner.py` — main planner (Gemma-based with caching + fallback).
- `planner/schemas.py` — typed plan schemas.
- `verification/verifier.py` — strict citations + NLI verifier.

### Retrieval / Fusion / Generation

- `retrieval/parallel_retriever.py` — parallel retrieval + expansion + rerank.
- `retrieval/dense_retriever.py` — FAISS retrieval (dual-index).
- `retrieval/sparse_retriever.py` — BM25 retrieval.
- `retrieval/reranker.py` — cross-encoder reranking.
- `fusion/context_fusion.py` — dedup/MMR/token budget + stable cite IDs.
- `generation/grounded_llm.py` — grounded generation (OpenRouter/Ollama).

### Ingestion / Indexing

- `ingestion/text_ingest.py` — PDF/doc parsing + chunking.
- `ingestion/image_ingest.py` — caption + OCR + embeddings.
- `ingestion/audio_ingest.py` — whisper ASR.
- `ingestion/video_ingest_fast.py` — keyframes + caption + ASR.
- `scripts/embed_and_index.py` — build FAISS + BM25.

### Evaluation / Ops

- `evaluation/ragas.py` — custom + optional RAGAS framework metrics.
- `evaluation/eval_retrieval.py` — recall/MRR evaluation.
- `evaluation/slo.yaml` + `scripts/check_slos.py` — SLO gates.
- `monitoring/jsonl_metrics.py` — JSONL telemetry writer.

---

## 11) Getting Started (Recommended Path)

### Quick (text-only) start

1) Setup

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull llama3:8b
ollama serve
```

2) Put docs in `data/raw/text/`
3) Ingest + index

```powershell
python -m ingestion.text_ingest
python -m scripts.embed_and_index
```

4) Chat

```powershell
python chat.py
```

### Multimodal start (optional)

- Images: `python -m ingestion.image_ingest`
- Audio: `python -m ingestion.audio_ingest`
- Video: `python -m ingestion.video_ingest_fast`
  Then re-run indexing.

---

## 12) Recommended Next Steps (Engineering Roadmap)

1) **Evaluation hardening**

- Increase dev set size (≥ 100 queries) and ensure consistent preprocessing.
- Report confidence calibration and verifier acceptance rates.

2) **Latency optimization**

- Reduce planner overhead (rule-based for more cases; cache; smaller model).
- Use reranker cascade (fast bi-encoder filter → cross-encoder top-50 → top-5).
- Stream generation outputs in UI.

3) **Observability**

- Add request IDs and structured JSON logs across modules.
- Add OpenTelemetry spans for planner/retrieval/rerank/fusion/generation/verify.

4) **Correctness**

- Enforce a single canonical citation format everywhere (`[filename#chunk_id]`).

---

## Appendix A — Artifacts referenced

- `artifacts/full_eval_results_summary.json`
- `artifacts/comprehensive_test_results.json`

## Appendix B — SLO gates

See `evaluation/slo.yaml` and `scripts/check_slos.py`.
