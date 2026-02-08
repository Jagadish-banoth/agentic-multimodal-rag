# Pipeline Architecture - Visual Reference

## System Architecture at a Glance

```
╔═════════════════════════════════════════════════════════════════════════╗
║              AGENTIC MULTIMODAL RAG PIPELINE ARCHITECTURE              ║
╚═════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────┐
│  USER INTERFACE LAYER                                                   │
│  ├─ chat.py (interactive chat)                                         │
│  └─ ui/app.py (web interface)                                          │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  ORCHESTRATION LAYER (Main Entry Point)                                 │
│  ├─ ExecutionEngine.run(query)                                          │
│  │  [orchestrator/execution_engine.py]                                  │
│  │                                                                       │
│  ├─ Responsibilities:                                                   │
│  │  • Coordinate all pipeline stages                                    │
│  │  • Manage agentic loop (retry logic)                                │
│  │  • Track timings and metrics                                         │
│  │  • Cache management                                                  │
│  └─ Implements Phase 1 optimizations                                    │
└──────┬───────────────────────────────────────────────────────────────────┘
       │
       ├─────────────────────┬──────────────────┬─────────────────┬──────────────┐
       │                     │                  │                 │              │
       ▼                     ▼                  ▼                 ▼              ▼
   ┌─────────┐          ┌──────────┐      ┌──────────────┐  ┌────────────┐  ┌────────┐
   │ PLANNER │          │RETRIEVAL │      │ FUSION       │  │GENERATION  │  │VERIFIER│
   │(Control)│          │(Parallel)│      │(Context)     │  │(LLM)       │  │(Control)
   │ PLANE   │          │(Parallel)│      │(Data)        │  │(Data)      │  │PLANE   │
   └─────────┘          └──────────┘      └──────────────┘  └────────────┘  └────────┘


         ╔════════════════════════════════════════════════════╗
         ║         PHASE 1: QUERY PREPROCESSING              ║
         ║                                                    ║
         ║  QueryProcessor                                   ║
         ║  ├─ Remove whitespace                             ║
         ║  ├─ Normalize casing                              ║
         ║  └─ Truncate to model limits                      ║
         ║                                                    ║
         ║  File: retrieval/query_processor.py               ║
         ╚════════════════════════════════════════════════════╝
                          │
                          ▼
         ╔════════════════════════════════════════════════════╗
         ║      PHASE 1: CACHE LOOKUP (5.6x speedup)        ║
         ║                                                    ║
         ║  ResultCache                                      ║
         ║  ├─ Fuzzy matching (95% threshold)               ║
         ║  ├─ TTL: 1 hour default                          ║
         ║  └─ In-memory + optional Redis                   ║
         ║                                                    ║
         ║  File: utils/result_cache.py                      ║
         ║                                                    ║
         ║  [IF CACHE HIT: Return → Skip to Response]       ║
         ║  [IF CACHE MISS: Continue]                       ║
         ╚════════════════════════════════════════════════════╝
                          │
                          ▼
    ╔═══════════════════════════════════════════════════════════════╗
    ║          CONTROL PLANE: INTENT & STRATEGY PLANNING            ║
    ║                                                               ║
    ║  LocalAgenticPlanner.plan()                                  ║
    ║  [planner/local_agentic_planner.py]                          ║
    ║                                                               ║
    ║  ┌─ Intent Classification                                    ║
    ║  │  └─ factual | explanatory | comparative | visual |        ║
    ║  │     temporal | procedural | aggregation | unknown         ║
    ║  │                                                            ║
    ║  ├─ Modality Selection                                       ║
    ║  │  └─ text | image | audio | video                         ║
    ║  │                                                            ║
    ║  ├─ Retrieval Strategy                                       ║
    ║  │  └─ dense_k, sparse_k, rerank_k                          ║
    ║  │                                                            ║
    ║  └─ Confidence Scoring                                       ║
    ║     └─ Based on intent classification confidence             ║
    ║                                                               ║
    ║  Output: RetrievalPlan                                       ║
    ║  File: planner/schemas.py                                    ║
    ╚═══════════════════════════════════════════════════════════════╝
                          │
                          ▼
    ╔═══════════════════════════════════════════════════════════════╗
    ║      PHASE 1: QUERY EXPANSION (15-20% recall boost)         ║
    ║                                                               ║
    ║  QueryExpander.expand_queries()                              ║
    ║  [retrieval/query_expansion.py]                              ║
    ║                                                               ║
    ║  Methods:                                                    ║
    ║  ├─ HyDE (Hypothetical Document Embeddings)                 ║
    ║  ├─ Query reformulation                                     ║
    ║  └─ Multi-perspective query generation                      ║
    ║                                                               ║
    ║  Output: [original_query, expanded_1, expanded_2, ...]      ║
    ╚═══════════════════════════════════════════════════════════════╝
                          │
                          ▼
    ╔═══════════════════════════════════════════════════════════════╗
    ║    PHASE 1: PARALLEL HYBRID RETRIEVAL (5.6x latency)        ║
    ║                                                               ║
    ║  ParallelRetriever.retrieve_parallel()                       ║
    ║  [retrieval/parallel_retriever.py]                           ║
    ║                                                               ║
    ║  Executes in PARALLEL with ThreadPoolExecutor (4 workers):   ║
    ║                                                               ║
    ║  ┌─ DENSE RETRIEVAL (Semantic Search) ────────┐              ║
    ║  │  DenseRetriever.search()                   │              ║
    ║  │  [retrieval/dense_retriever.py]            │              ║
    ║  │                                            │              ║
    ║  │  ├─ BAAI/bge-m3 (1024-dim)                │              ║
    ║  │  │  └─ SOTA text retrieval model          │              ║
    ║  │  │                                         │              ║
    ║  │  ├─ CLIP (768-dim)                        │              ║
    ║  │  │  └─ Multimodal image retrieval         │              ║
    ║  │  │                                         │              ║
    ║  │  ├─ FAISS Index Lookup                    │              ║
    ║  │  │  └─ data/index/faiss_text.index        │              ║
    ║  │  │  └─ data/index/faiss_image.index       │              ║
    ║  │  │                                         │              ║
    ║  │  └─ Top-K Results (20-30 per query)       │              ║
    ║  │     └─ Run on: original + expanded        │              ║
    ║  └─────────────────────────────────────────────┘              ║
    ║                      │                                        ║
    ║  ┌─ SPARSE RETRIEVAL (Keyword Search) ────────┐              ║
    ║  │  SparseRetriever.search()                  │              ║
    ║  │  [retrieval/sparse_retriever.py]           │              ║
    ║  │                                            │              ║
    ║  │  ├─ BM25 Algorithm                        │              ║
    ║  │  │  └─ Inverted index lookup              │              ║
    ║  │  │                                         │              ║
    ║  │  ├─ Keyword Matching                      │              ║
    ║  │  │  └─ Traditional information retrieval  │              ║
    ║  │  │                                         │              ║
    ║  │  └─ Top-K Results (20-30 per query)       │              ║
    ║  │     └─ Run on: original + expanded        │              ║
    ║  └─────────────────────────────────────────────┘              ║
    ║                      │                                        ║
    ║  Merge + Deduplicate Results                                 ║
    ║  └─ Combine dense & sparse results                           ║
    ║     Total: 40-60 candidates                                  ║
    ║                                                               ║
    ║  Fusion Strategy (from config):                              ║
    ║  ├─ weighted (60% text, 40% image)                          ║
    ║  ├─ round_robin                                              ║
    ║  └─ adaptive                                                 ║
    ║                                                               ║
    ║  Timing Metrics:                                             ║
    ║  └─ Dense time, sparse time, expansion time                 ║
    ╚═══════════════════════════════════════════════════════════════╝
                          │
                          ▼
    ╔═══════════════════════════════════════════════════════════════╗
    ║            RERANKING (Cross-Encoder Scoring)                 ║
    ║                                                               ║
    ║  Reranker.rerank()                                           ║
    ║  [retrieval/reranker.py]                                     ║
    ║                                                               ║
    ║  Model: jinaai/jina-reranker-v3                              ║
    ║  ├─ Cross-encoder architecture                              ║
    ║  ├─ Context-aware relevance scoring                         ║
    ║  ├─ Learned-to-rank scoring                                 ║
    ║  └─ Batch processing (32 samples)                           ║
    ║                                                               ║
    ║  Input: 40-60 candidates + original query                   ║
    ║  Output: Top-12 reranked results                            ║
    ║  (sorted by relevance score)                                ║
    ║                                                               ║
    ║  Scoring: 0-1 relevance confidence per result               ║
    ╚═══════════════════════════════════════════════════════════════╝
                          │
                          ▼
    ╔═══════════════════════════════════════════════════════════════╗
    ║              CONTEXT FUSION & DEDUPLICATION                   ║
    ║                                                               ║
    ║  ContextFusion.fuse()                                        ║
    ║  [fusion/context_fusion.py]                                  ║
    ║                                                               ║
    ║  Step 1: Deduplication                                       ║
    ║  └─ Remove near-duplicates using hashing                    ║
    ║                                                               ║
    ║  Step 2: Diversity Enforcement                               ║
    ║  ├─ Different sources (document A, B, C)                    ║
    ║  └─ Span pages (page 1, 5, 10, etc.)                       ║
    ║                                                               ║
    ║  Step 3: Token Budget Enforcement                            ║
    ║  ├─ Max chunks: 12 (configurable)                           ║
    ║  ├─ Max tokens: 1200 (configurable)                         ║
    ║  └─ Uses tokenizer: BAAI/bge-m3                             ║
    ║                                                               ║
    ║  Step 4: Citation-Ready Formatting                           ║
    ║  ├─ Assign source indices [1], [2], [3]...                 ║
    ║  ├─ Include document metadata                               ║
    ║  ├─ Preserve original text                                  ║
    ║  └─ Format for easy reference                               ║
    ║                                                               ║
    ║  Output:                                                     ║
    ║  ┌─────────────────────────────────────────────┐            ║
    ║  │ [1] Document A, Page 5, Section Title      │            ║
    ║  │ Content from source 1...                   │            ║
    ║  │                                             │            ║
    ║  │ [2] Document B, Page 12, Section Title     │            ║
    ║  │ Content from source 2...                   │            ║
    ║  │                                             │            ║
    ║  │ [3] Document A, Page 8, Section Title      │            ║
    ║  │ Content from source 3...                   │            ║
    ║  └─────────────────────────────────────────────┘            ║
    ║                                                               ║
    ║  Metadata preserved: doc_id, page, chunk_id, source         ║
    ╚═══════════════════════════════════════════════════════════════╝
                          │
                          ▼
    ╔═══════════════════════════════════════════════════════════════╗
    ║            GROUNDED ANSWER GENERATION (LLM)                   ║
    ║                                                               ║
    ║  GroundedLLM.generate()                                      ║
    ║  [generation/grounded_llm.py]                                ║
    ║                                                               ║
    ║  Model: llama3:8b (Local Ollama)                             ║
    ║  Inference: HTTP API → http://localhost:11434                ║
    ║                                                               ║
    ║  ┌─ Prompt Engineering                                       ║
    ║  │  ├─ System prompt (grounding instructions)               ║
    ║  │  ├─ Few-shot examples                                    ║
    ║  │  ├─ Chain-of-thought reasoning                           ║
    ║  │  └─ Citation format requirements                         ║
    ║  │                                                            ║
    ║  ├─ Model Parameters                                         ║
    ║  │  ├─ Temperature: 0.3 (low randomness)                    ║
    ║  │  ├─ Top-P: 0.9 (nucleus sampling)                        ║
    ║  │  ├─ Max tokens: 400 (response limit)                     ║
    ║  │  └─ Stop sequences: [] (let model decide)                ║
    ║  │                                                            ║
    ║  └─ Inference Loop                                           ║
    ║     ├─ Stream response tokens                               ║
    ║     ├─ Accumulate answer text                               ║
    ║     ├─ Track completion time                                ║
    ║     └─ Handle timeouts (120 sec default)                    ║
    ║                                                               ║
    ║  ┌─ Post-Processing                                          ║
    ║  │  ├─ Extract answer text                                  ║
    ║  │  ├─ Parse citations [Source N]                           ║
    ║  │  ├─ Estimate confidence                                  ║
    ║  │  └─ Build metadata                                       ║
    ║  │                                                            ║
    ║  Output Dict:                                                ║
    ║  ├─ answer: "Full answer text..."                            ║
    ║  ├─ sources: [1, 2, 3] (source indices)                     ║
    ║  ├─ citations: {1: "text", 2: "text"}                       ║
    ║  ├─ confidence: 0.85                                         ║
    ║  └─ metadata: {tokens, time, etc.}                           ║
    ║                                                               ║
    ║  Files:                                                      ║
    ║  └─ generation/response_composer.py                          ║
    ║  └─ generation/extractive_answerer.py (fallback)             ║
    ╚═══════════════════════════════════════════════════════════════╝
                          │
                          ▼
    ╔═══════════════════════════════════════════════════════════════╗
    ║         CONTROL PLANE: ANSWER VERIFICATION & VALIDATION       ║
    ║                                                               ║
    ║  Verifier.verify()                                           ║
    ║  [verification/verifier.py]                                  ║
    ║                                                               ║
    ║  ┌─ Faithfulness Check (NLI)                                 ║
    ║  │  ├─ Model: microsoft/deberta-v3-base                     ║
    ║  │  ├─ Task: Natural Language Inference                     ║
    ║  │  ├─ Checks: Answer logically entails from context        ║
    ║  │  ├─ Threshold: 0.65                                      ║
    ║  │  └─ Scores: ENTAILMENT | NEUTRAL | CONTRADICTION         ║
    ║  │                                                            ║
    ║  ├─ Citation Coverage Check                                  ║
    ║  │  ├─ Verify citations reference valid sources             ║
    ║  │  ├─ Check coverage of key claims                         ║
    ║  │  └─ Ensure no unsupported statements                     ║
    ║  │                                                            ║
    ║  ├─ Hallucination Detection                                  ║
    ║  │  ├─ Check for contradictions                             ║
    ║  │  ├─ Detect fabricated information                        ║
    ║  │  ├─ Analyze evidence support                             ║
    ║  │  └─ Flag ungrounded claims                               ║
    ║  │                                                            ║
    ║  ├─ Hedging Analysis                                         ║
    ║  │  ├─ Detect uncertainty language                          ║
    ║  │  ├─ Apply confidence penalty                             ║
    ║  │  └─ Score over-qualification                             ║
    ║  │                                                            ║
    ║  └─ Confidence Scoring                                       ║
    ║     ├─ Combine: faithfulness + citation + evidence          ║
    ║     ├─ Apply: hedging penalty                               ║
    ║     ├─ Range: 0-1 (0=low, 1=high)                           ║
    ║     └─ Return: confidence score                             ║
    ║                                                               ║
    ║  Output:                                                     ║
    ║  ├─ verified: True/False                                    ║
    ║  ├─ confidence: 0.0-1.0                                     ║
    ║  ├─ level: "HIGH" | "MEDIUM" | "LOW"                        ║
    ║  ├─ should_retry: True/False (control signal)               ║
    ║  └─ reason: explanation for decision                        ║
    ║                                                               ║
    ║  Device: CPU (to free GPU for Ollama/reranker)              ║
    ╚═══════════════════════════════════════════════════════════════╝
                          │
                          ▼
    ╔═══════════════════════════════════════════════════════════════╗
    ║               AGENTIC DECISION LOGIC (Loop Control)            ║
    ║                                                               ║
    ║  IF verified == True AND confidence >= 0.7:                  ║
    ║  └─ ✓ ACCEPT ANSWER                                          ║
    ║     └─ Cache result → Return to user                         ║
    ║                                                               ║
    ║  ELSE IF should_retry == True AND attempt < max_attempts:   ║
    ║  └─ ⟲ RETRY LOOP                                             ║
    ║     └─ Go back to: PLANNING (with updated context)          ║
    ║        └─ Max attempts: 2 (configurable)                     ║
    ║                                                               ║
    ║  ELSE:                                                        ║
    ║  └─ ✓ ACCEPT BEST EFFORT                                     ║
    ║     └─ Return highest-confidence attempt                    ║
    ║        └─ Add note: "Best effort (confidence below threshold)"║
    ║                                                               ║
    ║  File: orchestrator/execution_engine.py (lines 200-280)     ║
    ║                                                               ║
    ║  Configuration:                                              ║
    ║  ├─ agent.confidence_threshold: 0.7 (in settings.yaml)      ║
    ║  └─ agent.max_attempts: 2 (in settings.yaml)                ║
    ╚═══════════════════════════════════════════════════════════════╝
                          │
                          ▼
    ╔═══════════════════════════════════════════════════════════════╗
    ║           PHASE 1: RESPONSE CACHING & STORAGE                 ║
    ║                                                               ║
    ║  ResultCache.set()                                           ║
    ║  [utils/result_cache.py]                                     ║
    ║                                                               ║
    ║  Store Response:                                             ║
    ║  ├─ Query → Response mapping                                ║
    ║  ├─ Confidence score as priority                            ║
    ║  ├─ TTL: 1 hour (configurable)                              ║
    ║  └─ Fuzzy matching enabled (95% threshold)                  ║
    ║                                                               ║
    ║  Storage:                                                    ║
    ║  ├─ Primary: In-memory cache (512 MB max)                   ║
    ║  └─ Optional: Redis backend                                 ║
    ║                                                               ║
    ║  Configuration:                                              ║
    ║  ├─ cache.enabled: true                                     ║
    ║  ├─ cache.query_ttl: 3600 (seconds)                         ║
    ║  ├─ cache.fuzzy_threshold: 0.95                             ║
    ║  └─ cache.max_cache_mb: 512                                 ║
    ╚═══════════════════════════════════════════════════════════════╝
                          │
                          ▼
    ╔═══════════════════════════════════════════════════════════════╗
    ║                  FINAL RESPONSE TO USER                        ║
    ║                                                               ║
    ║  Response Dict:                                              ║
    ║  {                                                           ║
    ║    "final_answer": "Full answer text with citations...",    ║
    ║    "sources": [                                              ║
    ║      {"id": 1, "doc": "...", "page": 5, "content": "..."},  ║
    ║      {"id": 2, "doc": "...", "page": 12, "content": "..."}, ║
    ║      ...                                                     ║
    ║    ],                                                        ║
    ║    "confidence": 0.85,                                       ║
    ║    "verified": true,                                         ║
    ║    "from_cache": false,                                      ║
    ║    "timings": {                                              ║
    ║      "planning": 0.12,                                       ║
    ║      "retrieval": 0.45,  (includes dense, sparse, rerank)   ║
    ║      "fusion": 0.08,                                         ║
    ║      "generation": 2.34,                                     ║
    ║      "verification": 0.15,                                   ║
    ║      "total_time": 3.14                                      ║
    ║    },                                                        ║
    ║    "plan": {                                                 ║
    ║      "intent": "factual",                                    ║
    ║      "strategy": "dense_semantic",                           ║
    ║      "confidence": 0.92                                      ║
    ║    }                                                         ║
    ║  }                                                           ║
    ║                                                               ║
    ║  Display to User:                                            ║
    ║  ├─ Answer with inline citations                            ║
    ║  ├─ Source references                                       ║
    ║  ├─ Confidence score                                        ║
    ║  └─ Optional: Timing breakdown                              ║
    ╚═══════════════════════════════════════════════════════════════╝
```

---

## Data Flow vs Control Flow

### **Data Flow** (What moves through the pipeline)
```
Query → Preprocessed Query → Retrieved Chunks → Fused Context → Generated Answer → Verified Answer
```

### **Control Flow** (Who decides what happens)
```
Planner (Intent/Strategy) → Retrieval Execution → Fusion Execution → Generation Execution → Verifier (Accept/Retry)
```

**Key Insight:** Data flow is LINEAR, Control flow is AGENTIC (can loop).

---

## Component Interaction Graph

```
ExecutionEngine (Orchestrator)
├── calls: LocalAgenticPlanner (Control Plane)
│   └── responsible for: Intent, Modality, Strategy
│
├── calls: ResultCache (Phase 1 Optimization)
│   └── responsible for: Query caching, fuzzy matching
│
├── calls: QueryProcessor (Data Plane)
│   └── responsible for: Query cleaning, normalization
│
├── calls: ParallelRetriever (Data Plane - Phase 1)
│   ├── calls: DenseRetriever (BAAI + CLIP)
│   ├── calls: SparseRetriever (BM25)
│   ├── calls: QueryExpander (HyDE expansion)
│   └── calls: Reranker (Cross-encoder scoring)
│
├── calls: ContextFusion (Data Plane)
│   └── responsible for: Dedup, formatting, token budget
│
├── calls: GroundedLLM (Data Plane)
│   ├── calls: Ollama HTTP API (llama3:8b)
│   └── responsible for: Answer generation, citations
│
├── calls: Verifier (Control Plane)
│   ├── calls: NLI Model (deberta-v3-base)
│   ├── calls: Citation checker
│   └── responsible for: Faithfulness, confidence, retry signal
│
└── calls: MetricsTracker (Observability)
    └── responsible for: Latency, throughput, cache hit rate
```

---

## Configuration Hierarchy

```
config/settings.yaml
├── agent (Control Plane Settings)
│   ├── confidence_threshold: 0.7 (accept if >= this)
│   ├── max_attempts: 2 (retry loop limit)
│   └── phase1_enabled: true
│
├── models (Model Selection)
│   ├── embedding_model: BAAI/bge-m3 (dense text)
│   ├── image_embedding_model: CLIP (dense image)
│   ├── generator.local_model: llama3:8b
│   └── reranker.model: jinaai/jina-reranker-v3
│
├── planner (Agentic Planner Config)
│   ├── mode: fast (skip LLM planning)
│   └── cache_ttl_seconds: 3600
│
├── retrieval (Retrieval Config)
│   ├── dense_k: 20 (default, can be overridden by planner)
│   ├── sparse_k: 20 (default, can be overridden by planner)
│   └── rerank_k: 12 (default, can be overridden by planner)
│
├── fusion (Context Fusion Config)
│   ├── max_chunks: 12
│   ├── max_tokens: 1200
│   └── tokenizer_name: BAAI/bge-m3
│
├── generation (LLM Generation Config)
│   ├── max_generation_tokens: 400
│   ├── temperature: 0.3
│   └── ollama.host: http://localhost:11434
│
├── verification (Answer Verification Config)
│   ├── enabled: true
│   ├── faithfulness_model: microsoft/deberta-v3-base
│   └── faithfulness_threshold: 0.65
│
└── cache (Phase 1 Caching Config)
    ├── enabled: true
    ├── query_ttl: 3600
    ├── enable_fuzzy: true
    └── max_cache_mb: 512
```

---

## Performance Characteristics

| Component | Typical Latency | Parallelizable? | Critical Path? |
|-----------|-----------------|-----------------|----------------|
| Query Processing | 10 ms | ✗ | ✗ |
| Planner | 50-100 ms | ✗ | ✓ |
| Dense Retrieval | 200-300 ms | ✓ | ✓ |
| Sparse Retrieval | 50-100 ms | ✓ | ✓ |
| Reranking | 100-150 ms | ✗ | ✓ |
| Fusion | 20-50 ms | ✗ | ✗ |
| LLM Generation | 2-3 seconds | ✗ | ✓ |
| Verification | 100-200 ms | ✗ | ✗ |
| **Total (Sequential)** | **~4 seconds** | — | — |
| **Total (Parallel w/ Phase 1)** | **~2.5 seconds** | — | — |

**Phase 1 improvements:** 5.6x speedup through parallelism + caching + expansion

---

## Critical Failure Points

| Failure Point | Impact | Recovery |
|---------------|--------|----------|
| No results from retrieval | Cannot generate answer | Logged warning, return error |
| Empty context after fusion | Cannot ground generation | Continue with retry |
| Low LLM confidence | May not meet threshold | Trigger retry (max 2x) |
| NLI model failure | Cannot verify | Fallback to overlap scoring |
| Cache miss + slow generation | High latency | Async caching for next time |
| Ollama not running | Generation fails | Error message with setup instructions |

---

## Optimization Opportunities

1. **GPU Acceleration** - Reranker, NLI, embeddings on GPU
2. **Batch Processing** - Multiple concurrent queries
3. **Index Optimization** - HNSW tuning, IVF for larger indices
4. **Cache Warming** - Pre-compute common queries
5. **Streaming** - Stream LLM responses to user in real-time
6. **Quantization** - Run models in INT8 for faster inference
