# Agentic Multimodal RAG Pipeline - Complete Flow Guide

## High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY INPUT                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
          ┌────────────────────────────────┐
          │  ExecutionEngine.run(query)    │  [orchestrator/execution_engine.py]
          │  Main Orchestrator             │
          └────────────┬───────────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         ▼             ▼             ▼
    CONTROL PLANE (DECISION)     DATA PLANE (EXECUTION)
```

---

## Pipeline Execution Flow (Step-by-Step)

### **PHASE 0: QUERY PREPROCESSING**
```
User Input (raw query)
    ↓
QueryProcessor.preprocess(query)  [retrieval/query_processor.py]
    • Remove extra whitespace
    • Normalize text (lowercase)
    • Truncate to model limits
    ↓
Preprocessed Query
```

**File: [retrieval/query_processor.py](retrieval/query_processor.py)**

---

### **PHASE 1: CACHE LOOKUP** (Phase 1 Optimization)
```
Preprocessed Query
    ↓
ResultCache.get(query)  [utils/result_cache.py]
    • Check fuzzy matching (95% threshold)
    • Return if cached + within TTL
    ↓
[IF CACHE HIT] Return cached response → Skip to Step 6
[IF CACHE MISS] Continue to Step 2
```

**File: [utils/result_cache.py](utils/result_cache.py)**

---

### **PHASE 2: AGENTIC PLANNING** (Control Plane)
```
Preprocessed Query
    ↓
LocalAgenticPlanner.plan(query)  [planner/local_agentic_planner.py]
    
    ├─ Intent Classification
    │  └─ Determines: factual | explanatory | comparative | visual_reasoning | etc.
    │
    ├─ Modality Selection
    │  └─ Selects: text | image | audio | video
    │
    ├─ Retrieval Strategy Planning
    │  └─ Plans: dense_k, sparse_k, rerank_k values
    │
    └─ Returns: RetrievalPlan (with confidence, intent, modalities, strategy)
```

**File: [planner/local_agentic_planner.py](planner/local_agentic_planner.py)**
**Schema: [planner/schemas.py](planner/schemas.py)**

**Output Example:**
```python
RetrievalPlan(
    intent=QueryIntent.FACTUAL,
    modalities=[Modality.TEXT],
    strategy=RetrievalStrategy.DENSE_SEMANTIC,
    dense_k=20,
    sparse_k=20,
    rerank_k=12,
    confidence=0.85
)
```

---

### **PHASE 3: QUERY EXPANSION** (Phase 1 Optimization)
```
Query + Plan
    ↓
QueryExpander.expand_queries(query)  [retrieval/query_expansion.py]
    • Uses HyDE (Hypothetical Document Embeddings)
    • Generates alternative query formulations
    • Improves recall by 15-20%
    ↓
Original Query + Expanded Queries
```

**File: [retrieval/query_expansion.py](retrieval/query_expansion.py)**

---

### **PHASE 4: PARALLEL HYBRID RETRIEVAL** (Data Plane - Phase 1 Optimization)

The system uses **parallel execution** for 5.6x latency improvement:

```
Original + Expanded Queries
    ↓
ParallelRetriever.retrieve_parallel()  [retrieval/parallel_retriever.py]
    
    ├─ DENSE RETRIEVAL (Parallel)
    │  ├─ DenseRetriever.search()  [retrieval/dense_retriever.py]
    │  │  • BAAI/bge-m3 (1024-dim) for text
    │  │  • CLIP (768-dim) for images
    │  │  • FAISS index lookup
    │  │  • Top-K retrieval
    │  │  ↓
    │  │  Returns: top 20-30 dense results
    │  │
    │  └─ (Also runs on expanded queries)
    │
    ├─ SPARSE RETRIEVAL (Parallel)
    │  ├─ SparseRetriever.search()  [retrieval/sparse_retriever.py]
    │  │  • BM25 keyword search
    │  │  • Inverted index
    │  │  • Top-K retrieval
    │  │  ↓
    │  │  Returns: top 20-30 sparse results
    │  │
    │  └─ (Also runs on expanded queries)
    │
    └─ Fusion of Results
       └─ Deduplicates and merges dense + sparse results
          ↓
          Returns: 40-60 candidates
```

**Files:**
- [retrieval/dense_retriever.py](retrieval/dense_retriever.py)
- [retrieval/sparse_retriever.py](retrieval/sparse_retriever.py)
- [retrieval/parallel_retriever.py](retrieval/parallel_retriever.py)

---

### **PHASE 5: RERANKING** (Data Plane)

```
40-60 Candidates
    ↓
Reranker.rerank()  [retrieval/reranker.py]
    • Uses cross-encoder: jinaai/jina-reranker-v3
    • Scores relevance in context of query
    • Applies learned-to-rank scoring
    ↓
Top-12 Reranked Results (sorted by relevance)
```

**File: [retrieval/reranker.py](retrieval/reranker.py)**

---

### **PHASE 6: CONTEXT FUSION** (Data Plane)

```
Top-12 Results
    ↓
ContextFusion.fuse()  [fusion/context_fusion.py]
    
    ├─ Deduplication
    │  └─ Remove near-duplicates based on hashing
    │
    ├─ Diversity Enforcement
    │  └─ Ensure results from different sources/pages
    │
    ├─ Token Budget Enforcement
    │  └─ Fit context within 1200 tokens (configurable)
    │
    └─ Format for Generation
       └─ Create citation-ready structured context
       
    ↓
Formatted Context String (with source metadata)
```

**File: [fusion/context_fusion.py](fusion/context_fusion.py)**

**Output Example:**
```
[1] Document A, Page 5, Section "Introduction"
Content of source 1...

[2] Document B, Page 12, Section "Methods"
Content of source 2...

[3] Document A, Page 8, Section "Results"
Content of source 3...
```

---

### **PHASE 7: GROUNDED GENERATION** (Data Plane)

```
Query + Formatted Context
    ↓
GroundedLLM.generate()  [generation/grounded_llm.py]
    
    ├─ Prompt Engineering
    │  ├─ System prompt with grounding instructions
    │  ├─ Few-shot examples
    │  ├─ Chain-of-thought reasoning
    │  └─ Citation format requirements
    │
    ├─ Local Ollama Inference
    │  ├─ Model: llama3:8b
    │  ├─ Temperature: 0.3 (low randomness)
    │  ├─ Max tokens: 400
    │  └─ HTTP API call to localhost:11434
    │
    └─ Response Extraction & Parsing
       ├─ Extract answer text
       ├─ Extract citations [Source N]
       ├─ Confidence estimation
       └─ Generate metadata
    
    ↓
Answer Dict {
    "answer": "Full answer text...",
    "sources": [list of source indices],
    "confidence": 0.85,
    "citations": {1: "source content", 2: "source content", ...}
}
```

**File: [generation/grounded_llm.py](generation/grounded_llm.py)**

---

### **PHASE 8: VERIFICATION** (Control Plane - Agentic Loop)

```
Query + Generated Answer + Retrieved Context
    ↓
Verifier.verify()  [verification/verifier.py]
    
    ├─ Faithfulness Check
    │  ├─ NLI (Natural Language Inference) model
    │  ├─ microsoft/deberta-v3-base
    │  ├─ Checks if answer logically entails from context
    │  └─ Threshold: 0.65
    │
    ├─ Citation Coverage Check
    │  ├─ Verify citations are valid
    │  ├─ Check coverage of key claims
    │  └─ Ensure no unsupported statements
    │
    ├─ Hallucination Detection
    │  ├─ Check for contradictions
    │  ├─ Detect unsupported claims
    │  └─ Analyze evidence support
    │
    └─ Confidence Scoring
       ├─ Combine faithfulness + citation + evidence scores
       ├─ Apply hedging penalty
       └─ Return: confidence score (0-1)
    
    ↓
Verification Result {
    "verified": True/False,
    "confidence": 0.85,
    "level": "HIGH|MEDIUM|LOW",
    "should_retry": True/False,
    "reason": "explanation"
}
```

**File: [verification/verifier.py](verification/verifier.py)**

---

### **PHASE 9: AGENTIC LOOP DECISION** (Control Plane)

```
Verification Result
    ↓
DecisionLogic (ExecutionEngine):
    
    IF confidence >= 0.7 AND verified == True:
        ↓
        ✓ ACCEPT ANSWER
        └─ Cache result (Phase 1)
           └─ Return response to user
    
    ELSE IF should_retry == True AND attempts < max_attempts:
        ↓
        ⟲ RETRY (go back to Step 2: Planning)
        └─ Loop with confidence threshold
    
    ELSE:
        ↓
        ✓ RETURN BEST EFFORT
        └─ Return highest-confidence attempt
```

**Control Flow File: [orchestrator/execution_engine.py](orchestrator/execution_engine.py#L200-L280)**

---

### **PHASE 10: RESPONSE CACHING** (Phase 1 Optimization)

```
Accepted Answer
    ↓
ResultCache.set(query, response)  [utils/result_cache.py]
    • Store with TTL: 1 hour
    • Enable fuzzy query matching
    • Max cache size: 512 MB
    ↓
Answer cached for future requests
```

---

## Complete File Dependency Map

```
orchestrator/
├── execution_engine.py              ← MAIN ORCHESTRATOR
│   ├── imports: planner/local_agentic_planner.py
│   ├── imports: retrieval/dense_retriever.py
│   ├── imports: retrieval/sparse_retriever.py
│   ├── imports: retrieval/parallel_retriever.py
│   ├── imports: retrieval/reranker.py
│   ├── imports: retrieval/query_processor.py
│   ├── imports: retrieval/query_expansion.py
│   ├── imports: retrieval/metrics.py
│   ├── imports: fusion/context_fusion.py
│   ├── imports: generation/grounded_llm.py
│   ├── imports: verification/verifier.py
│   └── imports: utils/result_cache.py
│
planner/
├── local_agentic_planner.py         ← CONTROL PLANE: PLANNING
│   └── imports: planner/schemas.py
│
retrieval/
├── query_processor.py               ← DATA PLANE: QUERY PREPROCESSING
├── dense_retriever.py               ← DATA PLANE: SEMANTIC SEARCH
│   └── imports: utils/dual_embedder.py
├── sparse_retriever.py              ← DATA PLANE: KEYWORD SEARCH
├── parallel_retriever.py            ← DATA PLANE: PARALLEL EXECUTION
│   ├── imports: retrieval/dense_retriever.py
│   ├── imports: retrieval/sparse_retriever.py
│   ├── imports: retrieval/reranker.py
│   └── imports: retrieval/query_expansion.py
├── query_expansion.py               ← DATA PLANE: QUERY ENHANCEMENT
├── reranker.py                      ← DATA PLANE: RELEVANCE SCORING
└── metrics.py                       ← DATA PLANE: PERFORMANCE TRACKING
│
fusion/
└── context_fusion.py                ← DATA PLANE: CONTEXT PREPARATION

generation/
└── grounded_llm.py                  ← DATA PLANE: ANSWER GENERATION

verification/
└── verifier.py                      ← CONTROL PLANE: ANSWER VERIFICATION

utils/
├── result_cache.py                  ← PHASE 1: RESULT CACHING
├── dual_embedder.py                 ← EMBEDDERS: DUAL INDEX
├── sota_embedder.py                 ← EMBEDDERS: SOTA MODELS
├── embedding_cache.py               ← CACHING: EMBEDDINGS
└── model_loader.py                  ← UTILITIES: MODEL LOADING

config/
└── settings.yaml                    ← CONFIGURATION: All system parameters
```

---

## Data Flow Diagram

```
┌─────────────────────────┐
│   User Query (string)   │
└────────────┬────────────┘
             │
             ▼
    ┌────────────────────┐
    │ QueryProcessor     │ → Cleaned Query
    └────────────┬───────┘
                 │
                 ▼
    ┌────────────────────┐
    │ ResultCache.get()  │ → [IF HIT: Return cached result]
    └────────────┬───────┘
                 │ [MISS]
                 ▼
    ┌────────────────────┐
    │ LocalAgenticPlanner│ → RetrievalPlan (intent, modalities, k values)
    └────────────┬───────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
    ┌───────────┐   ┌──────────────┐
    │DenseSearch│   │SparseSearch  │  (Parallel)
    │ BAAI/CLIP │   │  BM25/Keyword│
    └─────┬─────┘   └────────┬─────┘
          │                  │
          └──────────┬───────┘
                     ▼
    ┌────────────────────────────┐
    │  Reranker (jinaai)         │ → Top-K results
    └────────────┬───────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │  ContextFusion             │ → Formatted context string
    └────────────┬───────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │  GroundedLLM (llama3:8b)   │ → Answer + citations + confidence
    └────────────┬───────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │  Verifier (NLI)            │ → Verified confidence score
    └────────────┬───────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │  Decision Logic            │
    │  (Confidence threshold)    │
    └────────────┬───────────────┘
                 │
        ┌────────┴────────┐
        │                 │
    [PASS]          [FAIL & Retry]
        │                 │
        │          ┌──────┴──────┐
        │          │ Loop back   │
        │          │ to planner  │
        │          │ (max 2x)    │
        │          └─────────────┘
        │
        ▼
    ┌─────────────────────────────┐
    │  ResultCache.set()          │ → Cached for 1 hour
    └────────────┬────────────────┘
                 │
                 ▼
    ┌─────────────────────────────┐
    │  Return Final Response       │
    │  {answer, sources,          │
    │   confidence, timings}      │
    └─────────────────────────────┘
```

---

## Control Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│               ExecutionEngine.run(query)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
          START → Preprocess Query
                       │
                       ▼
          Check Cache? ─YES→ Return Cached Result
                │
               NO
                │
                ▼
       ┌─────────────────────────┐     (Attempt 1/2)
       │  Agentic Loop Begins    │
       └────────────┬────────────┘
                    │
                    ▼
          ┌────────────────────┐
          │  PLAN QUERY        │ (Planner)
          │  - Intent          │
          │  - Modalities      │
          │  - Strategy        │
          └────────┬───────────┘
                   │
                   ▼
          ┌────────────────────┐
          │ RETRIEVE CONTENT   │ (Parallel Retrieval)
          │ - Dense Search     │
          │ - Sparse Search    │
          │ - Reranking        │
          └────────┬───────────┘
                   │
                   ▼
          ┌────────────────────┐
          │  FUSE CONTEXT      │ (Context Fusion)
          └────────┬───────────┘
                   │
                   ▼
          ┌────────────────────┐
          │ GENERATE ANSWER    │ (LLM)
          │ + Citations        │
          └────────┬───────────┘
                   │
                   ▼
          ┌────────────────────┐
          │ VERIFY ANSWER      │ (Verifier)
          │ - Faithfulness     │
          │ - Citations        │
          │ - Hallucination    │
          └────────┬───────────┘
                   │
                   ▼
          ┌─────────────────────────┐
          │ Confidence >= Threshold?│
          └────┬──────────────┬─────┘
              YES             NO
               │               │
               ▼               ▼
           ✓ ACCEPT        Should Retry?
           Cache Result        │
           Return Answer   ┌───┴────┐
                          YES      NO
                           │        │
                  ┌─────────┘        ▼
                  │             Return Best
                  ▼             Effort Answer
          Attempt < Max?
               │
          ┌────┴────┐
         YES         NO
          │           │
          │        ✓ Return
          └──→ Loop
          (Attempt 2)
```

---

## Configuration Control Points

All pipeline behavior is controlled via **[config/settings.yaml](config/settings.yaml)**:

```yaml
# Agent control
agent:
  confidence_threshold: 0.7      # When to accept answer
  max_attempts: 2                # Retry loop limit

# Model selection
models:
  embedding_model: "BAAI/bge-m3" # Dense retrieval
  image_embedding_model: "openai/clip-vit-large-patch14"  # Image
  
# Planner settings
planner:
  mode: "fast"                   # Skip LLM planning
  cache_ttl_seconds: 3600        # Plan caching

# Retrieval parameters
retrieval:
  dense_k: 20                    # Dense results to fetch
  sparse_k: 20                   # Sparse results to fetch
  rerank_k: 12                   # Final reranked results

# Reranker
reranker:
  model: "jinaai/jina-reranker-v3"

# Fusion
fusion:
  max_chunks: 12                 # Max context chunks
  max_tokens: 1200               # Context token limit

# Generation
generation:
  max_generation_tokens: 400
  temperature: 0.3               # Low randomness
  ollama:
    host: "http://localhost:11434"

# Verification
verification:
  enabled: true
  faithfulness_threshold: 0.65

# Caching (Phase 1)
cache:
  enabled: true
  query_ttl: 3600                # 1 hour
  enable_fuzzy: true             # Fuzzy matching
```

---

## Entry Points

### 1. **Interactive Chat Mode** (Recommended for Development)
```bash
python chat.py
```
- [chat.py](chat.py) creates an ExecutionEngine instance and loops user queries
- Real-time responses with timing breakdown

### 2. **Direct Orchestrator** (Non-interactive, Single Query)
```bash
python -m orchestrator.execution_engine
```

### 3. **RAGAS Evaluation Pipeline** (Benchmarking)
```bash
python scripts/run_ragas_pipeline.py --input evaluation/ragas_dev.jsonl --output artifacts/predictions.jsonl --mock
```
- [scripts/run_ragas_pipeline.py](scripts/run_ragas_pipeline.py)
- Uses ExecutionEngine for batch evaluation

---

## Key Optimizations (Phase 1)

| Feature | File | Benefit |
|---------|------|---------|
| **Parallel Retrieval** | [retrieval/parallel_retriever.py](retrieval/parallel_retriever.py) | 5.6x latency reduction |
| **Query Expansion** | [retrieval/query_expansion.py](retrieval/query_expansion.py) | 15-20% recall improvement |
| **Result Caching** | [utils/result_cache.py](utils/result_cache.py) | 0ms latency for repeated queries |
| **Fuzzy Matching** | [utils/result_cache.py](utils/result_cache.py) | Cache hits on similar queries |
| **Query Preprocessing** | [retrieval/query_processor.py](retrieval/query_processor.py) | Better matching accuracy |

---

## Performance Metrics

The system tracks:
- **Retrieval latency** (dense + sparse + reranking)
- **Fusion latency** (deduplication + formatting)
- **Generation latency** (LLM inference)
- **Verification latency** (NLI scoring)
- **Total pipeline latency**
- **Cache hit rate**

**File: [retrieval/metrics.py](retrieval/metrics.py)**

---

## Troubleshooting

| Issue | Likely Cause | Files to Check |
|-------|--------------|-----------------|
| Low confidence scores | Weak retrieval | [retrieval/dense_retriever.py](retrieval/dense_retriever.py), [retrieval/sparse_retriever.py](retrieval/sparse_retriever.py) |
| Slow generation | LLM inference time | [generation/grounded_llm.py](generation/grounded_llm.py), Ollama settings |
| Many retries | Verification threshold too high | [config/settings.yaml](config/settings.yaml) `agent.confidence_threshold` |
| Out of memory | Large context fusion | [config/settings.yaml](config/settings.yaml) `fusion.max_tokens` |
| Cache not working | Redis/memory cache disabled | [config/settings.yaml](config/settings.yaml) `cache.enabled` |

---

## Summary

The **Agentic Multimodal RAG** system is a **closed-loop pipeline** with:

1. **Control Plane** (Decision-making):
   - Planner: Intent classification, strategy selection
   - Verifier: Answer validation, confidence scoring

2. **Data Plane** (Execution):
   - Retrieval: Dense + sparse hybrid search
   - Fusion: Context deduplication and formatting
   - Generation: Grounded LLM answer generation

3. **Optimizations** (Phase 1):
   - Parallel retrieval
   - Query expansion
   - Result caching
   - Fuzzy matching

**All orchestration happens in: [orchestrator/execution_engine.py](orchestrator/execution_engine.py)**
