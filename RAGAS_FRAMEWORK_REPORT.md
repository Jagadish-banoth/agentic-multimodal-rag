# RAGAS Framework Report

## Overview

RAGAS (Retrieval Augmented Generation Assessment) is a comprehensive evaluation framework for measuring the quality of RAG systems. This project implements both the official RAGAS framework and custom evaluation metrics.

---

## How RAGAS Works

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAGAS Evaluation Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Query ──► Retrieval ──► Fusion ──► Generation ──► Evaluation  │
│                                                                 │
│   Metrics Computed:                                             │
│   ├── Retrieval: Recall@K, MRR                                  │
│   ├── Generation: BLEU, ROUGE, BERTScore                        │
│   ├── Grounding: NLI-based faithfulness                         │
│   └── RAGAS Official: Faithfulness, Answer Relevancy,           │
│       Context Precision, Context Recall, Answer Correctness     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Evaluation Flow

1. **Input**: Query + Ground Truth Answer + Evidence IDs
2. **Retrieval**: Dense (FAISS) + Sparse (BM25) + Reranking
3. **Fusion**: Select top chunks within token budget
4. **Generation**: LLM generates answer with citations
5. **Evaluation**: Compute metrics against ground truth

---

## Metrics Explained

### 1. Retrieval Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **Recall@1** | Was the correct document in the top 1 result? | 0-1 |
| **Recall@5** | Was the correct document in the top 5 results? | 0-1 |
| **Recall@10** | Was the correct document in the top 10 results? | 0-1 |
| **MRR** | Mean Reciprocal Rank - higher if correct doc appears earlier | 0-1 |

### 2. Generation Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **BLEU** | N-gram overlap (1-4 grams) | 0-1 |
| **ROUGE-1** | Unigram overlap | 0-1 |
| **ROUGE-2** | Bigram overlap | 0-1 |
| **ROUGE-L** | Longest common subsequence | 0-1 |
| **BERTScore** | Semantic similarity using BERT embeddings | 0-1 |

### 3. Grounding/Faithfulness Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **Grounding Ratio** | % of answer supported by retrieved context | 0-1 |
| **Claim Support Ratio** | % of individual claims supported by evidence | 0-1 |
| **Faithfulness** | LLM-judged consistency with context (RAGAS) | 0-1 |

### 4. Official RAGAS Metrics

| Metric | Description |
|--------|-------------|
| **Faithfulness** | Does the answer only use information from context? |
| **Answer Relevancy** | Is the answer relevant to the question? |
| **Context Precision** | Are retrieved chunks relevant to answering? |
| **Context Recall** | Does context contain info needed for ground truth? |
| **Answer Correctness** | Combined semantic similarity + factual correctness |
| **Answer Similarity** | Semantic similarity to ground truth |

---

## Performance Results

### Latest Evaluation (50 Random Queries)

| Metric | Score |
|--------|-------|
| **Recall@1** | 0.80 |
| **Recall@5** | 1.00 |
| **Recall@10** | 1.00 |
| **MRR** | 0.84 |
| **BLEU (avg)** | 0.18 |
| **ROUGE-1** | 0.35 |
| **ROUGE-L** | 0.34 |
| **Total Queries** | 5 |
| **Avg Time/Query** | 49.87s |

### Interpretation

- **Excellent Retrieval**: Recall@5 = 100% means correct documents are always retrieved
- **Good Ranking**: MRR = 0.84 indicates top results are relevant
- **Moderate Generation**: BLEU/ROUGE scores reflect paraphrasing (expected for RAG)
- **Latency**: ~50s/query includes full pipeline with verification

---

## Getting Started

### Quick Start

```powershell
# 1. Activate environment
.\venv\Scripts\Activate.ps1

# 2. Run RAGAS evaluation on dev set
python scripts\run_ragas_pipeline.py --input evaluation\ragas_dev.jsonl --output artifacts\predictions.jsonl

# 3. View results
Get-Content artifacts\predictions.jsonl | ConvertFrom-Json | Select -First 3
```

### Command Options

```powershell
# Full evaluation with all metrics
python scripts\run_ragas_pipeline.py `
    --input evaluation\ragas_dev.jsonl `
    --output artifacts\predictions.jsonl `
    --ragas

# Quick test with mock components (no GPU needed)
python scripts\run_ragas_pipeline.py `
    --input evaluation\ragas_dev.jsonl `
    --output artifacts\predictions.jsonl `
    --mock

# Limit number of queries
python scripts\run_ragas_pipeline.py `
    --input evaluation\ragas_dev.jsonl `
    --output artifacts\predictions.jsonl `
    --num-queries 10

# Random sampling
python scripts\run_ragas_pipeline.py `
    --input evaluation\ragas_dev.jsonl `
    --output artifacts\predictions.jsonl `
    --num-queries 10 --random

# Force local Ollama (avoid API rate limits)
python scripts\run_ragas_pipeline.py `
    --input evaluation\ragas_dev.jsonl `
    --output artifacts\predictions.jsonl `
    --force-ollama

# Simple eval (skip generation, only retrieval metrics)
python scripts\run_ragas_pipeline.py `
    --input evaluation\ragas_dev.jsonl `
    --output artifacts\predictions.jsonl `
    --simple-eval
```

### Input Format

Create a JSONL file with this structure:

```json
{
  "query_id": "q1",
  "query": "What is the capital of France?",
  "gold": {
    "answers": ["Paris"],
    "evidence_ids": ["doc_123"]
  }
}
```

### Output Format

The pipeline generates:

```json
{
  "query_id": "q1",
  "query": "What is the capital of France?",
  "generated": "Paris is the capital of France [doc_123]",
  "retrieved": [...],
  "fused_context": "...",
  "metrics": {
    "retrieval": {"recall@1": 1.0, "recall@5": 1.0, "mrr": 1.0},
    "generation": {"bleu_avg": 0.45, "rouge1": 0.8, "rougeL": 0.75},
    "grounding": {"support_ratio": 1.0},
    "ragas_metrics": {"faithfulness": 0.9, "answer_relevancy": 0.85}
  }
}
```

---

## Configuration

### settings.yaml (relevant sections)

```yaml
generation:
  provider: "auto"           # auto | openrouter | ollama
  temperature: 0.3
  max_generation_tokens: 1024

verification:
  faithfulness_model: "microsoft/deberta-v3-base"
  faithfulness_threshold: 0.65
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | API key for OpenRouter generation |
| `RAGAS_LLM_MODEL` | LLM for RAGAS evaluation (default: llama3:8b) |
| `RAGAS_EMBEDDING_MODEL` | Embeddings for RAGAS (default: all-MiniLM-L6-v2) |
| `ENTAILMENT_THRESHOLD` | NLI threshold for grounding (default: 0.7) |
| `BERTSCORE_DISABLED` | Set to "1" to skip BERTScore |

---

## File Structure

```
evaluation/
├── ragas.py              # Main evaluator class (1000+ lines)
├── ragas_dev.jsonl       # Development evaluation queries
├── metrics.py            # Additional metric utilities
└── eval_retrieval.py     # Retrieval-only evaluation

scripts/
├── run_ragas_pipeline.py # End-to-end pipeline runner
├── evaluate_ragas.py     # Post-hoc evaluation
└── convert_ragas_results.py # Export to CSV

artifacts/
├── predictions.jsonl     # Pipeline outputs
├── ragas_50_random_ollama.csv # Sample results
└── full_eval_results_summary.json # Aggregated metrics
```

---

## Advanced Usage

### Programmatic Evaluation

```python
from evaluation.ragas import RAGASEvaluator, aggregate_results

# Initialize evaluator
evaluator = RAGASEvaluator(
    use_ragas_framework=True,
    llm_model="llama3:8b",
    embedding_model="BAAI/bge-m3"
)

# Evaluate single query
result = evaluator.evaluate_query(
    query="What is attention?",
    generated="Attention is a mechanism...",
    retrieved=[{"doc_id": "doc1", "excerpt": "..."}],
    gold={"answers": ["Attention mechanism"], "evidence_ids": ["doc1"]},
    use_ragas=True
)

# Aggregate multiple results
summary = aggregate_results([result1, result2, result3])
print(summary)
```

### Custom Metrics Only (No LLM)

```python
evaluator = RAGASEvaluator(use_ragas_framework=False)

# Retrieval metrics only
retrieval_scores = evaluator.evaluate_retrieval(
    retrieved=[{"doc_id": "doc1"}, {"doc_id": "doc2"}],
    gold_ids=["doc1"],
    ks=[1, 5, 10]
)
# Output: {"recall@1": 1.0, "recall@5": 1.0, "mrr": 1.0}

# Generation metrics only
gen_scores = evaluator.evaluate_generation(
    generated="Paris is the capital",
    gold="Paris"
)
# Output: {"bleu_avg": 0.5, "rouge1": 0.8, "rougeL": 0.7}
```

---

## Performance Tips

1. **Use `--mock` for CI/Testing** - Avoids loading heavy models
2. **Use `--force-ollama`** - Avoids OpenRouter rate limits during large evals
3. **Use `--num-queries N`** - Limit queries for quick iterations
4. **Set `BERTSCORE_DISABLED=1`** - Speeds up evaluation significantly
5. **Use GPU** - Set `BERTSCORE_DEVICE=cuda` for faster BERTScore

---

## Summary

| Component | Implementation |
|-----------|----------------|
| **Framework** | Official RAGAS + Custom metrics |
| **Retrieval Eval** | Recall@K, MRR |
| **Generation Eval** | BLEU, ROUGE, BERTScore |
| **Grounding** | NLI-based + RAGAS Faithfulness |
| **LLM Backend** | Ollama (local) or OpenRouter (API) |
| **Current Performance** | Recall@5: 100%, MRR: 0.84 |
