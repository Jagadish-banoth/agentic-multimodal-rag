# RAGAS Evaluation Report

**Agentic Multimodal RAG System - Comprehensive Evaluation Metrics**

Generated from artifacts in `artifacts/` directory.

---

## Executive Summary

| Dataset | Total Queries | Accuracy | Avg Confidence |
|---------|---------------|----------|----------------|
| RAGAS 50 Random (Ollama) | 30 | **76.67%** (23/30) | 0.88 |
| Full Evaluation | 5 | - | - |

---

## 1. Retrieval Metrics

Measures how well the system retrieves relevant documents.

### RAGAS 50 Queries Evaluation (30 queries)

| Metric | Score | Description |
|--------|-------|-------------|
| **Recall@1** | 0.6333 | Relevant doc found in top-1 result |
| **Recall@5** | 0.8333 | Relevant doc found in top-5 results |
| **Recall@10** | 0.8333 | Relevant doc found in top-10 results |
| **MRR** | 0.6956 | Mean Reciprocal Rank |

### Full Evaluation (5 queries)

| Metric | Score | Description |
|--------|-------|-------------|
| **Recall@1** | 0.8000 | Relevant doc found in top-1 result |
| **Recall@5** | 1.0000 | Relevant doc found in top-5 results |
| **Recall@10** | 1.0000 | Relevant doc found in top-10 results |
| **MRR** | 0.8400 | Mean Reciprocal Rank |

---

## 2. RAGAS Framework Metrics

Core evaluation metrics from the RAGAS (Retrieval Augmented Generation Assessment) framework.

### RAGAS 50 Queries Evaluation

| Metric | Score | Description |
|--------|-------|-------------|
| **Faithfulness** | NaN* | How factually consistent the answer is with context |
| **Answer Relevancy** | 0.5697 | How relevant the answer is to the question |
| **Context Precision** | NaN* | Precision of retrieved context ranking |
| **Context Recall** | NaN* | Coverage of required information in context |
| **Answer Correctness** | NaN* | Factual correctness compared to reference |
| **Answer Similarity** | 0.5819 | Semantic similarity to reference answer |

*NaN values indicate metrics that could not be computed for some queries (e.g., when LLM judging failed or reference was unavailable).

---

## 3. Generation Quality Metrics

Measures the quality of generated text compared to reference answers.

### RAGAS 50 Queries Evaluation

| Metric | Score | Description |
|--------|-------|-------------|
| **BLEU Avg** | 0.0677 | Average BLEU score (1-4 gram) |
| **ROUGE-1** | 0.1461 | Unigram overlap with reference |
| **ROUGE-L** | 0.1406 | Longest common subsequence |
| **BERTScore F1** | 0.6971 | Semantic similarity (BERT embeddings) |

### Full Evaluation (5 queries - Higher Reference Overlap)

| Metric | Score | Description |
|--------|-------|-------------|
| **BLEU-1** | 0.2255 | Unigram BLEU score |
| **BLEU-2** | 0.1928 | Bigram BLEU score |
| **BLEU-3** | 0.1572 | Trigram BLEU score |
| **BLEU-4** | 0.1418 | 4-gram BLEU score |
| **BLEU Avg** | 0.1793 | Average BLEU score |
| **ROUGE-1** | 0.3528 | Unigram overlap with reference |
| **ROUGE-2** | 0.2863 | Bigram overlap |
| **ROUGE-L** | 0.3354 | Longest common subsequence |
| **BERTScore P** | -0.2817 | BERTScore Precision |
| **BERTScore R** | 0.5118 | BERTScore Recall |
| **BERTScore F1** | 0.0849 | BERTScore F1 |

---

## 4. Grounding & Verification Metrics

Measures how well answers are grounded in retrieved evidence.

### RAGAS 50 Queries Evaluation

| Metric | Score | Description |
|--------|-------|-------------|
| **Grounding Ratio** | 0.1542 | Proportion of claims supported by evidence |
| **Claim Support Ratio** | 0.1542 | Ratio of supported claims to total claims |

---

## 5. Accuracy & Confidence

### RAGAS 50 Queries Evaluation

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 76.67% (23/30) | Percentage of correct answers |
| **Avg Confidence** | 0.8800 | Average model confidence score |

---

## 6. Performance Metrics

### Full Evaluation

| Metric | Value |
|--------|-------|
| **Total Time** | 249.37 seconds |
| **Avg Time per Query** | 49.87 seconds |

---

## 7. Metric Definitions

### Retrieval Metrics
- **Recall@K**: Proportion of queries where the relevant document appears in top-K results
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank of first relevant document

### RAGAS Metrics
- **Faithfulness**: Measures if claims in the answer can be inferred from the context
- **Answer Relevancy**: Measures how pertinent the generated answer is to the question
- **Context Precision**: Signal-to-noise ratio of retrieved contexts
- **Context Recall**: Ground truth coverage by retrieved context
- **Answer Similarity**: Semantic similarity between generated and reference answers

### Generation Quality Metrics
- **BLEU**: Measures n-gram overlap between generated and reference text
- **ROUGE**: Recall-oriented measure of n-gram/LCS overlap
- **BERTScore**: Semantic similarity using BERT embeddings

### Grounding Metrics
- **Grounding Ratio**: Fraction of generated statements with supporting evidence
- **Claim Support Ratio**: Proportion of extractable claims that are supported

---

## 8. Source Artifact Files

| File | Description | Queries |
|------|-------------|---------|
| `ragas_50_random_ollama.csv` | Per-query RAGAS evaluation results | 30 |
| `ragas_50_random_ollama.jsonl` | Detailed per-query data with retrieved contexts | 30 |
| `full_eval_results_summary.json` | Aggregated evaluation summary | 5 |
| `full_eval.jsonl` | Detailed predictions with full retrieval info | 5 |
| `predictions.jsonl` | Generation outputs with structured evidence | - |

---

## 9. Model Configuration

| Component | Model |
|-----------|-------|
| **Embeddings (Text)** | sentence-transformers/all-MiniLM-L6-v2 |
| **Embeddings (Multimodal)** | clip-ViT-B-32 |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| **Planner** | gemma2:2b (Ollama) |
| **Generator** | llama3:8b (Ollama) / nvidia/nemotron (OpenRouter) |
| **Evaluator** | RAGAS with LLM-as-judge |

---

## Key Insights

1. **Retrieval Performance**: Strong Recall@5/10 (83-100%) indicates the retrieval pipeline finds relevant documents. Lower Recall@1 suggests reranking can be improved.

2. **Answer Quality**: BERTScore F1 of ~0.70 shows reasonable semantic alignment; lower BLEU/ROUGE scores are typical for abstractive generation.

3. **Accuracy**: 76.67% accuracy demonstrates the system answers most questions correctly.

4. **Grounding Gap**: Low grounding ratio (15.4%) indicates answers may include information beyond what's explicitly in retrieved evidence - an area for improvement.

5. **Confidence Calibration**: High confidence (0.88) with 76.67% accuracy suggests slightly overconfident predictions.

---

*Report generated from evaluation artifacts. For reproducibility, see `evaluation/` and `scripts/run_ragas_pipeline.py`.*
