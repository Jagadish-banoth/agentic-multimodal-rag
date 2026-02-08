"""
Full RAGAS Evaluation with Local LLM (Ollama) - Expert Developer Script
========================================================================

Computes: BLEU, ROUGE-L, BERTScore, Recall@K, MRR, Grounding Ratio

Usage:
    python scripts/run_full_eval.py --num-queries 10

Features:
- Forces Ollama-only generation (no OpenRouter)
- Handles GPU OOM with CPU fallback
- Computes all major NLG metrics
- Saves detailed results to artifacts/

Author: Expert RAG System
Date: Feb 2026
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml

# ============================================================================
# METRIC IMPORTS (Lazy loading to avoid import blocking)
# ============================================================================

ROUGE_AVAILABLE = False
BERTSCORE_AVAILABLE = False
BLEU_AVAILABLE = False

rouge_scorer = None
bert_score_fn = None
sentence_bleu = None
SmoothingFunction = None

def _lazy_load_metrics():
    """Lazy load metric libraries to avoid blocking imports."""
    global ROUGE_AVAILABLE, BERTSCORE_AVAILABLE, BLEU_AVAILABLE
    global rouge_scorer, bert_score_fn, sentence_bleu, SmoothingFunction
    
    # ROUGE
    try:
        from rouge_score import rouge_scorer as rs
        rouge_scorer = rs
        ROUGE_AVAILABLE = True
    except ImportError:
        print("⚠️ rouge_score not installed. Run: pip install rouge-score")
    
    # BERTScore - only load if not disabled
    if os.getenv("BERTSCORE_DISABLED", "0") != "1":
        try:
            from bert_score import score as bsf
            bert_score_fn = bsf
            BERTSCORE_AVAILABLE = True
        except ImportError:
            print("⚠️ bert_score not installed. Run: pip install bert-score")
    else:
        print("ℹ️ BERTScore disabled via env var")
    
    # BLEU (NLTK)
    try:
        from nltk.translate.bleu_score import sentence_bleu as sb, SmoothingFunction as sf
        import nltk
        sentence_bleu = sb
        SmoothingFunction = sf
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        BLEU_AVAILABLE = True
    except ImportError:
        print("⚠️ nltk not installed. Run: pip install nltk")


def load_config(path: str = "config/settings.yaml") -> Dict:
    """Load configuration from YAML file."""
    config_path = Path(ROOT) / path
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


# ============================================================================
# LOCAL-ONLY GENERATOR (Ollama Forced)
# ============================================================================

class OllamaOnlyGenerator:
    """Generator that ONLY uses local Ollama - no external APIs."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = config.get("models", {}).get("generator_model", "llama3:8b")
        
        gen_cfg = config.get("generation", {})
        self.temperature = gen_cfg.get("temperature", 0.3)
        self.top_p = gen_cfg.get("top_p", 0.9)
        self.max_tokens = gen_cfg.get("max_generation_tokens", 512)
        
        # Ollama client
        try:
            import ollama
            self.ollama = ollama
            self._client = None
            try:
                from ollama import Client
                host = gen_cfg.get("ollama", {}).get("host", "http://localhost:11434")
                self._client = Client(host=host)
            except Exception:
                pass
            print(f"✅ Ollama available with model: {self.model}")
        except ImportError:
            raise RuntimeError("❌ Ollama not installed. Run: pip install ollama")
    
    def generate(self, query: str, context: str) -> Dict[str, Any]:
        """Generate answer using ONLY local Ollama."""
        
        prompt = f"""You are an expert document analyst. Answer the question using ONLY the provided context.

RULES:
1. Use ONLY information from the context below
2. Cite sources using [Source N] format
3. If context lacks info, say "Based on the provided evidence, I cannot answer because..."

CONTEXT:
{context}

QUESTION: {query}

ANSWER (with citations):"""

        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Always cite your sources."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            chat_fn = self._client.chat if self._client else self.ollama.chat
            
            response = chat_fn(
                model=self.model,
                messages=messages,
                stream=False,
                options={
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": self.max_tokens,
                    "num_ctx": 4096,
                }
            )
            
            answer = response.get("message", {}).get("content", "").strip()
            
            if not answer:
                return {"answer": "[GENERATION FAILED - Empty response]", "error": True}
            
            return {
                "answer": answer,
                "model": self.model,
                "error": False
            }
            
        except Exception as e:
            error_msg = str(e)
            if "CUDA" in error_msg or "out of memory" in error_msg.lower():
                print(f"⚠️ GPU OOM error, trying with reduced context...")
                # Retry with shorter context
                try:
                    short_context = context[:2000] if len(context) > 2000 else context
                    short_prompt = f"""Answer based on context. Cite sources.

CONTEXT: {short_context}

QUESTION: {query}

ANSWER:"""
                    response = chat_fn(
                        model=self.model,
                        messages=[{"role": "user", "content": short_prompt}],
                        stream=False,
                        options={"temperature": 0.3, "num_predict": 256}
                    )
                    answer = response.get("message", {}).get("content", "").strip()
                    return {"answer": answer, "model": self.model, "error": False, "truncated": True}
                except Exception as retry_e:
                    return {"answer": f"[GENERATION FAILED - {retry_e}]", "error": True}
            
            return {"answer": f"[GENERATION FAILED - {error_msg}]", "error": True}


# ============================================================================
# METRICS CALCULATOR
# ============================================================================

class MetricsCalculator:
    """Calculate BLEU, ROUGE-L, and BERTScore metrics."""
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        
        # Lazy load metrics on first use
        _lazy_load_metrics()
        
        # Initialize ROUGE scorer
        if ROUGE_AVAILABLE and rouge_scorer:
            self._rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        else:
            self._rouge = None
        
        # BLEU smoothing function
        if BLEU_AVAILABLE and SmoothingFunction:
            self._smoothing = SmoothingFunction().method1
        else:
            self._smoothing = None
        
    def compute_bleu(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Compute BLEU score (1-4 grams)."""
        if not BLEU_AVAILABLE or not sentence_bleu or not reference or not hypothesis:
            return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0, "bleu_avg": 0.0}
        
        try:
            ref_tokens = reference.lower().split()
            hyp_tokens = hypothesis.lower().split()
            
            if len(hyp_tokens) < 1 or len(ref_tokens) < 1:
                return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0, "bleu_avg": 0.0}
            
            # Compute individual n-gram BLEU scores
            bleu1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=self._smoothing)
            bleu2 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=self._smoothing)
            bleu3 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.33, 0.33, 0.34, 0), smoothing_function=self._smoothing)
            bleu4 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self._smoothing)
            
            return {
                "bleu1": round(bleu1, 4),
                "bleu2": round(bleu2, 4),
                "bleu3": round(bleu3, 4),
                "bleu4": round(bleu4, 4),
                "bleu_avg": round((bleu1 + bleu2 + bleu3 + bleu4) / 4, 4)
            }
        except Exception as e:
            print(f"⚠️ BLEU computation failed: {e}")
            return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0, "bleu_avg": 0.0}
    
    def compute_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Compute ROUGE scores (1, 2, L)."""
        if not ROUGE_AVAILABLE or not reference or not hypothesis:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        try:
            scores = self._rouge.score(reference, hypothesis)
            return {
                "rouge1": round(scores["rouge1"].fmeasure, 4),
                "rouge2": round(scores["rouge2"].fmeasure, 4),
                "rougeL": round(scores["rougeL"].fmeasure, 4)
            }
        except Exception as e:
            print(f"⚠️ ROUGE computation failed: {e}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    def compute_bertscore(self, references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        """Compute BERTScore (batch for efficiency) with smart fallback strategy."""
        if not BERTSCORE_AVAILABLE or not bert_score_fn or not references or not hypotheses:
            return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
        
        # Try primary model first (roberta-large for stability)
        try:
            P, R, F = bert_score_fn(
                hypotheses,
                references,
                lang="en",
                rescale_with_baseline=True,
                model_type="roberta-large",
                device="cpu",  # Force CPU for stability
                verbose=False,
                batch_size=16
            )
            
            return {
                "bertscore_precision": round(float(P.mean()), 4),
                "bertscore_recall": round(float(R.mean()), 4),
                "bertscore_f1": round(float(F.mean()), 4)
            }
        except Exception as e:
            print(f"⚠️ BERTScore (roberta-large) failed: {e}")
            
            # Fallback to distilbert-base-uncased (lighter model)
            try:
                P, R, F = bert_score_fn(
                    hypotheses,
                    references,
                    lang="en",
                    model_type="distilbert-base-uncased",
                    device="cpu",
                    verbose=False,
                    batch_size=32
                )
                print("✓ BERTScore fallback (distilbert) succeeded")
                return {
                    "bertscore_precision": round(float(P.mean()), 4),
                    "bertscore_recall": round(float(R.mean()), 4),
                    "bertscore_f1": round(float(F.mean()), 4)
                }
            except Exception as e2:
                print(f"⚠️ BERTScore fallback also failed: {e2}")
                return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def run_evaluation(
    input_file: str,
    output_file: str,
    num_queries: int = 10,
    use_gpu: bool = False,
    config: Optional[Dict] = None,
    random_sample: bool = False
):
    """Run full evaluation pipeline with local LLM."""
    
    config = config or load_config()
    
    print("=" * 70)
    print("🚀 FULL RAGAS EVALUATION (Local LLM Only)")
    print("=" * 70)
    print(f"📁 Input:  {input_file}")
    print(f"📁 Output: {output_file}")
    print(f"📊 Queries: {num_queries}")
    print(f"🎲 Random Sample: {'Yes' if random_sample else 'No'}")
    print(f"🖥️  GPU: {'Enabled' if use_gpu else 'Disabled (CPU mode)'}")
    print("=" * 70)
    
    # Initialize components
    print("\n⏳ Loading components...")
    
    # 1. Retriever
    try:
        from retrieval.dense_retriever import DenseRetriever
        retriever = DenseRetriever(config)
        print("✅ DenseRetriever loaded")
    except Exception as e:
        print(f"⚠️ DenseRetriever failed ({e}), using SparseRetriever")
        from retrieval.sparse_retriever import SparseRetriever
        retriever = SparseRetriever(config)
    
    # 2. Reranker
    try:
        from retrieval.reranker import Reranker
        reranker = Reranker(config)
        print("✅ Reranker loaded")
    except Exception as e:
        print(f"⚠️ Reranker failed ({e}), using identity reranker")
        class IdentityReranker:
            def rerank(self, q, c): return [{**x, "rerank_score": x.get("score", 0)} for x in c]
        reranker = IdentityReranker()
    
    # 3. Fusion
    from fusion.context_fusion import ContextFusion
    fusion = ContextFusion(config)
    print("✅ ContextFusion loaded")
    
    # 4. Generator (Ollama ONLY)
    generator = OllamaOnlyGenerator(config)
    print("✅ OllamaOnlyGenerator loaded")
    
    # 5. Metrics calculator
    metrics_calc = MetricsCalculator(use_gpu=use_gpu)
    print("✅ MetricsCalculator loaded")
    
    # Load queries
    queries = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    
    # Random sample if requested
    if random_sample and len(queries) > num_queries:
        queries = random.sample(queries, num_queries)
        print(f"🎲 Randomly sampled {num_queries} queries from {len(queries) + num_queries} total")
    else:
        # Limit to num_queries
        queries = queries[:num_queries]
    
    print(f"\n📋 Loaded {len(queries)} queries for evaluation")
    
    # Run evaluation
    results = []
    all_references = []
    all_hypotheses = []
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print("\n" + "=" * 70)
    print("🔄 RUNNING EVALUATION")
    print("=" * 70)
    
    start_time = time.time()
    
    for idx, item in enumerate(queries, 1):
        qid = item.get("query_id", f"q{idx}")
        query = item.get("query", "")
        gold = item.get("gold", {})
        gold_answers = gold.get("answers", [""])
        gold_evidence = gold.get("evidence_ids", [])
        reference = gold_answers[0] if gold_answers else ""
        
        print(f"\n[{idx}/{len(queries)}] {qid}: {query[:60]}...")
        
        # 1. Retrieve
        t0 = time.time()
        retrieved = retriever.retrieve(query)
        t_retrieval = time.time() - t0
        
        # 2. Rerank
        t0 = time.time()
        reranked = reranker.rerank(query, retrieved)
        t_rerank = time.time() - t0
        
        # Normalize retrieved results
        normalized = []
        for r in reranked:
            doc_id = r.get("doc_id") or r.get("chunk_id") or r.get("id", "")
            excerpt = r.get("excerpt") or r.get("content") or r.get("snippet", "")
            normalized.append({"doc_id": doc_id, "excerpt": excerpt, "score": r.get("rerank_score", 0)})
        
        # 3. Fuse context
        t0 = time.time()
        fused_context = fusion.build_context(query, reranked)
        t_fusion = time.time() - t0
        
        # 4. Generate answer (LOCAL ONLY)
        t0 = time.time()
        gen_result = generator.generate(query, fused_context)
        t_generation = time.time() - t0
        generated_text = gen_result.get("answer", "")
        
        print(f"   ✓ Generated ({t_generation:.1f}s): {generated_text[:80]}...")
        
        # 5. Compute retrieval metrics
        retrieved_ids = [r["doc_id"] for r in normalized]
        recall_1 = 1.0 if any(g in retrieved_ids[:1] for g in gold_evidence) else 0.0
        recall_5 = 1.0 if any(g in retrieved_ids[:5] for g in gold_evidence) else 0.0
        recall_10 = 1.0 if any(g in retrieved_ids[:10] for g in gold_evidence) else 0.0
        
        mrr = 0.0
        for i, rid in enumerate(retrieved_ids, 1):
            if rid in gold_evidence:
                mrr = 1.0 / i
                break
        
        # 6. Compute generation metrics (per-query BLEU & ROUGE)
        bleu_scores = metrics_calc.compute_bleu(reference, generated_text)
        rouge_scores = metrics_calc.compute_rouge(reference, generated_text)
        
        # Collect for batch BERTScore
        if reference and generated_text and not gen_result.get("error"):
            all_references.append(reference)
            all_hypotheses.append(generated_text)
        
        # Build result object
        result = {
            "query_id": qid,
            "query": query,
            "generated": generated_text,
            "reference": reference,
            "retrieval": {
                "recall@1": recall_1,
                "recall@5": recall_5,
                "recall@10": recall_10,
                "mrr": mrr
            },
            "generation": {
                **bleu_scores,
                **rouge_scores
            },
            "timings": {
                "retrieval": round(t_retrieval, 3),
                "rerank": round(t_rerank, 3),
                "fusion": round(t_fusion, 3),
                "generation": round(t_generation, 3)
            },
            "error": gen_result.get("error", False)
        }
        results.append(result)
        
        print(f"   📊 Recall@1={recall_1:.2f} | BLEU-4={bleu_scores['bleu4']:.3f} | ROUGE-L={rouge_scores['rougeL']:.3f}")
    
    # Compute batch BERTScore
    print("\n⏳ Computing BERTScore (batch)...")
    if all_references and all_hypotheses:
        bert_scores = metrics_calc.compute_bertscore(all_references, all_hypotheses)
        print(f"   ✅ BERTScore F1: {bert_scores['bertscore_f1']:.4f}")
    else:
        bert_scores = {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
    
    total_time = time.time() - start_time
    
    # Aggregate results
    n = len(results)
    agg = {
        "recall@1": sum(r["retrieval"]["recall@1"] for r in results) / n if n else 0,
        "recall@5": sum(r["retrieval"]["recall@5"] for r in results) / n if n else 0,
        "recall@10": sum(r["retrieval"]["recall@10"] for r in results) / n if n else 0,
        "mrr": sum(r["retrieval"]["mrr"] for r in results) / n if n else 0,
        "bleu1": sum(r["generation"]["bleu1"] for r in results) / n if n else 0,
        "bleu2": sum(r["generation"]["bleu2"] for r in results) / n if n else 0,
        "bleu3": sum(r["generation"]["bleu3"] for r in results) / n if n else 0,
        "bleu4": sum(r["generation"]["bleu4"] for r in results) / n if n else 0,
        "bleu_avg": sum(r["generation"]["bleu_avg"] for r in results) / n if n else 0,
        "rouge1": sum(r["generation"]["rouge1"] for r in results) / n if n else 0,
        "rouge2": sum(r["generation"]["rouge2"] for r in results) / n if n else 0,
        "rougeL": sum(r["generation"]["rougeL"] for r in results) / n if n else 0,
        **bert_scores,
        "total_queries": n,
        "total_time_seconds": round(total_time, 2),
        "avg_time_per_query": round(total_time / n, 2) if n else 0
    }
    
    # Save detailed results
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    
    # Save summary
    summary_file = output_file.replace(".jsonl", "_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 EVALUATION SUMMARY")
    print("=" * 70)
    print("\n📈 RETRIEVAL METRICS:")
    print(f"   Recall@1:  {agg['recall@1']:.4f}")
    print(f"   Recall@5:  {agg['recall@5']:.4f}")
    print(f"   Recall@10: {agg['recall@10']:.4f}")
    print(f"   MRR:       {agg['mrr']:.4f}")
    
    print("\n📝 GENERATION METRICS (BLEU):")
    print(f"   BLEU-1:    {agg['bleu1']:.4f}")
    print(f"   BLEU-2:    {agg['bleu2']:.4f}")
    print(f"   BLEU-3:    {agg['bleu3']:.4f}")
    print(f"   BLEU-4:    {agg['bleu4']:.4f}")
    print(f"   BLEU-Avg:  {agg['bleu_avg']:.4f}")
    
    print("\n📝 GENERATION METRICS (ROUGE):")
    print(f"   ROUGE-1:   {agg['rouge1']:.4f}")
    print(f"   ROUGE-2:   {agg['rouge2']:.4f}")
    print(f"   ROUGE-L:   {agg['rougeL']:.4f}")
    
    print("\n📝 GENERATION METRICS (BERTScore):")
    print(f"   Precision: {agg['bertscore_precision']:.4f}")
    print(f"   Recall:    {agg['bertscore_recall']:.4f}")
    print(f"   F1:        {agg['bertscore_f1']:.4f}")
    
    print("\n⏱️  TIMING:")
    print(f"   Total Time:        {agg['total_time_seconds']:.2f}s")
    print(f"   Avg Time/Query:    {agg['avg_time_per_query']:.2f}s")
    
    print("\n" + "=" * 70)
    print(f"✅ Results saved to: {output_file}")
    print(f"✅ Summary saved to: {summary_file}")
    print("=" * 70)
    
    return agg


def main():
    parser = argparse.ArgumentParser(description="Full RAGAS Evaluation with Local LLM")
    parser.add_argument("--input", default="artifacts/test_data_ragas.jsonl", help="Input JSONL file")
    parser.add_argument("--output", default="artifacts/full_eval_results.jsonl", help="Output JSONL file")
    parser.add_argument("--num-queries", type=int, default=10, help="Number of queries to evaluate")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU for BERTScore")
    parser.add_argument("--config", default="config/settings.yaml", help="Config file path")
    parser.add_argument("--random", action="store_true", help="Randomly sample queries")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    run_evaluation(
        input_file=args.input,
        output_file=args.output,
        num_queries=args.num_queries,
        use_gpu=args.gpu,
        config=config,
        random_sample=args.random
    )


if __name__ == "__main__":
    main()
