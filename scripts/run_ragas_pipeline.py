"""Run the full RAGAS pipeline: retrieve -> rerank -> fuse -> generate -> evaluate

Supports a mock/dry-run mode for CI/tests to avoid loading heavy models.

Usage:
    python scripts/run_ragas_pipeline.py --input evaluation/ragas_dev.jsonl --output out/predictions.jsonl --mock
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict

# Ensure project root is on sys.path when running as a script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml

from evaluation.ragas import RAGASEvaluator, aggregate_results


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------ Mock components for fast tests ------
class MockRetriever:
    def retrieve(self, query: str, top_k: int = 5):
        # Return items that mimic DenseRetriever output
        return [
            {"doc_id": "doc1", "score": 1.0, "content": "Paris is the capital of France.", "source": "wiki:France", "page_start": 1},
            {"doc_id": "doc2", "score": 0.5, "content": "France has many cities.", "source": "wiki:France", "page_start": 2},
        ]


class MockReranker:
    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        # simple identity with rerank_score
        out = []
        for i, c in enumerate(candidates):
            c["rerank_score"] = float(c.get("score", 0.0))
            out.append(c)
        return sorted(out, key=lambda x: x["rerank_score"], reverse=True)


class MockFusion:
    def __init__(self, config):
        self.config = config

    def build_context(self, query: str, retrieved_chunks: List[Dict]) -> str:
        blocks = []
        for i, c in enumerate(retrieved_chunks, start=1):
            blocks.append(f"[CHUNK {i}]\n{c.get('content','')}\n")
        return "\n---\n".join(blocks)


class MockGenerator:
    def generate(self, query: str, fused_context: str) -> str:
        # simple rule-based
        if "France" in fused_context:
            return "Paris"
        return "I do not know based on the provided context."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--simple-eval", action="store_true", help="Skip generation, only eval retrieval+grounding")
    parser.add_argument("--num-queries", type=int, default=None, help="Number of queries to process")
    parser.add_argument("--random", action="store_true", help="Randomly sample queries instead of taking first N")
    parser.add_argument(
        "--ragas",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Compute official RAGAS metrics (faithfulness, answer relevancy, context precision/recall, etc.)",
    )
    parser.add_argument(
        "--force-ollama",
        action="store_true",
        help="Force Ollama-only generation (avoids OpenRouter rate limits during eval)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config) if os.path.exists(args.config) else {}

    if args.force_ollama:
        cfg.setdefault("generation", {})
        cfg["generation"]["provider"] = "ollama"

    if args.mock:
        retriever = MockRetriever()
        reranker = MockReranker()
        fusion = MockFusion(cfg)
        generator = MockGenerator()
    else:
        # Lazy import heavy modules
        from retrieval.dense_retriever import DenseRetriever
        from retrieval.reranker import Reranker
        from retrieval.sparse_retriever import SparseRetriever
        from fusion.context_fusion import ContextFusion
        from generation.grounded_llm import GroundedLLM

        try:
            retriever = DenseRetriever(cfg)
        except Exception as e:
            print(f"DenseRetriever init failed; falling back to SparseRetriever: {e}")
            retriever = SparseRetriever(cfg)

        try:
            reranker = Reranker(cfg)
        except Exception as e:
            print(f"Reranker init failed; using identity reranker: {e}")

            class IdentityReranker:
                def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
                    out = []
                    for c in candidates:
                        c["rerank_score"] = float(c.get("score", 0.0))
                        out.append(c)
                    return out

            reranker = IdentityReranker()

        fusion = ContextFusion(cfg)
        generator = GroundedLLM(cfg)

    # Configure evaluator
    llm_model = (
        cfg.get("planner", {}).get("generator", {}).get("local_model")
        or cfg.get("planner", {}).get("generator_model")
        or "llama3:8b"
    )
    embedding_model = cfg.get("models", {}).get("embedding_model")

    evaluator = RAGASEvaluator(
        use_ragas_framework=bool(args.ragas),
        llm_model=llm_model,
        embedding_model=embedding_model,
    )

    def normalize_for_eval(items: List[Dict]) -> List[Dict]:
        normalized: List[Dict] = []
        for r in items:
            doc_id = r.get("doc_id") or r.get("chunk_id") or r.get("id")
            excerpt = r.get("excerpt") or r.get("content") or r.get("snippet") or ""
            out = dict(r)
            out["doc_id"] = doc_id
            out["excerpt"] = excerpt
            normalized.append(out)
        return normalized

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load all queries
    all_queries = []
    with open(args.input, "r", encoding="utf-8") as inf:
        for line in inf:
            all_queries.append(json.loads(line))
    
    # Sample queries if requested
    if args.num_queries is not None:
        if args.random:
            queries_to_process = random.sample(all_queries, min(args.num_queries, len(all_queries)))
            print(f"📊 Randomly sampling {len(queries_to_process)} queries from {len(all_queries)} total")
        else:
            queries_to_process = all_queries[:args.num_queries]
            print(f"📊 Processing first {len(queries_to_process)} queries from {len(all_queries)} total")
    else:
        queries_to_process = all_queries
        print(f"📊 Processing all {len(queries_to_process)} queries")

    results = []
    with open(args.output, "w", encoding="utf-8") as outf:
        for item in queries_to_process:
            qid = item.get("query_id")
            query = item.get("query")

            retrieved = retriever.retrieve(query)
            reranked = reranker.rerank(query, retrieved)
            normalized_reranked = normalize_for_eval(reranked)
            fused = fusion.build_context(query, reranked)
            
            # Generate answer (skip if --simple-eval)
            if args.simple_eval:
                generated_text = "[SKIPPED in simple-eval mode]"
                generated = None
            else:
                try:
                    generated = generator.generate(query, fused)
                    if isinstance(generated, dict):
                        generated_text = generated.get("answer", "")
                    else:
                        generated_text = generated
                except Exception as gen_e:
                    # LLM failed; try extractive fallback
                    print(f"⚠️ Generation failed for '{query[:50]}...': {gen_e}")
                    try:
                        from generation.extractive_answerer import extract_answer
                        fallback_res = extract_answer(query, fused)
                        generated_text = fallback_res.get("answer", "")
                        print(f"✓ Used extractive fallback: {generated_text[:60]}")
                        generated = {"answer": generated_text, "source": "extractive_fallback"}
                    except Exception as extract_e:
                        print(f"✗ Extractive fallback also failed: {extract_e}")
                        generated_text = "[GENERATION FAILED]"
                        generated = None

            # Evaluate per-query (skip generation metrics if simple-eval)
            gold = item.get("gold", {})
            if args.simple_eval:
                # Only compute retrieval + grounding metrics
                eval_res = {
                    "query": query,
                    "generated": generated_text,
                    "reference": gold.get("answers", [""])[0] if gold.get("answers") else "",
                    "retrieval": evaluator.evaluate_retrieval(normalized_reranked, gold.get("evidence_ids", [])),
                    "generation": {},  # Skip for simple-eval
                    "grounding": evaluator.evidence_support_score(generated_text, normalized_reranked),
                    "correct": False,  # N/A for simple-eval
                }
            else:
                eval_res = evaluator.evaluate_query(
                    query,
                    generated_text,
                    normalized_reranked,
                    gold,
                    use_ragas=bool(args.ragas),
                )
            eval_res["query_id"] = qid
            results.append(eval_res)

            # Build output object (includes metrics)
            out = {
                "query_id": qid,
                "query": query,
                "generated": generated_text,
                "generated_structured": generated if isinstance(generated, dict) else None,
                "retrieved": normalized_reranked,
                "fused_context": fused,
                "metrics": eval_res,
            }
            outf.write(json.dumps(out, default=str) + "\n")

    # Aggregate and print summary
    summary = aggregate_results(results)
    print("\n" + "="*60)
    print("RAGAS EVALUATION SUMMARY")
    print("="*60)
    for k, v in summary.items():
        print(f"{k:.<40} {v:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
