# evaluation/eval_retrieval.py
"""
Evaluation script for Day 6
Computes Recall@K and MRR for hybrid retrieval + reranking
"""

import csv
import yaml
import statistics

from retrieval.dense_retriever import DenseRetriever
from retrieval.sparse_retriever import SparseRetriever
from retrieval.reranker import Reranker


# ----------------------------
# Load config
# ----------------------------
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from evaluation import metrics

with open("config/settings.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# default values from config
dense_k = cfg["retrieval"]["dense_k"]
sparse_k = cfg["retrieval"]["sparse_k"]
rerank_k = cfg["retrieval"]["rerank_k"]


# ----------------------------
# Initialize retrievers (one-time)
# ----------------------------
# create inside main for better testability; keep module level defaults for interactive use

def _init_retrievers(config):
    d = DenseRetriever(config)
    s = SparseRetriever(config)
    r = Reranker(config)
    return d, s, r


# ----------------------------
# Metrics per query
# ----------------------------

def recall_mrr_at_k(dense, sparse, reranker, query: str, gold_chunk_id: str, k: int, dense_top_k: int, sparse_top_k: int):
    # Dense + sparse retrieval
    dense_hits = dense.retrieve(query, top_k=dense_top_k)
    sparse_hits = sparse.retrieve(query, top_k=sparse_top_k)

    # Merge candidates (unique by chunk_id)
    seen = set()
    candidates = []
    for h in dense_hits + sparse_hits:
        cid = h["chunk_id"]
        if cid not in seen:
            candidates.append(h)
            seen.add(cid)

    # Rerank (cross-encoder batching is handled inside reranker)
    reranked = reranker.rerank(query, candidates)

    topk = [c["chunk_id"] for c in reranked[:k]]

    recall = metrics.recall_at_k(topk, gold_chunk_id)
    mrr = metrics.mrr_at_k(topk, gold_chunk_id)
    return recall, mrr


# ----------------------------
# Main
# ----------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate retrieval recall@K and MRR on dev_queries.csv")
    parser.add_argument("--queries", default="evaluation/dev_queries.csv", help="Path to CSV with columns: query,expected_chunk_id")
    parser.add_argument("--k", type=int, default=rerank_k, help="Top-K to evaluate (Recall@K and MRR)" )
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads to evaluate queries in parallel")
    parser.add_argument("--out", default=None, help="Optional CSV file to write per-query metrics (query,expected_chunk_id,recall,mrr)")
    args = parser.parse_args(argv)

    qpath = args.queries

    # load queries
    rows = []
    try:
        with open(qpath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append((r["query"], r["expected_chunk_id"]))
    except FileNotFoundError:
        raise RuntimeError(f"Queries file not found: {qpath}. Create a CSV with columns 'query' and 'expected_chunk_id'.")

    if not rows:
        raise RuntimeError(f"Queries file {qpath} is empty or malformed")

    # init retrievers
    dense, sparse, reranker = _init_retrievers(cfg)

    recalls = []
    mrrs = []

    # optional output file
    out_rows = []

    # parallel evaluation; ThreadPoolExecutor is fine because retrievers/reranker are read-only and perform CPU/IO-bound work
    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(recall_mrr_at_k, dense, sparse, reranker, q, g, args.k, dense_k, sparse_k): (q, g) for q, g in rows}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            q, g = futures[fut]
            try:
                rec, mrr = fut.result()
            except Exception as e:
                logging.exception(f"Error evaluating query: {q}")
                rec, mrr = 0.0, 0.0
            recalls.append(rec)
            mrrs.append(mrr)
            out_rows.append({"query": q, "expected_chunk_id": g, "recall": rec, "mrr": mrr})

    # write optional per-query CSV
    if args.out:
        import csv as _csv

        with open(args.out, "w", encoding="utf-8", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=["query", "expected_chunk_id", "recall", "mrr"])
            writer.writeheader()
            for r in out_rows:
                writer.writerow(r)

    print("\n=== Retrieval Evaluation ===")
    print(f"Queries evaluated : {len(rows)}")
    print(f"Recall@{args.k}       : {metrics.aggregate(recalls):.4f}")
    print(f"MRR              : {metrics.aggregate(mrrs):.4f}")


if __name__ == "__main__":
    main()
