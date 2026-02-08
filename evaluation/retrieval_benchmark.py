"""FAANG-style retrieval benchmark.

What it does:
- Loads a ground-truth query set (CSV or JSONL)
- Benchmarks multiple retriever variants (dense/sparse/hybrid/parallel/faang)
- Computes Recall@K, MRR, nDCG@K
- Tracks latency percentiles (p50/p95/p99)

Input formats:
1) CSV (simple): columns: query, expected_chunk_id
   Optional columns:
   - expected_chunk_ids: semicolon-separated list
   - relevance_json: JSON object mapping chunk_id -> graded relevance

2) JSONL (richer): one object per line:
   {
     "query_id": "q1",
     "query": "...",
     "gold": {
       "evidence_ids": ["chunk123", "chunk987"],
       "relevance": {"chunk123": 3, "chunk987": 1}
     }
   }

Usage:
  python -m evaluation.retrieval_benchmark --input evaluation/dev_queries.csv --k 5 10
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from evaluation import metrics


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "settings.yaml"


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class QueryCase:
    query_id: str
    query: str
    gold_ids: List[str]
    relevance: Dict[str, float]


def _parse_csv(path: str) -> List[QueryCase]:
    rows: List[QueryCase] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            q = (r.get("query") or "").strip()
            if not q:
                continue

            qid = (r.get("query_id") or f"row_{i}").strip()
            expected = (r.get("expected_chunk_id") or "").strip()
            expected_many = (r.get("expected_chunk_ids") or "").strip()
            rel_json = (r.get("relevance_json") or "").strip()

            gold_ids: List[str] = []
            relevance: Dict[str, float] = {}

            if rel_json:
                try:
                    relevance = {str(k): float(v) for k, v in json.loads(rel_json).items()}
                    gold_ids = list(relevance.keys())
                except Exception:
                    relevance = {}

            if not gold_ids:
                if expected_many:
                    gold_ids = [x.strip() for x in expected_many.split(";") if x.strip()]
                elif expected:
                    gold_ids = [expected]

            if not relevance:
                relevance = {gid: 1.0 for gid in gold_ids}

            rows.append(QueryCase(query_id=qid, query=q, gold_ids=gold_ids, relevance=relevance))

    if not rows:
        raise RuntimeError(f"No valid rows in CSV: {path}")
    return rows


def _parse_jsonl(path: str) -> List[QueryCase]:
    rows: List[QueryCase] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = (obj.get("query") or "").strip()
            if not q:
                continue
            qid = str(obj.get("query_id") or f"line_{i}")
            gold = obj.get("gold") or {}
            ids = list(gold.get("evidence_ids") or [])
            rel = gold.get("relevance") or {}
            relevance = {str(k): float(v) for k, v in rel.items()} if isinstance(rel, dict) else {}
            if not relevance:
                relevance = {gid: 1.0 for gid in ids}
            rows.append(QueryCase(query_id=qid, query=q, gold_ids=ids, relevance=relevance))

    if not rows:
        raise RuntimeError(f"No valid rows in JSONL: {path}")
    return rows


def load_cases(path: str) -> List[QueryCase]:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return _parse_csv(path)
    if p.suffix.lower() in {".jsonl", ".json"}:
        return _parse_jsonl(path)
    raise ValueError(f"Unsupported input format: {path}")


class RetrieverAdapter:
    name: str

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        raise NotImplementedError


class DenseAdapter(RetrieverAdapter):
    name = "dense"

    def __init__(self, cfg: Dict[str, Any]):
        from retrieval.dense_retriever import DenseRetriever

        self._r = DenseRetriever(cfg)

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        return self._r.retrieve(query, top_k=top_k)


class SparseAdapter(RetrieverAdapter):
    name = "sparse"

    def __init__(self, cfg: Dict[str, Any]):
        from retrieval.sparse_retriever import SparseRetriever

        self._r = SparseRetriever(cfg)

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        return self._r.retrieve(query, top_k=top_k)


class HybridRerankAdapter(RetrieverAdapter):
    name = "hybrid_rerank"

    def __init__(self, cfg: Dict[str, Any]):
        from retrieval.dense_retriever import DenseRetriever
        from retrieval.sparse_retriever import SparseRetriever
        from retrieval.reranker import Reranker

        self._dense = DenseRetriever(cfg)
        self._sparse = SparseRetriever(cfg)
        self._reranker = Reranker(cfg)
        self._dense_k = int(cfg.get("retrieval", {}).get("dense_k", 50))
        self._sparse_k = int(cfg.get("retrieval", {}).get("sparse_k", 50))

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        dense_hits = self._dense.retrieve(query, top_k=self._dense_k)
        sparse_hits = self._sparse.retrieve(query, top_k=self._sparse_k)

        seen = set()
        candidates: List[Dict[str, Any]] = []
        for h in dense_hits + sparse_hits:
            cid = h.get("chunk_id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            candidates.append(h)

        if getattr(self._reranker, "is_available", lambda: False)() and candidates:
            reranked = self._reranker.rerank(query, candidates, top_n=top_k)
            return reranked[:top_k]
        return candidates[:top_k]


class ParallelAdapter(RetrieverAdapter):
    name = "parallel"

    def __init__(self, cfg: Dict[str, Any]):
        from retrieval.dense_retriever import DenseRetriever
        from retrieval.sparse_retriever import SparseRetriever
        from retrieval.reranker import Reranker
        from retrieval.parallel_retriever import ParallelRetriever

        dense = DenseRetriever(cfg)
        sparse = SparseRetriever(cfg)
        reranker = Reranker(cfg)

        # QueryExpander is optional inside ParallelRetriever usage; keep None to avoid heavy deps here.
        self._pr = ParallelRetriever(
            dense_retriever=dense,
            sparse_retriever=sparse,
            reranker=reranker,
            query_expander=None,
            max_workers=4,
        )

        retrieval_cfg = cfg.get("retrieval", {})
        self._dense_k = int(retrieval_cfg.get("dense_k", 50))
        self._sparse_k = int(retrieval_cfg.get("sparse_k", 50))

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        results, _timings = self._pr.retrieve_parallel(
            query,
            dense_k=self._dense_k,
            sparse_k=self._sparse_k,
            rerank_k=top_k,
            use_expansion=False,
        )
        return results[:top_k]


class FAANGAdapter(RetrieverAdapter):
    name = "faang"

    def __init__(self, cfg: Dict[str, Any]):
        from retrieval.dense_retriever import DenseRetriever
        from retrieval.sparse_retriever import SparseRetriever
        from retrieval.reranker import Reranker
        from retrieval.faang_retriever import FAANGRetriever

        dense = DenseRetriever(cfg)
        sparse = SparseRetriever(cfg)
        reranker = Reranker(cfg)
        self._fr = FAANGRetriever(dense, sparse, reranker, cfg)

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        return self._fr.retrieve(query, top_k=top_k, expand_context=True, use_sparse=True)


def build_retrievers(names: Sequence[str], cfg: Dict[str, Any]) -> List[RetrieverAdapter]:
    mapping = {
        "dense": DenseAdapter,
        "sparse": SparseAdapter,
        "hybrid_rerank": HybridRerankAdapter,
        "parallel": ParallelAdapter,
        "faang": FAANGAdapter,
    }
    out: List[RetrieverAdapter] = []
    for n in names:
        if n not in mapping:
            raise ValueError(f"Unknown retriever: {n}. Choose from: {sorted(mapping)}")
        out.append(mapping[n](cfg))
    return out


def _ids_from_hits(hits: List[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    for h in hits:
        cid = h.get("chunk_id") or h.get("doc_id") or h.get("id")
        if cid is not None:
            ids.append(str(cid))
    return ids


def evaluate_one(
    adapter: RetrieverAdapter,
    case: QueryCase,
    ks: Sequence[int],
) -> Tuple[Dict[str, float], float, int]:
    # Retrieve at max k once; slice for metrics
    max_k = max(ks)
    t0 = time.perf_counter()
    hits = adapter.retrieve(case.query, top_k=max_k)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    ids = _ids_from_hits(hits)

    out: Dict[str, float] = {}
    for k in ks:
        out[f"recall_hit@{k}"] = metrics.recall_hit_at_k(ids, case.gold_ids, k)
        out[f"recall@{k}"] = metrics.recall_fraction_at_k(ids, case.gold_ids, k)
        out[f"ndcg@{k}"] = metrics.ndcg_at_k(ids, case.relevance, k)
    out["mrr"] = metrics.mrr(ids, case.gold_ids)

    return out, latency_ms, len(hits)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="FAANG-style retrieval benchmark")
    parser.add_argument("--input", required=True, help="CSV or JSONL ground-truth file")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to config/settings.yaml")
    parser.add_argument("--retrievers", nargs="+", default=["dense", "sparse", "hybrid_rerank", "parallel", "faang"])
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--out", default="artifacts/retrieval_benchmark.json", help="Output JSON report")
    parser.add_argument("--per_query_out", default=None, help="Optional JSONL per-query output")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cases = load_cases(args.input)
    retrievers = build_retrievers(args.retrievers, cfg)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    per_query_fp = open(args.per_query_out, "w", encoding="utf-8") if args.per_query_out else None

    report: Dict[str, Any] = {
        "input": args.input,
        "ks": list(args.k),
        "num_cases": len(cases),
        "retrievers": {},
    }

    for adapter in retrievers:
        rows: List[Dict[str, Any]] = []
        latencies: List[float] = []

        for case in cases:
            m, latency_ms, num_results = evaluate_one(adapter, case, args.k)
            latencies.append(latency_ms)
            row = {
                "query_id": case.query_id,
                "retriever": adapter.name,
                "latency_ms": latency_ms,
                "num_results": num_results,
                **m,
            }
            rows.append(row)
            if per_query_fp:
                per_query_fp.write(json.dumps(row, ensure_ascii=False) + "\n")

        # aggregate
        agg: Dict[str, float] = {}
        if rows:
            keys = [k for k in rows[0].keys() if k not in {"query_id", "retriever"}]
            for k in keys:
                if k in {"latency_ms", "num_results"}:
                    continue
                agg[k] = metrics.aggregate(r.get(k, 0.0) for r in rows)

        report["retrievers"][adapter.name] = {
            "metrics_mean": agg,
            "latency_ms": metrics.summarize_latency_ms(latencies),
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if per_query_fp:
        per_query_fp.close()

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
