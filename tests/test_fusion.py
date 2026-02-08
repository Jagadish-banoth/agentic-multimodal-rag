"""Ad-hoc fusion demo.

This file is intended as a runnable script. It is safe to import so CI/pytest
won't accidentally run a heavy end-to-end pipeline.
"""


def main() -> None:
    import yaml

    from retrieval.dense_retriever import DenseRetriever
    from retrieval.sparse_retriever import SparseRetriever
    from retrieval.reranker import Reranker
    from fusion.context_fusion import ContextFusion

    cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))

    dense = DenseRetriever(cfg)
    sparse = SparseRetriever(cfg)
    reranker = Reranker(cfg)
    fusion = ContextFusion(cfg)

    query = "What is Self-Attention?"

    dense_hits = dense.retrieve(query)
    sparse_hits = sparse.retrieve(query)

    seen = set()
    candidates = []
    for h in dense_hits + sparse_hits:
        if h["chunk_id"] not in seen:
            candidates.append(h)
            seen.add(h["chunk_id"])

    reranked = reranker.rerank(query, candidates)
    context = fusion.build_context(reranked, query)

    print("\n===== FINAL FUSED CONTEXT =====\n")
    print(context)


if __name__ == "__main__":
    main()
