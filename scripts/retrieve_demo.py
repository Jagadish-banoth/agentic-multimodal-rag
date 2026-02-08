import logging
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration."""
    config_path = ROOT / "config/settings.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_candidates_rrf(dense_hits: list, sparse_hits: list, k: int = 60) -> list:
    """Merge dense+sparse results using Reciprocal Rank Fusion (RRF).

    Mirrors the merge strategy used by the pipeline ExecutionEngine.
    RRF score: score = Σ 1/(k + rank_i), ranks are 1-indexed.
    """
    scores = {}

    for rank, hit in enumerate(dense_hits, start=1):
        chunk_id = hit.get("chunk_id", hit.get("index"))
        if not chunk_id:
            continue
        if chunk_id not in scores:
            scores[chunk_id] = {
                "result": hit,
                "rrf_score": 0.0,
                "dense_rank": None,
                "sparse_rank": None,
            }
        scores[chunk_id]["rrf_score"] += 1.0 / (k + rank)
        scores[chunk_id]["dense_rank"] = rank

    for rank, hit in enumerate(sparse_hits, start=1):
        chunk_id = hit.get("chunk_id", hit.get("index"))
        if not chunk_id:
            continue
        if chunk_id not in scores:
            scores[chunk_id] = {
                "result": hit,
                "rrf_score": 0.0,
                "dense_rank": None,
                "sparse_rank": None,
            }
        scores[chunk_id]["rrf_score"] += 1.0 / (k + rank)
        scores[chunk_id]["sparse_rank"] = rank

    merged = [
        {
            **item["result"],
            "fusion_score": item["rrf_score"],
            "dense_rank": item["dense_rank"],
            "sparse_rank": item["sparse_rank"],
        }
        for item in sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    ]
    return merged


def get_snippet(result: dict, max_len: int = 500) -> str:
    """Extract displayable snippet from result - show more content."""
    meta = result.get("metadata", {})
    
    # Try multiple fields - prefer full content
    content = (
            result.get("content") or
            meta.get("content") or
            result.get("snippet") or
            meta.get("snippet") or
            meta.get("caption") or
            meta.get("transcript") or
            f"[{meta.get('modality', result.get('modality', 'unknown'))} chunk]"
        )
    
    # Show content length info
    full_len = len(content) if content else 0
    snippet = content[:max_len].replace("\n", " ").strip()
    
    if full_len > max_len:
        snippet += f"... [{full_len} chars total]"
    
    return snippet


def format_result(rank: int, result: dict, show_full: bool = False) -> str:
    """Format a single result for display."""
    meta = result.get("metadata", {})
    chunk_id = result.get("chunk_id", "unknown")
    score = result.get(
        "rerank_score",
        result.get("fusion_score", result.get("weighted_score", result.get("score", 0)))
    )
    modality = (result.get("modality") or meta.get("modality") or "text").upper()
    index_type = result.get("index_type", result.get("retrieval_method", "unknown"))
    content_len = result.get("content_length", len(result.get("content", "")))
    
    # Check if this is a context-expanded chunk
    context_marker = ""
    if result.get("is_context"):
        context_marker = f" [{result.get('context_type', 'context')}]"
    
    snippet = get_snippet(result, max_len=500 if show_full else 300)
    
    output = f"\n{rank}. [{modality}]{context_marker} {chunk_id}\n"
    output += f"   Score: {score:.4f} | Type: {index_type} | Content: {content_len} chars\n"
    output += f"   {snippet}\n"
    output += "   " + "─" * 60
    
    return output


def main():
    """Interactive retrieval demo."""
    config = load_config()
    
    logger.info("Initializing pipeline-aligned retrievers...")

    # Match orchestrator/ExecutionEngine: DenseRetriever + SparseRetriever + Reranker
    from retrieval.dense_retriever import DenseRetriever
    from retrieval.sparse_retriever import SparseRetriever
    from retrieval.reranker import Reranker

    dense = DenseRetriever(config)
    sparse = SparseRetriever(config)
    reranker = Reranker(config)
    
    # FAANG-level retriever (new)
    faang_retriever = None
    try:
        from retrieval.faang_retriever import FAANGRetriever
        faang_retriever = FAANGRetriever(
            dense_retriever=dense,
            sparse_retriever=sparse,
            reranker=reranker,
            config=config
        )
        logger.info("✓ FAANG retriever initialized (full chunk + context expansion)")
    except Exception as e:
        logger.warning(f"FAANG retriever unavailable: {e}")

    # Optional Phase 1 modules (same as ExecutionEngine)
    query_processor = None
    try:
        from retrieval.query_processor import QueryProcessor
        query_processor = QueryProcessor(config)
        logger.info("✓ Query processor initialized")
    except Exception as e:
        logger.warning(f"Query processor unavailable: {e}")

    expander = None
    try:
        from retrieval.query_expansion import QueryExpander
        expander = QueryExpander(config)
        logger.info("✓ Query expander initialized")
    except Exception as e:
        logger.warning(f"Query expander unavailable: {e}")

    parallel_retriever = None
    try:
        from retrieval.parallel_retriever import ParallelRetriever
        parallel_retriever = ParallelRetriever(
            dense_retriever=dense,
            sparse_retriever=sparse,
            reranker=reranker,
            query_expander=expander,
            max_workers=4,
        )
        logger.info("✓ Parallel retriever initialized")
    except Exception as e:
        logger.warning(f"Parallel retriever unavailable: {e}")

    use_parallel = parallel_retriever is not None
    use_faang = faang_retriever is not None  # Default to FAANG mode if available
    
    # Get config parameters
    dense_k = config["retrieval"]["dense_k"]
    sparse_k = config["retrieval"]["sparse_k"]
    final_k = config["retrieval"]["rerank_k"]
    
    # Print index stats
    print("\n" + "=" * 70)
    print("MULTIMODAL RETRIEVAL DEMO (FAANG-LEVEL)")
    print("=" * 70)

    try:
        d_stats = dense.get_stats()
        print(f"\n📊 Index Stats:")
        if d_stats.get("dual_mode"):
            print(f"   Text Index:  {d_stats['text_index']['vectors']:,} vectors × {d_stats['text_index']['dim']}-dim")
            print(f"   Image Index: {d_stats['image_index']['vectors']:,} vectors × {d_stats['image_index']['dim']}-dim")
        else:
            print(f"   Legacy Index: {d_stats.get('total_vectors', 0):,} vectors")
        if faang_retriever:
            f_stats = faang_retriever.get_stats()
            print(f"   Full Chunks: {f_stats['total_chunks']:,} (with context expansion)")
    except Exception:
        pass
    
    print(f"\n⚙️  Retrieval Config:")
    print(f"   Dense K: {dense_k} | Sparse K: {sparse_k} | Final K: {final_k}")
    mode_str = "faang" if use_faang else ("parallel" if use_parallel else "sequential")
    print(f"   Mode: {mode_str} (type 'mode faang', 'mode parallel', or 'mode seq')")
    print(f"\nType 'exit' to quit, 'stats' for index stats")
    print("=" * 70)
    
    while True:
        try:
            query = input("\nQuery> ").strip()
            
            if not query:
                continue
            
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting...")
                break
            
            if query.lower() == "stats":
                try:
                    print(f"\nDense:  {dense.get_stats()}")
                except Exception:
                    print("\nDense:  (stats unavailable)")
                try:
                    print(f"Sparse: {sparse.get_stats()}")
                except Exception:
                    print("Sparse: (stats unavailable)")
                if faang_retriever:
                    print(f"FAANG:  {faang_retriever.get_stats()}")
                print(f"Reranker available: {getattr(reranker, 'is_available', lambda: False)()}")
                print(f"Mode: {'faang' if use_faang else ('parallel' if use_parallel else 'sequential')}")
                continue

            if query.lower() in ("mode seq", "mode sequential"):
                use_parallel = False
                use_faang = False
                print("\nMode set to sequential.")
                continue

            if query.lower() in ("mode parallel", "mode par"):
                if parallel_retriever is None:
                    print("\n❌ Parallel mode unavailable (ParallelRetriever not initialized).")
                else:
                    use_parallel = True
                    use_faang = False
                    print("\nMode set to parallel.")
                continue
            
            if query.lower() in ("mode faang", "mode full"):
                if faang_retriever is None:
                    print("\n❌ FAANG mode unavailable (FAANGRetriever not initialized).")
                else:
                    use_faang = True
                    print("\n✓ Mode set to FAANG (full chunks + context expansion).")
                continue
            
            print("\n" + "─" * 70)
            print(f"Processing: {query}")
            print("─" * 70)
            
            # Query preprocessing (matches ExecutionEngine behavior)
            if query_processor:
                processed = query_processor.preprocess(query)
                if processed:
                    query = processed

            timings = {}
            t0 = time.time()
            
            # Retrieval: FAANG mode (best), parallel, or sequential
            if use_faang and faang_retriever is not None:
                # FAANG-level retrieval with full chunks and context expansion
                final = faang_retriever.retrieve(
                    query,
                    top_k=final_k,
                    expand_context=True,
                    use_sparse=True
                )
                timings["faang_retrieval"] = time.time() - t0
                timings["retrieval_total"] = timings["faang_retrieval"]
                logger.info(f"FAANG retrieval: {len(final)} results with full content")
                
            elif use_parallel and parallel_retriever is not None:
                final, timings = parallel_retriever.retrieve_parallel(
                    query,
                    dense_k=dense_k,
                    sparse_k=sparse_k,
                    rerank_k=final_k,
                    use_expansion=True,
                    expand_timeout=2.0,
                )
                timings["retrieval_total"] = timings.get("total", time.time() - t0)
            else:
                # Dense
                t_dense = time.time()
                dense_hits = dense.retrieve(query, top_k=dense_k)
                timings["dense"] = time.time() - t_dense
                logger.info(f"Dense: {len(dense_hits)} results")

                # Sparse
                t_sparse = time.time()
                sparse_hits = sparse.retrieve(query, top_k=sparse_k)
                timings["sparse"] = time.time() - t_sparse
                logger.info(f"Sparse: {len(sparse_hits)} results")

                # Merge (RRF)
                t_merge = time.time()
                candidates = merge_candidates_rrf(dense_hits, sparse_hits)
                timings["merge"] = time.time() - t_merge
                logger.info(f"Merged: {len(candidates)} unique results")

                # Rerank
                t_rerank = time.time()
                if getattr(reranker, "is_available", lambda: False)() and candidates:
                    final = reranker.rerank(query, candidates, top_n=final_k)
                else:
                    final = candidates[:final_k]
                timings["reranking"] = time.time() - t_rerank
                timings["retrieval_total"] = time.time() - t0

            # Ensure top-k
            final = final[:final_k]
            
            # Display results
            if not final:
                print("\n❌ No results found.")
                continue

            if timings:
                timing_str = " | ".join([f"{k}: {v:.2f}s" for k, v in timings.items() if isinstance(v, (int, float))])
                if timing_str:
                    print(f"\n⏱️  {timing_str}")
            
            print(f"\n✅ Top {len(final)} Results:")
            for i, result in enumerate(final, 1):
                print(format_result(i, result))
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting...")
            break
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()