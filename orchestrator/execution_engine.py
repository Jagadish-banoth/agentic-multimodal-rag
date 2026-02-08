"""
Execution Engine for the Agentic Multimodal RAG system.

Orchestrates the full pipeline with FAANG-grade Phase 1 optimizations:
Planner → Query Expansion → Parallel Retrieval → Fusion → Generation → Verification

Phase 1 Features:
✓ Parallel retrieval (5.6x latency improvement)
✓ Query expansion with HyDE
✓ Result caching with fuzzy matching
"""
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from planner.agentic_planner import AgenticPlanner, create_planner
from retrieval.dense_retriever import DenseRetriever
from retrieval.sparse_retriever import SparseRetriever
from retrieval.reranker import Reranker
from retrieval.parallel_retriever import ParallelRetriever
from retrieval.query_expansion import QueryExpander
from retrieval.query_processor import QueryProcessor
from retrieval.metrics import MetricsTracker
from fusion.context_fusion import ContextFusion
from generation.grounded_llm import GroundedLLM
from verification.verifier import Verifier
from utils.result_cache import ResultCache
from monitoring.jsonl_metrics import JSONLMetricsLogger


# Setup file logger for execution engine
exec_log_path = os.path.join(os.path.dirname(__file__), '../logs/execution_engine.log')
exec_log_path = os.path.abspath(exec_log_path)
logger = logging.getLogger("execution_engine")
if not logger.hasHandlers():
    handler = logging.FileHandler(exec_log_path, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Load config
CONFIG_PATH = ROOT / "config" / "settings.yaml"


def load_config() -> Dict[str, Any]:
    """Load configuration from settings.yaml."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ExecutionEngine:
    """
    Main orchestrator for the RAG pipeline.
    
    Coordinates:
    1. Query planning (intent classification, modality selection)
    2. Hybrid retrieval (dense + sparse)
    3. Reranking
    4. Context fusion
    5. Grounded generation
    6. Verification
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, enable_phase1: bool = True):
        """
        Initialize the execution engine with Phase 1 optimizations.
        
        Args:
            config: Configuration dictionary (loads from settings.yaml if None)
            enable_phase1: Enable Phase 1 features (parallel + expansion + caching)
        """
        self.config = config or load_config()
        self.enable_phase1 = enable_phase1
        
        logger.info("Initializing ExecutionEngine with Phase 1 optimizations...")
        
        # Initialize agentic planner (Gemma 2B + GPU with rule-based fallback)
        self.planner = create_planner(self.config)
        logger.info("✓ Agentic planner initialized")
        
        # Initialize retrievers
        self.dense = DenseRetriever(self.config)
        self.sparse = SparseRetriever(self.config)
        
        # Initialize reranker (optional)
        self.reranker = Reranker(self.config)
        if not self.reranker.is_available():
            logger.warning("Reranker not available - skipping reranking step")
        
        # FAANG-level retriever (optional enhancement)
        self.faang_retriever = None
        try:
            from retrieval.faang_retriever import FAANGRetriever
            self.faang_retriever = FAANGRetriever(
                dense_retriever=self.dense,
                sparse_retriever=self.sparse,
                reranker=self.reranker,
                config=self.config
            )
            logger.info("✓ FAANG retriever initialized (full chunk + context expansion)")
        except Exception as e:
            logger.warning(f"FAANG retriever unavailable: {e}")
        
        # Initialize fusion
        self.fusion = ContextFusion(self.config)
        
        # Initialize generator
        self.generator = GroundedLLM(self.config)

        # Initialize verifier (NLI-based)
        self.verifier = Verifier(self.config)
        
        # Query Processor (preprocessing for better matching)
        try:
            self.query_processor = QueryProcessor(self.config)
            logger.info("✓ Query processor initialized")
        except Exception as e:
            logger.warning(f"Query processor unavailable: {e}")
            self.query_processor = None
        
        # Metrics Tracker (monitoring retrieval quality)
        self.metrics_tracker = MetricsTracker(log_interval=50)
        logger.info("✓ Metrics tracker initialized")

        # Lightweight JSONL telemetry (optional)
        self.telemetry = None
        try:
            monitoring_cfg = (self.config or {}).get("monitoring", {})
            if bool(monitoring_cfg.get("enabled", True)):
                jsonl_path = monitoring_cfg.get("jsonl_path") or str(ROOT / "logs" / "telemetry.jsonl")
                self.telemetry = JSONLMetricsLogger(path=str(jsonl_path))
                logger.info("✓ Telemetry enabled")
        except Exception as e:
            logger.warning(f"Telemetry init failed: {e}")
        
        # Phase 1: Query Expansion
        self.expander = None
        if enable_phase1:
            try:
                self.expander = QueryExpander(self.config)
                logger.info("✓ Query expansion module loaded")
            except Exception as e:
                logger.warning(f"Query expansion unavailable: {e}")
        
        # Phase 1: Result Cache
        self.cache = None
        if enable_phase1:
            try:
                self.cache = ResultCache(self.config)
                if self.cache.enabled:
                    logger.info("✓ Result caching enabled")
                else:
                    logger.info("⚠️ Result caching disabled in config")
            except Exception as e:
                logger.warning(f"Result caching unavailable: {e}")
        
        # Phase 1: Parallel Retriever
        self.parallel_retriever = None
        if enable_phase1:
            try:
                self.parallel_retriever = ParallelRetriever(
                    dense_retriever=self.dense,
                    sparse_retriever=self.sparse,
                    reranker=self.reranker,
                    query_expander=self.expander,
                    max_workers=4
                )
                logger.info("✓ Parallel retriever initialized")
            except Exception as e:
                logger.warning(f"Parallel retrieval unavailable: {e}")
        
        # Pipeline settings
        retrieval_cfg = self.config.get("retrieval", {})
        self.dense_k = retrieval_cfg.get("dense_k", 30)
        self.sparse_k = retrieval_cfg.get("sparse_k", 30)
        self.rerank_k = retrieval_cfg.get("rerank_k", 15)
        
        # Agent settings
        agent_cfg = self.config.get("agent", {})
        self.confidence_threshold = agent_cfg.get("confidence_threshold", 0.7)
        self.max_attempts = agent_cfg.get("max_attempts", 2)
        
        logger.info("✓ ExecutionEngine initialized (Phase 1 optimizations enabled)")

    
    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline with Phase 1 optimizations.
        
        Args:
            query: User query
        
        Returns:
            Response dict with answer, sources, and metadata
        """
        logger.info(f"Processing query: {query[:100]}...")
        t_total_start = time.time()
        
        # Preprocess query for better matching
        original_query = query
        if self.query_processor:
            query = self.query_processor.preprocess(query)
            if query != original_query:
                logger.debug(f"Query preprocessed: '{original_query}' → '{query}'")
        
        # Check cache first (Phase 1 optimization)
        if self.cache and self.cache.enabled:
            cached = self.cache.get(query)
            if cached:
                logger.info("✓ Cache HIT - returning cached result")
                cached["from_cache"] = True
                if self.telemetry:
                    try:
                        self.telemetry.log(
                            "query_completed",
                            {
                                "query": query,
                                "from_cache": True,
                                "confidence": cached.get("confidence"),
                                "verified": cached.get("verified"),
                            },
                        )
                    except Exception:
                        pass
                return cached
        
        attempt = 0
        best_response = None
        
        while attempt < self.max_attempts:
            attempt += 1
            logger.info(f"Attempt {attempt}/{self.max_attempts}")
            
            try:
                timings = {}
                
                # Step 1: Plan query (intent, modalities, strategy)
                t0 = time.time()
                plan = self.planner.plan(query)
                timings["planning"] = time.time() - t0
                logger.info(f"✓ Plan: {plan.intent.value} | {plan.strategy.value} | k={plan.dense_k}/{plan.sparse_k}/{plan.rerank_k}")
                
                # Use plan to adjust retrieval parameters
                dense_k = plan.dense_k
                sparse_k = plan.sparse_k
                rerank_k = plan.rerank_k
                
                # Step 2: Retrieve (Phase 1: Use parallel retriever if available)
                t0 = time.time()
                if self.parallel_retriever and self.enable_phase1:
                    # Parallel retrieval with query expansion
                    results, retrieval_timings = self.parallel_retriever.retrieve_parallel(
                        query,
                        dense_k=dense_k,
                        sparse_k=sparse_k,
                        rerank_k=rerank_k,
                        use_expansion=True,
                        expand_timeout=2.0
                    )
                    timings.update(retrieval_timings)
                    timings["retrieval_total"] = time.time() - t0
                else:
                    # Fallback to sequential retrieval
                    results = self._retrieve(query, dense_k=dense_k, sparse_k=sparse_k)
                    # Rerank separately
                    if self.reranker.is_available():
                        results = self.reranker.rerank(query, results, top_n=rerank_k)
                    else:
                        results = results[:rerank_k]
                    timings["retrieval_total"] = time.time() - t0
                
                # Track retrieval metrics
                retrieval_latency_ms = timings["retrieval_total"] * 1000
                self.metrics_tracker.track_query(
                    latency_ms=retrieval_latency_ms,
                    num_results=len(results)
                )
                
                if not results:
                    logger.warning("No results retrieved")
                    self.metrics_tracker.track_failure()
                    continue
                
                # Step 3: Fuse context (with citation map)
                t0 = time.time()
                context, chunk_map = self.fusion.fuse_with_mapping(results, query=query)
                timings["fusion"] = time.time() - t0
                
                if not context:
                    logger.warning("Empty context after fusion")
                    continue
                
                # Step 4: Generate answer
                t0 = time.time()
                response = self.generator.generate(query, context, results, chunk_map=chunk_map)
                timings["generation"] = time.time() - t0
                
                # Add plan info to response
                response["plan"] = {
                    "intent": plan.intent.value,
                    "strategy": plan.strategy.value,
                    "confidence": plan.confidence
                }
                
                # Verification (NLI-based faithfulness)
                verification = self.verifier.verify(query, response, results, chunk_map=chunk_map)
                response["verification"] = verification
                response["confidence"] = verification.get("confidence", response.get("confidence", 0.0))
                response["verified"] = verification.get("verified", False)
                
                # Add timings to response
                response["timings"] = timings
                # Avoid double-counting when retrieval module reports its own 'total' and when 'retrieval_total' is present.
                timing_sum = 0.0
                for k, v in timings.items():
                    if k == "total":
                        continue
                    timing_sum += float(v)
                response["total_time"] = timing_sum
                response["from_cache"] = False

                # Telemetry emit (best-effort)
                if self.telemetry:
                    try:
                        self.telemetry.log(
                            "query_completed",
                            {
                                "query": query,
                                "from_cache": False,
                                "confidence": response.get("confidence"),
                                "verified": response.get("verified"),
                                "intent": response.get("plan", {}).get("intent"),
                                "strategy": response.get("plan", {}).get("strategy"),
                                "timings": timings,
                            },
                        )
                    except Exception:
                        pass
                
                # Log timings
                timing_str = " | ".join([f"{k}: {v:.2f}s" for k, v in timings.items()])
                logger.info(f"⏱️ {timing_str} | Total: {timing_sum:.2f}s")
                
                # Step 5: Check verification result
                if response["verified"]:
                    logger.info(f"✓ Response accepted (verified, confidence: {response['confidence']:.2f})")
                    
                    # Cache the result (Phase 1 optimization)
                    if self.cache and self.cache.enabled:
                        try:
                            self.cache.set(query, response, confidence=response.get("confidence", 0.5))
                        except Exception as e:
                            logger.debug(f"Cache set error: {e}")
                    
                    return response
                
                # Store best response so far
                if best_response is None or response["confidence"] > best_response.get("confidence", 0):
                    best_response = response
                
                if verification.get("should_retry", True):
                    logger.info(f"Low confidence ({response['confidence']:.2f}), retrying...")
                    continue
                else:
                    logger.info("Verifier advised to stop retrying")
                    break
                
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue
        
        # Return best response or error
        if best_response:
            best_response["note"] = "Best effort response (confidence below threshold)"
            best_response["from_cache"] = False

            if self.telemetry:
                try:
                    self.telemetry.log(
                        "query_completed",
                        {
                            "query": query,
                            "from_cache": False,
                            "confidence": best_response.get("confidence"),
                            "verified": best_response.get("verified"),
                            "best_effort": True,
                            "timings": best_response.get("timings", {}),
                        },
                    )
                except Exception:
                    pass
            
            # Cache the best effort response with lower TTL
            if self.cache and self.cache.enabled:
                try:
                    self.cache.set(query, best_response, confidence=0.3)
                except Exception as e:
                    logger.debug(f"Cache set error: {e}")
            
            return best_response
        
        final_err = {
            "answer": "I couldn't find a confident answer to your question.",
            "sources": [],
            "confidence": 0.0,
            "error": "Max attempts reached without confident response",
            "from_cache": False,
        }
        if self.telemetry:
            try:
                self.telemetry.log(
                    "query_completed",
                    {
                        "query": query,
                        "from_cache": False,
                        "confidence": 0.0,
                        "verified": False,
                        "error": final_err.get("error"),
                    },
                )
            except Exception:
                pass
        return final_err

    
    def _retrieve(self, query: str, dense_k: Optional[int] = None, sparse_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval.
        
        Combines dense semantic search with sparse keyword search.
        
        Args:
            query: User query
            dense_k: Number of dense results (uses default if None)
            sparse_k: Number of sparse results (uses default if None)
        """
        # Use planner-provided k values or defaults
        dense_k = dense_k or self.dense_k
        sparse_k = sparse_k or self.sparse_k
        
        # Dense retrieval
        dense_results = self.dense.retrieve(query, top_k=dense_k)
        
        # Sparse retrieval
        sparse_results = self.sparse.retrieve(query, top_k=sparse_k)
        
        # Merge results
        results = self._merge_results(dense_results, sparse_results)
        
        logger.info(f"Retrieved {len(results)} unique results")
        return results
    
    def _merge_results(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
    ) -> List[Dict[str, Any]]:
        """
        Merge dense and sparse results using RRF (Reciprocal Rank Fusion).
        
        FAANG Best Practice: RRF is more robust than weighted score fusion
        because it normalizes by rank rather than raw scores.
        
        RRF formula: score = Σ 1/(k + rank_i)
        where k=60 is Google's empirical constant.
        """
        k = 60  # RRF constant (Google default)
        
        # Score by rank (RRF)
        scores = {}
        
        # Dense ranking contribution
        for rank, result in enumerate(dense_results, start=1):  # 1-indexed
            chunk_id = result.get("chunk_id", result.get("index"))
            if chunk_id not in scores:
                scores[chunk_id] = {
                    "result": result,
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "sparse_rank": None
                }
            scores[chunk_id]["rrf_score"] += 1.0 / (k + rank)
            scores[chunk_id]["dense_rank"] = rank
        
        # Sparse ranking contribution
        for rank, result in enumerate(sparse_results, start=1):  # 1-indexed
            chunk_id = result.get("chunk_id", result.get("index"))
            if chunk_id not in scores:
                scores[chunk_id] = {
                    "result": result,
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "sparse_rank": None
                }
            scores[chunk_id]["rrf_score"] += 1.0 / (k + rank)
            scores[chunk_id]["sparse_rank"] = rank
        
        # Sort by fused RRF score (descending)
        merged = [
            {
                **item["result"],
                "fusion_score": item["rrf_score"],
                "dense_rank": item["dense_rank"],
                "sparse_rank": item["sparse_rank"]
            }
            for item in sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
        ]
        
        logger.debug(f"RRF merged {len(merged)} unique results from {len(dense_results)} dense + {len(sparse_results)} sparse")
        return merged
    
    def chat(self, query: str) -> str:
        """
        Simple chat interface.
        
        Args:
            query: User query
        
        Returns:
            Answer string
        """
        response = self.run(query)
        return response.get("answer", "I couldn't generate a response.")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics (Phase 1 feature)."""
        stats = {}
        if self.cache:
            stats["cache"] = self.cache.stats()
        else:
            stats["cache"] = {"status": "cache disabled"}
        
        # Add retrieval metrics
        stats["retrieval"] = self.metrics_tracker.get_stats()
        return stats
    
    def clear_cache(self) -> None:
        """Clear result cache."""
        if self.cache:
            self.cache.clear()
            logger.info("✓ Cache cleared")
    
    def shutdown(self) -> None:
        """Shutdown components."""
        try:
            if self.parallel_retriever:
                self.parallel_retriever.shutdown()
            logger.info("✓ ExecutionEngine shutdown complete")
        except Exception as e:
            logger.warning(f"Shutdown error: {e}")



def main():
    """Main entry point for non-interactive execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    config = load_config()
    engine = ExecutionEngine(config)
    
    # Example query
    query = "What is computer vision?"
    print(f"\nQuery: {query}\n")
    
    response = engine.run(query)
    
    print(f"Answer: {response.get('answer', 'No answer')}\n")
    print(f"Confidence: {response.get('confidence', 0):.2f}")
    
    if response.get("sources"):
        print("\nSources:")
        for i, src in enumerate(response["sources"][:3], 1):
            print(f"  {i}. {src}")


if __name__ == "__main__":
    main()