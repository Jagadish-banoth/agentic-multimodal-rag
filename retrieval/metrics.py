"""Retrieval performance tracking and metrics."""
import logging
import time
from typing import List, Set, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics."""
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg: float = 0.0  # Normalized Discounted Cumulative Gain
    latency_ms: float = 0.0
    num_results: int = 0


class MetricsTracker:
    """Track retrieval performance over time.
    
    FAANG-grade monitoring for production systems.
    """
    
    def __init__(self, log_interval: int = 100):
        """Initialize metrics tracker.
        
        Args:
            log_interval: Log stats every N queries
        """
        self.queries_processed = 0
        self.total_latency = 0.0
        self.total_recall = 0.0
        self.total_precision = 0.0
        self.total_mrr = 0.0
        self.failures = 0
        self.log_interval = log_interval
        self.start_time = time.time()
    
    def compute_recall_at_k(
        self, retrieved: List[str], relevant: Set[str], k: int
    ) -> float:
        """Recall@K = |retrieved ∩ relevant| / |relevant|
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs (ground truth)
            k: Number of top results to consider
            
        Returns:
            Recall@K score [0, 1]
        """
        if not relevant:
            return 0.0
        
        retrieved_set = set(retrieved[:k])
        hits = len(retrieved_set & relevant)
        return hits / len(relevant)
    
    def compute_precision_at_k(
        self, retrieved: List[str], relevant: Set[str], k: int
    ) -> float:
        """Precision@K = |retrieved ∩ relevant| / k
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs (ground truth)
            k: Number of top results to consider
            
        Returns:
            Precision@K score [0, 1]
        """
        if k == 0:
            return 0.0
        
        retrieved_set = set(retrieved[:k])
        hits = len(retrieved_set & relevant)
        return hits / min(k, len(retrieved))
    
    def compute_mrr(
        self, retrieved: List[str], relevant: Set[str]
    ) -> float:
        """Mean Reciprocal Rank: 1 / rank_of_first_relevant.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            
        Returns:
            MRR score [0, 1]
        """
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return 1.0 / rank
        return 0.0
    
    def compute_ndcg(
        self, retrieved: List[str], relevant: Dict[str, float], k: int
    ) -> float:
        """Normalized Discounted Cumulative Gain.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Dict mapping doc_id -> relevance score
            k: Number of top results to consider
            
        Returns:
            NDCG@K score [0, 1]
        """
        if not relevant:
            return 0.0
        
        # DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], start=1):
            rel = relevant.get(doc_id, 0.0)
            dcg += (2 ** rel - 1) / np.log2(i + 1)
        
        # Ideal DCG (IDCG)
        ideal_rels = sorted(relevant.values(), reverse=True)[:k]
        idcg = sum(
            (2 ** rel - 1) / np.log2(i + 1)
            for i, rel in enumerate(ideal_rels, start=1)
        )
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def track_query(
        self,
        latency_ms: float,
        num_results: int,
        recall: Optional[float] = None,
        precision: Optional[float] = None,
        mrr: Optional[float] = None
    ) -> None:
        """Track metrics for a single query.
        
        Args:
            latency_ms: Query latency in milliseconds
            num_results: Number of results retrieved
            recall: Recall@K score (if ground truth available)
            precision: Precision@K score (if ground truth available)
            mrr: MRR score (if ground truth available)
        """
        self.queries_processed += 1
        self.total_latency += latency_ms
        
        if recall is not None:
            self.total_recall += recall
        if precision is not None:
            self.total_precision += precision
        if mrr is not None:
            self.total_mrr += mrr
        
        # Log stats periodically
        if self.queries_processed % self.log_interval == 0:
            self.log_stats()
    
    def track_failure(self) -> None:
        """Track a failed retrieval."""
        self.failures += 1
        self.queries_processed += 1
    
    def log_stats(self) -> None:
        """Log accumulated statistics."""
        if self.queries_processed == 0:
            return
        
        n = self.queries_processed
        avg_latency = self.total_latency / n
        uptime = time.time() - self.start_time
        qps = n / uptime if uptime > 0 else 0
        
        logger.info(
            f"📊 Retrieval Stats (last {self.log_interval} queries): "
            f"Avg Latency={avg_latency:.1f}ms, "
            f"QPS={qps:.2f}, "
            f"Failures={self.failures}"
        )
        
        if self.total_recall > 0:
            avg_recall = self.total_recall / n
            logger.info(f"  Avg Recall@5={avg_recall:.3f}")
        
        if self.total_precision > 0:
            avg_precision = self.total_precision / n
            logger.info(f"  Avg Precision@5={avg_precision:.3f}")
        
        # Reset counters
        self.total_latency = 0.0
        self.total_recall = 0.0
        self.total_precision = 0.0
        self.total_mrr = 0.0
        self.queries_processed = 0
        self.failures = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics.
        
        Returns:
            Dict with current metrics
        """
        n = self.queries_processed
        if n == 0:
            return {
                "queries_processed": 0,
                "failures": self.failures,
                "avg_latency_ms": 0.0
            }
        
        return {
            "queries_processed": n,
            "failures": self.failures,
            "avg_latency_ms": self.total_latency / n,
            "avg_recall": self.total_recall / n if self.total_recall > 0 else None,
            "avg_precision": self.total_precision / n if self.total_precision > 0 else None,
            "avg_mrr": self.total_mrr / n if self.total_mrr > 0 else None
        }
