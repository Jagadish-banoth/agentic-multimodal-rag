#!/usr/bin/env python3
"""
End-to-end test script for agentic multimodal RAG system.
Tests retrieval, generation, and verification with comprehensive analysis.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import yaml
from planner.local_agentic_planner import LocalAgenticPlanner
from retrieval.dual_retriever import DualRetriever
from retrieval.sparse_retriever import SparseRetriever
from retrieval.reranker import Reranker
from fusion.context_fusion import ContextFusion
from generation.grounded_llm import GroundedLLM
from verification.verifier import Verifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Config loader
def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class EndToEndTester:
    """Comprehensive end-to-end testing with analysis."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize all components."""
        self.config = load_config(config_path)
        self.results = []
        
        logger.info("=" * 70)
        logger.info("INITIALIZING END-TO-END TEST SYSTEM")
        logger.info("=" * 70)
        
        # Initialize components
        self.planner = LocalAgenticPlanner(self.config)
        self.dense_retriever = DualRetriever(self.config)
        self.sparse_retriever = SparseRetriever(self.config)
        self.reranker = Reranker(self.config)
        self.fusion = ContextFusion(self.config)
        self.generator = GroundedLLM(self.config)
        self.verifier = Verifier(self.config)
        
        logger.info("✓ All components initialized")
    
    def run_query(self, query: str, query_type: str = "unknown") -> Dict[str, Any]:
        """Run a single query through the full pipeline."""
        logger.info("\n" + "=" * 70)
        logger.info(f"QUERY: {query}")
        logger.info(f"TYPE: {query_type}")
        logger.info("=" * 70)
        
        start_time = time.time()
        result = {
            "query": query,
            "type": query_type,
            "success": False,
            "timings": {},
            "metrics": {}
        }
        
        try:
            # Step 1: Planning
            t0 = time.time()
            plan = self.planner.plan_query(query)
            result["timings"]["planning"] = time.time() - t0
            result["plan"] = {
                "intent": plan.intent,
                "modalities": plan.modalities,
                "retrieval_depth": plan.retrieval_depth
            }
            logger.info(f"✓ Plan: {plan.intent} | {plan.modalities}")
            
            # Step 2: Dense Retrieval
            t0 = time.time()
            dense_results = self.dense_retriever.retrieve(
                query=query,
                k=self.config.get("retrieval", {}).get("dense_k", 20)
            )
            result["timings"]["dense_retrieval"] = time.time() - t0
            result["metrics"]["dense_count"] = len(dense_results)
            logger.info(f"✓ Dense retrieval: {len(dense_results)} results")
            
            # Step 3: Sparse Retrieval (BM25)
            t0 = time.time()
            sparse_results = self.sparse_retriever.retrieve(
                query=query,
                k=self.config.get("retrieval", {}).get("sparse_k", 20)
            )
            result["timings"]["sparse_retrieval"] = time.time() - t0
            result["metrics"]["sparse_count"] = len(sparse_results)
            logger.info(f"✓ Sparse retrieval: {len(sparse_results)} results")
            
            # Step 4: Merge and Rerank
            t0 = time.time()
            all_results = dense_results + sparse_results
            
            # Deduplicate by chunk_id
            seen = set()
            unique_results = []
            for r in all_results:
                cid = r.get("chunk_id")
                if cid and cid not in seen:
                    seen.add(cid)
                    unique_results.append(r)
            
            reranked = self.reranker.rerank(query, unique_results)
            result["timings"]["reranking"] = time.time() - t0
            result["metrics"]["reranked_count"] = len(reranked)
            logger.info(f"✓ Reranked: {len(unique_results)} → {len(reranked)}")
            
            # Step 5: Context Fusion
            t0 = time.time()
            fused_context = self.fusion.fuse(reranked)
            result["timings"]["fusion"] = time.time() - t0
            result["metrics"]["context_tokens"] = len(fused_context.split())
            logger.info(f"✓ Fused context: {len(fused_context.split())} tokens")
            
            # Step 6: Grounded Generation
            t0 = time.time()
            answer = self.generator.generate(
                query=query,
                context=fused_context,
                top_chunks=reranked[:5]
            )
            result["timings"]["generation"] = time.time() - t0
            result["answer"] = answer
            logger.info(f"✓ Generated answer: {len(answer)} chars")
            
            # Step 7: Verification
            t0 = time.time()
            verification = self.verifier.verify(
                query=query,
                answer=answer,
                context=fused_context
            )
            result["timings"]["verification"] = time.time() - t0
            result["verification"] = {
                "faithful": verification.get("faithful", False),
                "confidence": verification.get("confidence", 0.0),
                "verdict": verification.get("verdict", "unknown")
            }
            logger.info(
                f"✓ Verification: {verification.get('verdict')} "
                f"(confidence: {verification.get('confidence', 0):.2f})"
            )
            
            result["success"] = True
            result["timings"]["total"] = time.time() - start_time
            
        except Exception as e:
            logger.error(f"✗ Query failed: {e}", exc_info=True)
            result["error"] = str(e)
            result["timings"]["total"] = time.time() - start_time
        
        return result
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze test results and provide insights."""
        logger.info("\n" + "=" * 70)
        logger.info("ANALYSIS REPORT")
        logger.info("=" * 70)
        
        analysis = {
            "total_queries": len(results),
            "successful": sum(1 for r in results if r.get("success")),
            "failed": sum(1 for r in results if not r.get("success")),
            "avg_timings": {},
            "verification_stats": {},
            "issues": []
        }
        
        # Calculate averages
        successful_results = [r for r in results if r.get("success")]
        
        if successful_results:
            for key in ["planning", "dense_retrieval", "sparse_retrieval", 
                       "reranking", "fusion", "generation", "verification", "total"]:
                times = [r["timings"].get(key, 0) for r in successful_results]
                analysis["avg_timings"][key] = sum(times) / len(times) if times else 0
            
            # Verification stats
            faithful_count = sum(
                1 for r in successful_results 
                if r.get("verification", {}).get("faithful", False)
            )
            confidences = [
                r.get("verification", {}).get("confidence", 0) 
                for r in successful_results
            ]
            
            analysis["verification_stats"] = {
                "faithful_count": faithful_count,
                "faithful_rate": faithful_count / len(successful_results),
                "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
                "min_confidence": min(confidences) if confidences else 0,
                "max_confidence": max(confidences) if confidences else 0
            }
        
        # Identify issues
        for r in results:
            if not r.get("success"):
                analysis["issues"].append({
                    "query": r.get("query"),
                    "error": r.get("error")
                })
            elif not r.get("verification", {}).get("faithful", False):
                analysis["issues"].append({
                    "query": r.get("query"),
                    "issue": "Low faithfulness",
                    "confidence": r.get("verification", {}).get("confidence", 0)
                })
        
        # Print analysis
        logger.info(f"\n📊 Overall Success Rate: {analysis['successful']}/{analysis['total_queries']}")
        logger.info(f"   Faithful Responses: {analysis['verification_stats'].get('faithful_count', 0)}")
        logger.info(f"   Avg Confidence: {analysis['verification_stats'].get('avg_confidence', 0):.3f}")
        
        logger.info(f"\n⏱️  Average Timings:")
        for key, val in analysis["avg_timings"].items():
            logger.info(f"   {key:20s}: {val:.3f}s")
        
        if analysis["issues"]:
            logger.info(f"\n⚠️  Issues Found: {len(analysis['issues'])}")
            for i, issue in enumerate(analysis["issues"][:5], 1):
                logger.info(f"   {i}. {issue.get('query', 'unknown')[:50]}...")
                logger.info(f"      → {issue.get('error') or issue.get('issue')}")
        
        return analysis
    
    def run_test_suite(self, queries: List[str], query_types: List[str] = None):
        """Run full test suite."""
        if query_types is None:
            query_types = ["unknown"] * len(queries)
        
        logger.info("\n" + "=" * 70)
        logger.info(f"RUNNING TEST SUITE: {len(queries)} queries")
        logger.info("=" * 70)
        
        for i, (query, qtype) in enumerate(zip(queries, query_types), 1):
            logger.info(f"\n[{i}/{len(queries)}] Processing: {query[:60]}...")
            result = self.run_query(query, qtype)
            self.results.append(result)
            
            # Print brief result
            if result.get("success"):
                conf = result.get("verification", {}).get("confidence", 0)
                logger.info(f"✓ Success | Confidence: {conf:.2f} | Time: {result['timings']['total']:.2f}s")
                logger.info(f"  Answer: {result.get('answer', '')[:100]}...")
            else:
                logger.info(f"✗ Failed: {result.get('error', 'unknown')}")
        
        # Final analysis
        analysis = self.analyze_results(self.results)
        
        # Save results
        output_file = project_root / "artifacts" / "test_results.json"
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, "w") as f:
            json.dump({
                "results": self.results,
                "analysis": analysis
            }, f, indent=2)
        
        logger.info(f"\n💾 Results saved to: {output_file}")
        
        return analysis


def main():
    """Main test execution."""
    # Define comprehensive test queries
    test_queries = [
        # Factual
        ("What is ELMo and when was it released?", "factual"),
        ("How many parameters does GPT-3 have?", "factual"),
        
        # Conceptual
        ("Explain the attention mechanism in transformers", "conceptual"),
        ("What are the key innovations in transformer-based models?", "conceptual"),
        
        # Application
        ("What are practical use cases for large language models?", "application"),
        ("How can LLMs be used for text classification?", "application"),
        
        # Comparative
        ("Compare GPT-2 and GPT-3", "comparative"),
        
        # Multi-hop
        ("How has the evolution from ELMo to GPT-3 changed AI?", "multi-hop"),
    ]
    
    queries = [q for q, _ in test_queries]
    types = [t for _, t in test_queries]
    
    # Run tests
    tester = EndToEndTester()
    analysis = tester.run_test_suite(queries, types)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total Queries: {analysis['total_queries']}")
    print(f"Successful: {analysis['successful']}")
    print(f"Failed: {analysis['failed']}")
    print(f"Avg Confidence: {analysis['verification_stats'].get('avg_confidence', 0):.3f}")
    print(f"Avg Total Time: {analysis['avg_timings'].get('total', 0):.3f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
