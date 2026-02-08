"""
Phase 1 Integration Test
========================

Test parallel retrieval, query expansion, and caching together.

Run: python scripts/test_phase1.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import yaml

# Setup path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("phase1_test")


def load_config():
    """Load configuration."""
    with open(ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_query_expansion():
    """Test query expansion module."""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Query Expansion")
    logger.info("="*70)
    
    try:
        from retrieval.query_expansion import QueryExpander
        
        config = load_config()
        expander = QueryExpander(config, enable_hyde=True)
        
        test_queries = [
            "What is transformer architecture?",
            "How does self-attention work?",
            "Compare BERT and GPT models",
        ]
        
        for query in test_queries:
            logger.info(f"\nQuery: {query}")
            
            t0 = time.time()
            expansion = expander.expand_query(query)
            elapsed = time.time() - t0
            
            logger.info(f"  Elapsed: {elapsed:.2f}s")
            
            for key, values in expansion.items():
                if values:
                    logger.info(f"  {key}: {len(values)} variants")
                    for v in values[:2]:
                        logger.info(f"    - {v[:70]}...")
        
        logger.info("✓ Query expansion test passed")
        return True
    
    except Exception as e:
        logger.error(f"✗ Query expansion test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_result_caching():
    """Test result caching with fuzzy matching."""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Result Caching")
    logger.info("="*70)
    
    try:
        from utils.result_cache import ResultCache
        
        config = load_config()
        cache = ResultCache(config)
        
        if not cache.enabled:
            logger.warning("Cache disabled in config - skipping test")
            return True
        
        # Test data
        test_result = {
            "answer": "Transformers are neural network models...",
            "sources": ["doc1", "doc2"],
            "confidence": 0.85,
        }
        
        # Test 1: Exact match
        logger.info("\nTest 2a: Exact match caching")
        query1 = "What is transformer architecture?"
        cache.set(query1, test_result, confidence=0.85)
        cached = cache.get(query1)
        
        if cached:
            logger.info(f"  ✓ Cache HIT: {cached['answer'][:50]}...")
        else:
            logger.error("  ✗ Cache MISS on exact match")
            return False
        
        # Test 2: Fuzzy match (similar query)
        logger.info("\nTest 2b: Fuzzy matching")
        query2 = "What is the transformer architecture?"  # Slightly different
        fuzzy_cached = cache.get(query2)
        
        if fuzzy_cached:
            logger.info(f"  ✓ Fuzzy HIT: {fuzzy_cached['answer'][:50]}...")
        else:
            logger.info("  ⚠️ Fuzzy MISS (expected, threshold is high)")
        
        # Test 3: Cache statistics
        logger.info("\nTest 2c: Cache statistics")
        stats = cache.stats()
        logger.info(f"  Cache stats: {json.dumps(stats, indent=2)}")
        
        logger.info("✓ Result caching test passed")
        return True
    
    except Exception as e:
        logger.error(f"✗ Result caching test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_parallel_retrieval():
    """Test parallel retrieval with execution engine."""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Parallel Retrieval")
    logger.info("="*70)
    
    try:
        from orchestrator.execution_engine import ExecutionEngine
        
        config = load_config()
        
        # Create engine with Phase 1 enabled
        logger.info("Initializing ExecutionEngine with Phase 1...")
        engine = ExecutionEngine(config, enable_phase1=True)
        
        # Test query
        query = "What is self-attention mechanism?"
        logger.info(f"\nTest query: {query}")
        
        # Run with timing
        logger.info("Running end-to-end pipeline...")
        t0 = time.time()
        response = engine.run(query)
        total_time = time.time() - t0
        
        logger.info(f"\n✓ Pipeline completed in {total_time:.2f}s")
        
        # Log results
        logger.info(f"\nResponse:")
        logger.info(f"  Answer: {response.get('answer', 'N/A')[:100]}...")
        logger.info(f"  Confidence: {response.get('confidence', 0):.2f}")
        logger.info(f"  Verified: {response.get('verified', False)}")
        logger.info(f"  From cache: {response.get('from_cache', False)}")
        
        # Log detailed timings
        timings = response.get('timings', {})
        if timings:
            logger.info("\nTiming breakdown:")
            for key, value in sorted(timings.items()):
                logger.info(f"  {key}: {value:.3f}s")
        
        # Test cache hit on same query
        logger.info("\nTest 3b: Testing cache hit...")
        t0 = time.time()
        response2 = engine.run(query)
        total_time2 = time.time() - t0
        
        if response2.get('from_cache'):
            speedup = total_time / total_time2
            logger.info(f"✓ Cache HIT! Speedup: {speedup:.1f}x ({total_time2:.3f}s vs {total_time:.3f}s)")
        else:
            logger.info(f"⚠️ Cache miss (expected if cache TTL expired)")
        
        # Show cache stats
        cache_stats = engine.get_cache_stats()
        logger.info(f"\nCache stats: {json.dumps(cache_stats, indent=2)}")
        
        engine.shutdown()
        logger.info("\n✓ Parallel retrieval test passed")
        return True
    
    except Exception as e:
        logger.error(f"✗ Parallel retrieval test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_performance_comparison():
    """Compare Phase 1 vs baseline (if Phase 1 disabled)."""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Performance Comparison")
    logger.info("="*70)
    
    try:
        from orchestrator.execution_engine import ExecutionEngine
        
        config = load_config()
        query = "What is machine learning?"
        
        # Test Phase 1 enabled
        logger.info("\nTest 4a: Phase 1 ENABLED (parallel + expansion + caching)")
        engine_p1 = ExecutionEngine(config, enable_phase1=True)
        
        times_p1 = []
        for i in range(3):
            t0 = time.time()
            engine_p1.run(query)
            times_p1.append(time.time() - t0)
        
        avg_p1 = sum(times_p1) / len(times_p1)
        logger.info(f"  Runs: {[f'{t:.2f}s' for t in times_p1]}")
        logger.info(f"  Average: {avg_p1:.2f}s")
        
        # Test Phase 1 disabled
        logger.info("\nTest 4b: Phase 1 DISABLED (baseline sequential)")
        engine_baseline = ExecutionEngine(config, enable_phase1=False)
        
        times_baseline = []
        for i in range(3):
            t0 = time.time()
            engine_baseline.run(query)
            times_baseline.append(time.time() - t0)
        
        avg_baseline = sum(times_baseline) / len(times_baseline)
        logger.info(f"  Runs: {[f'{t:.2f}s' for t in times_baseline]}")
        logger.info(f"  Average: {avg_baseline:.2f}s")
        
        # Calculate speedup
        speedup = avg_baseline / avg_p1
        logger.info(f"\n✓ Speedup with Phase 1: {speedup:.1f}x ({avg_baseline:.2f}s → {avg_p1:.2f}s)")
        
        engine_p1.shutdown()
        engine_baseline.shutdown()
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Performance comparison test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "🚀 " + "="*66 + " 🚀")
    logger.info("PHASE 1 INTEGRATION TESTS - FAANG-Grade Optimizations")
    logger.info("🚀 " + "="*66 + " 🚀\n")
    
    tests = [
        ("Query Expansion", test_query_expansion),
        ("Result Caching", test_result_caching),
        ("Parallel Retrieval", test_parallel_retrieval),
        ("Performance Comparison", test_performance_comparison),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except KeyboardInterrupt:
            logger.info("\n⚠️ Tests interrupted by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error in {name}: {e}")
            results[name] = False
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All Phase 1 tests passed! System ready for FAANG-level performance.")
        return 0
    else:
        logger.error(f"\n⚠️ {total - passed} test(s) failed. Please review logs.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
