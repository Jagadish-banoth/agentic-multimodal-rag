"""Test retrieval accuracy with production-grade improvements."""
import sys
from pathlib import Path

# Add project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from orchestrator.execution_engine import ExecutionEngine


def test_retrieval_quality():
    """Test retrieval with realistic queries."""
    print("=" * 70)
    print("RETRIEVAL QUALITY TEST")
    print("=" * 70)
    
    # Load config
    config_path = ROOT / "config" / "settings.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Initialize engine
    print("\n[*] Initializing execution engine...")
    engine = ExecutionEngine(config, enable_phase1=True)
    
    # Test queries (diverse set)
    test_queries = [
        "What is artificial intelligence?",
        "Explain machine learning algorithms",
        "How does computer vision work?",
        "What are neural networks?",
        "Describe data preprocessing techniques"
    ]
    
    print(f"\n[*] Testing with {len(test_queries)} queries...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}/{len(test_queries)}: {query}")
        print('='*70)
        
        try:
            response = engine.run(query)
            
            # Display results
            print(f"\n[OK] Answer Generated:")
            print(f"  {response.get('answer', 'No answer')[:200]}...")
            
            print(f"\n[METRICS]")
            print(f"  - Confidence: {response.get('confidence', 0):.2%}")
            print(f"  - Verified: {response.get('verified', False)}")
            print(f"  - From Cache: {response.get('from_cache', False)}")
            
            # Timing breakdown
            timings = response.get('timings', {})
            if timings:
                print(f"\n[TIMINGS]")
                for key, value in timings.items():
                    print(f"  - {key}: {value:.3f}s")
                print(f"  - Total: {response.get('total_time', 0):.3f}s")
            
            # Sources
            sources = response.get('sources', [])
            print(f"\n[SOURCES] ({len(sources)} chunks):")
            for j, source in enumerate(sources[:3], 1):
                doc_id = source.get('doc_id', 'unknown')
                chunk_idx = source.get('chunk_idx', 0)
                score = source.get('score', 0)
                print(f"  {j}. {doc_id}[{chunk_idx}] (score: {score:.3f})")
            
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    # Display overall stats
    print(f"\n\n{'='*70}")
    print("OVERALL STATISTICS")
    print('='*70)
    stats = engine.get_cache_stats()
    
    if 'cache' in stats:
        cache_stats = stats['cache']
        print(f"\n[CACHE]")
        print(f"  - Hits: {cache_stats.get('hits', 0)}")
        print(f"  - Misses: {cache_stats.get('misses', 0)}")
        print(f"  - Hit Rate: {cache_stats.get('hit_rate', 0):.1%}")
    
    if 'retrieval' in stats:
        ret_stats = stats['retrieval']
        print(f"\n[RETRIEVAL]")
        print(f"  - Queries Processed: {ret_stats.get('queries_processed', 0)}")
        print(f"  - Failures: {ret_stats.get('failures', 0)}")
        avg_latency = ret_stats.get('avg_latency_ms', 0)
        if avg_latency:
            print(f"  - Avg Latency: {avg_latency:.1f}ms")
    
    # Cleanup
    engine.shutdown()
    
    print(f"\n{'='*70}")
    print("[OK] Test Complete!")
    print('='*70)


if __name__ == "__main__":
    test_retrieval_quality()
