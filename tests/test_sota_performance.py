"""
COMPREHENSIVE RAG SYSTEM TEST
==============================
Tests the full pipeline with challenging queries to verify SOTA performance.
"""

import sys
from pathlib import Path
import yaml
import json
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval.dual_retriever import DualRetriever
from fusion.context_fusion import ContextFusion
from generation.grounded_llm import GroundedLLM


# Comprehensive test suite covering different query types
TEST_SUITE = [
    {
        "category": "Factual Recall",
        "queries": [
            "What is ELMo and when was it released?",
            "How many parameters does GPT-3 have?",
            "When was the Transformer model introduced?",
        ]
    },
    {
        "category": "Conceptual Understanding",
        "queries": [
            "Explain the attention mechanism in transformers",
            "What are the key innovations in transformer-based models?",
            "How do language models generate text?",
        ]
    },
    {
        "category": "Application & Use Cases",
        "queries": [
            "What are practical use cases for large language models?",
            "How can LLMs be used in healthcare?",
            "What are the business applications of ChatGPT?",
        ]
    },
    {
        "category": "Comparative Analysis",
        "queries": [
            "Compare GPT-2 and GPT-3",
            "What are the advantages of BERT over ELMo?",
            "How do RNNs differ from Transformers?",
        ]
    },
]


def run_comprehensive_test():
    """Run full test suite and generate performance report."""
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RAG SYSTEM TEST - SOTA PERFORMANCE VERIFICATION")
    print("=" * 80)
    
    # Initialize
    print("\n[INITIALIZATION]")
    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)
    
    retriever = DualRetriever(config)
    fusion = ContextFusion(config)
    generator = GroundedLLM(config)
    print("✓ All components initialized\n")
    
    # Results storage
    all_results = []
    category_stats = {}
    
    # Run tests by category
    for category_info in TEST_SUITE:
        category = category_info["category"]
        queries = category_info["queries"]
        
        print(f"\n{'=' * 80}")
        print(f"CATEGORY: {category}")
        print(f"{'=' * 80}")
        
        category_results = []
        
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] {query}")
            print("-" * 80)
            
            start_time = time.time()
            
            try:
                # Retrieval
                results = retriever.retrieve(query=query, top_k=15)
                retrieval_time = time.time() - start_time
                
                # Fusion
                context = fusion.fuse(results)
                fusion_time = time.time() - start_time - retrieval_time
                
                # Generation
                gen_start = time.time()
                response = generator.generate(query=query, context=context, sources=results[:5])
                generation_time = time.time() - gen_start
                
                total_time = time.time() - start_time
                
                answer = response.get("answer", "")
                confidence = response.get("confidence", 0.0)
                
                # Quality metrics
                answer_length = len(answer)
                has_citations = "[Source" in answer or "[CHUNK" in answer
                word_count = len(answer.split())
                
                result = {
                    "category": category,
                    "query": query,
                    "answer": answer,
                    "confidence": confidence,
                    "metrics": {
                        "answer_length": answer_length,
                        "word_count": word_count,
                        "has_citations": has_citations,
                        "retrieval_count": len(results),
                        "context_length": len(context),
                    },
                    "timings": {
                        "retrieval": retrieval_time,
                        "fusion": fusion_time,
                        "generation": generation_time,
                        "total": total_time,
                    },
                    "success": True
                }
                
                # Display summary
                print(f"✓ Success")
                print(f"  Answer: {answer[:150]}...")
                print(f"  Length: {word_count} words | Confidence: {confidence:.2f} | Time: {total_time:.2f}s")
                if has_citations:
                    print(f"  ✓ Contains grounded citations")
                
                category_results.append(result)
                all_results.append(result)
                
            except Exception as e:
                print(f"✗ ERROR: {e}")
                result = {
                    "category": category,
                    "query": query,
                    "error": str(e),
                    "success": False
                }
                category_results.append(result)
                all_results.append(result)
        
        # Category stats
        successful = [r for r in category_results if r.get("success")]
        if successful:
            category_stats[category] = {
                "total": len(category_results),
                "successful": len(successful),
                "avg_confidence": sum(r["confidence"] for r in successful) / len(successful),
                "avg_word_count": sum(r["metrics"]["word_count"] for r in successful) / len(successful),
                "avg_time": sum(r["timings"]["total"] for r in successful) / len(successful),
                "citation_rate": sum(1 for r in successful if r["metrics"]["has_citations"]) / len(successful),
            }
    
    # Final Report
    print("\n" + "=" * 80)
    print("PERFORMANCE REPORT")
    print("=" * 80)
    
    successful_results = [r for r in all_results if r.get("success")]
    total_queries = len(all_results)
    
    print(f"\n📊 OVERALL METRICS:")
    print(f"  Total Queries: {total_queries}")
    print(f"  Successful: {len(successful_results)}")
    print(f"  Success Rate: {len(successful_results)/total_queries*100:.1f}%")
    
    if successful_results:
        avg_conf = sum(r["confidence"] for r in successful_results) / len(successful_results)
        avg_words = sum(r["metrics"]["word_count"] for r in successful_results) / len(successful_results)
        avg_time = sum(r["timings"]["total"] for r in successful_results) / len(successful_results)
        citation_rate = sum(1 for r in successful_results if r["metrics"]["has_citations"]) / len(successful_results)
        
        print(f"\n  Average Confidence: {avg_conf:.3f}")
        print(f"  Average Answer Length: {avg_words:.0f} words")
        print(f"  Citation Rate: {citation_rate*100:.1f}%")
        print(f"  Average Latency: {avg_time:.2f}s")
    
    print(f"\n📈 BY CATEGORY:")
    for category, stats in category_stats.items():
        print(f"\n  {category}:")
        print(f"    Success: {stats['successful']}/{stats['total']}")
        print(f"    Avg Confidence: {stats['avg_confidence']:.3f}")
        print(f"    Avg Words: {stats['avg_word_count']:.0f}")
        print(f"    Citations: {stats['citation_rate']*100:.1f}%")
        print(f"    Avg Time: {stats['avg_time']:.2f}s")
    
    # SOTA Assessment
    print(f"\n{'=' * 80}")
    print("SOTA PERFORMANCE ASSESSMENT")
    print(f"{'=' * 80}")
    
    sota_checks = []
    
    if successful_results:
        if avg_conf >= 0.90:
            sota_checks.append("✅ Excellent confidence scores (≥0.90)")
        elif avg_conf >= 0.80:
            sota_checks.append("✓ Good confidence scores (≥0.80)")
        else:
            sota_checks.append("⚠️  Confidence needs improvement (<0.80)")
        
        if citation_rate >= 0.80:
            sota_checks.append("✅ Excellent grounding/citation rate (≥80%)")
        elif citation_rate >= 0.60:
            sota_checks.append("✓ Good grounding/citation rate (≥60%)")
        else:
            sota_checks.append("⚠️  Citations need improvement (<60%)")
        
        if avg_time <= 5.0:
            sota_checks.append("✅ Excellent latency (≤5s)")
        elif avg_time <= 10.0:
            sota_checks.append("✓ Good latency (≤10s)")
        else:
            sota_checks.append("⚠️  Latency needs optimization (>10s)")
        
        if avg_words >= 100:
            sota_checks.append("✅ Comprehensive answers (≥100 words)")
        elif avg_words >= 50:
            sota_checks.append("✓ Adequate answer length (≥50 words)")
        else:
            sota_checks.append("⚠️  Answers too brief (<50 words)")
    
    for check in sota_checks:
        print(f"  {check}")
    
    # Save results
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "comprehensive_test_results.json", "w") as f:
        json.dump({
            "results": all_results,
            "category_stats": category_stats,
            "overall": {
                "total": total_queries,
                "successful": len(successful_results),
                "avg_confidence": avg_conf if successful_results else 0,
                "avg_words": avg_words if successful_results else 0,
                "avg_time": avg_time if successful_results else 0,
                "citation_rate": citation_rate if successful_results else 0,
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: artifacts/comprehensive_test_results.json")
    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    run_comprehensive_test()
