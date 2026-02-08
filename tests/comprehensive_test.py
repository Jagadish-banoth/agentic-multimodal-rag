"""
Direct test of orchestrator with comprehensive queries and analysis.
"""

import sys
import yaml
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orchestrator.execution_engine import ExecutionEngine


TEST_QUERIES = [
    {
        "query": "What is ELMo and when was it released?",
        "type": "factual",
        "expected_keywords": ["ELMo", "2018", "Allen Institute", "94 million"]
    },
    {
        "query": "How many parameters does GPT-3 have?",
        "type": "factual",
        "expected_keywords": ["175", "billion", "parameters"]
    },
    {
        "query": "Explain the attention mechanism in transformers",
        "type": "conceptual",
        "expected_keywords": ["attention", "transformer", "query", "key", "value"]
    },
    {
        "query": "What are practical use cases for large language models?",
        "type": "application",
        "expected_keywords": ["ChatGPT", "text", "generation", "classification"]
    },
    {
        "query": "Compare GPT-2 and GPT-3",
        "type": "comparative",
        "expected_keywords": ["GPT-2", "GPT-3", "parameters", "capabilities"]
    },
]


def evaluate_answer(answer: str, expected_keywords: list) -> float:
    """Simple keyword-based evaluation."""
    answer_lower = answer.lower()
    matches = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return matches / len(expected_keywords) if expected_keywords else 0.0


def main():
    print("\n" + "=" * 70)
    print("END-TO-END RAG SYSTEM TEST")
    print("=" * 70)
    
    # Load config
    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)
    
    # Initialize engine
    print("\nInitializing ExecutionEngine...")
    engine = ExecutionEngine(config)
    print("✓ Engine ready\n")
    
    results = []
    
    for i, test in enumerate(TEST_QUERIES, 1):
        query = test["query"]
        qtype = test["type"]
        expected = test.get("expected_keywords", [])
        
        print(f"\n[{i}/{len(TEST_QUERIES)}] {qtype.upper()}")
        print(f"Query: {query}")
        print("-" * 70)
        
        start = time.time()
        
        try:
            # Run query
            output = engine.run(query)
            elapsed = time.time() - start
            
            # Extract answer
            answer = output.get("final_answer") or output.get("answer", "No answer")
            confidence = output.get("confidence", 0.0)
            faithful = output.get("faithful", False)
            
            # Evaluate
            keyword_score = evaluate_answer(answer, expected)
            
            result = {
                "query": query,
                "type": qtype,
                "answer": answer,
                "confidence": confidence,
                "faithful": faithful,
                "keyword_score": keyword_score,
                "time": elapsed,
                "timings": output.get("timings", {}),
                "success": True
            }
            
            # Print results
            print(f"✓ Success ({elapsed:.2f}s)")
            print(f"\nAnswer ({len(answer)} chars):")
            print(answer[:300] + ("..." if len(answer) > 300 else ""))
            print(f"\nMetrics:")
            print(f"  - Confidence: {confidence:.2f}")
            print(f"  - Faithful: {faithful}")
            print(f"  - Keyword Match: {keyword_score:.2f}")
            
            # Print timings if available
            if output.get("timings"):
                print(f"  - Timings: {', '.join(f'{k}={v:.2f}s' for k, v in output['timings'].items())}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            result = {
                "query": query,
                "type": qtype,
                "error": str(e),
                "success": False
            }
        
        results.append(result)
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    successful = [r for r in results if r.get("success")]
    
    if successful:
        avg_conf = sum(r["confidence"] for r in successful) / len(successful)
        avg_kw = sum(r["keyword_score"] for r in successful) / len(successful)
        avg_time = sum(r["time"] for r in successful) / len(successful)
        faithful_count = sum(1 for r in successful if r.get("faithful"))
        
        print(f"\nSuccess Rate: {len(successful)}/{len(results)}")
        print(f"Avg Confidence: {avg_conf:.3f}")
        print(f"Avg Keyword Match: {avg_kw:.3f}")
        print(f"Avg Time: {avg_time:.2f}s")
        print(f"Faithful Answers: {faithful_count}/{len(successful)}")
        
        # Identify issues
        issues = []
        for r in successful:
            if r["confidence"] < 0.7:
                issues.append(f"Low confidence ({r['confidence']:.2f}): {r['query'][:50]}...")
            if r["keyword_score"] < 0.5:
                issues.append(f"Low keyword match ({r['keyword_score']:.2f}): {r['query'][:50]}...")
            if not r.get("faithful"):
                issues.append(f"Not faithful: {r['query'][:50]}...")
        
        if issues:
            print(f"\n⚠️  Issues Found ({len(issues)}):")
            for issue in issues[:10]:
                print(f"  - {issue}")
        
        # SOTA recommendations
        print("\n" + "=" * 70)
        print("SOTA IMPROVEMENT RECOMMENDATIONS")
        print("=" * 70)
        
        recommendations = []
        
        if avg_conf < 0.8:
            recommendations.append("🔧 Improve confidence scores:")
            recommendations.append("   - Enhance verifier with better NLI model (DeBERTa-v3-large)")
            recommendations.append("   - Add answer calibration with temperature scaling")
        
        if avg_kw < 0.7:
            recommendations.append("🔧 Improve answer quality:")
            recommendations.append("   - Use larger LLM (llama3:70b or mixtral:8x7b)")
            recommendations.append("   - Add few-shot examples in prompts")
            recommendations.append("   - Implement chain-of-thought reasoning")
        
        if avg_time > 10:
            recommendations.append("🔧 Optimize latency:")
            recommendations.append("   - Enable GPU inference for reranker")
            recommendations.append("   - Implement result caching")
            recommendations.append("   - Batch processing where possible")
        
        if faithful_count < len(successful):
            recommendations.append("🔧 Improve faithfulness:")
            recommendations.append("   - Stronger citation requirements in prompts")
            recommendations.append("   - Add claim verification step")
            recommendations.append("   - Use retrieval-augmented generation with explicit grounding")
        
        for rec in recommendations:
            print(rec)
    
    # Save results
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "comprehensive_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: artifacts/comprehensive_test_results.json")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
