#!/usr/bin/env python3
"""
Simplified end-to-end test for the agentic multimodal RAG system.
"""

import os
import sys
import json
import time
from pathlib import Path

# Test queries about the document
TEST_QUERIES = [
    "What is ELMo and when was it released?",
    "How many parameters does GPT-3 have?",
    "Explain the attention mechanism in transformers",
    "What are practical use cases for large language models?",
    "Compare GPT-2 and GPT-3",
]

def test_with_chat():
    """Test using the chat.py interface."""
    import subprocess
    
    results = []
    
    print("=" * 70)
    print("RUNNING END-TO-END TESTS")
    print("=" * 70)
    
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/{len(TEST_QUERIES)}] Query: {query}")
        print("-" * 70)
        
        start_time = time.time()
        
        try:
            # Run chat.py with the query
            result = subprocess.run(
                [sys.executable, "chat.py"],
                input=f"{query}\nexit\n",
                capture_output=True,
                text=True,
                timeout=60
            )
            
            elapsed = time.time() - start_time
            
            # Extract answer from output
            output_lines = result.stdout.split('\n')
            answer_started = False
            answer_lines = []
            
            for line in output_lines:
                if "Answer:" in line:
                    answer_started = True
                    continue
                if answer_started:
                    if line.strip().startswith("Sources:") or line.strip().startswith("Confidence:"):
                        break
                    if line.strip():
                        answer_lines.append(line.strip())
            
            answer = " ".join(answer_lines) if answer_lines else "No answer generated"
            
            # Extract confidence if present
            confidence = 0.0
            for line in output_lines:
                if "Confidence:" in line:
                    try:
                        confidence = float(line.split("Confidence:")[1].strip())
                    except:
                        pass
            
            results.append({
                "query": query,
                "answer": answer,
                "confidence": confidence,
                "time": elapsed,
                "success": True
            })
            
            print(f"✓ Success ({elapsed:.2f}s)")
            print(f"  Answer: {answer[:150]}...")
            print(f"  Confidence: {confidence:.2f}")
            
        except subprocess.TimeoutExpired:
            print(f"✗ Timeout")
            results.append({
                "query": query,
                "error": "Timeout",
                "success": False
            })
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append({
                "query": query,
                "error": str(e),
                "success": False
            })
    
    # Save results
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "e2e_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    successful = sum(1 for r in results if r.get("success"))
    print(f"Total: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    
    if successful > 0:
        avg_time = sum(r.get("time", 0) for r in results if r.get("success")) / successful
        avg_conf = sum(r.get("confidence", 0) for r in results if r.get("success")) / successful
        print(f"Avg Time: {avg_time:.2f}s")
        print(f"Avg Confidence: {avg_conf:.2f}")
    
    print(f"\nResults saved to: artifacts/e2e_test_results.json")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    test_with_chat()
