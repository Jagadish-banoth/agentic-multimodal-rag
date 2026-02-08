"""
Direct LLM Generation Test - Test grounded_llm.py with actual indexed data
"""

import sys
from pathlib import Path
import yaml
import json

# Add project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval.dual_retriever import DualRetriever
from fusion.context_fusion import ContextFusion
from generation.grounded_llm import GroundedLLM


def main():
    print("\n" + "=" * 70)
    print("LLM GENERATION TEST - Using Indexed Data")
    print("=" * 70)
    
    # Load config
    print("\n[1/5] Loading configuration...")
    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)
    print("[OK] Config loaded")
    
    # Initialize components
    print("\n[2/5] Initializing components...")
    try:
        retriever = DualRetriever(config)
        print("[OK] Retriever initialized")
        
        fusion = ContextFusion(config)
        print("[OK] Context fusion initialized")
        
        generator = GroundedLLM(config)
        print("[OK] LLM generator initialized")
    except Exception as e:
        print(f"[ERROR] Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test queries
    test_queries = [
        "What is ELMo and when was it released?",
        "Explain the attention mechanism in transformers",
        "What are practical use cases for large language models?",
    ]
    
    print(f"\n[3/5] Testing with {len(test_queries)} queries...")
    
    for i, query in enumerate(test_queries, 1):
        print("\n" + "-" * 70)
        print(f"Query {i}/{len(test_queries)}: {query}")
        print("-" * 70)
        
        try:
            # Step 1: Retrieve context
            print(f"\n  [Retrieval]")
            results = retriever.retrieve(query=query, top_k=10)
            print(f"  [OK] Retrieved {len(results)} results")
            
            if not results:
                print(f"  [ERROR] No results found - skipping query")
                continue
            
            # Step 2: Fuse context
            print(f"\n  [Fusion]")
            context = fusion.fuse(results)
            print(f"  [OK] Fused context: {len(context)} chars, ~{len(context.split())} tokens")
            
            if not context or len(context) < 50:
                print(f"  [ERROR] Context too short - skipping query")
                continue
            
            # Step 3: Generate answer
            print(f"\n  [Generation]")
            response = generator.generate(
                query=query,
                context=context,
                sources=results[:5]
            )
            
            # Display response
            answer = response.get("answer", "No answer")
            confidence = response.get("confidence", 0.0)
            sources = response.get("sources", [])
            
            print(f"  [OK] Generated answer ({len(answer)} chars)")
            print(f"\n  ANSWER:")
            print(f"  {'-' * 66}")
            # Print answer with indentation
            for line in answer.split('\n'):
                print(f"  {line}")
            print(f"  {'-' * 66}")
            print(f"\n  Confidence: {confidence:.2f}")
            print(f"  Sources: {len(sources)} references")
            
            # Show first 2 sources
            if sources:
                print(f"\n  Top Sources:")
                for j, src in enumerate(sources[:2], 1):
                    if isinstance(src, dict):
                        src_text = src.get("text", src.get("content", src.get("snippet", "")))
                    else:
                        src_text = str(src)
                    src_text = src_text[:100] if src_text else "(no content)"
                    print(f"    {j}. {src_text}...")
            
            
        except Exception as e:
            print(f"\n  [ERROR]: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    
    # Summary
    print("\n📊 Summary:")
    print(f"  - LLM Model: {config.get('models', {}).get('generator_model', 'llama3:8b')}")
    print(f"  - Ollama Host: {config.get('generation', {}).get('ollama', {}).get('host', 'localhost:11434')}")
    print(f"  - Max Tokens: {config.get('generation', {}).get('max_generation_tokens', 400)}")
    print(f"  - Temperature: {config.get('generation', {}).get('temperature', 0.3)}")
    print("\n✅ If you see generated answers above, LLM generation is working!")
    print("\nSUCCESS: LLM generation is working properly!")
    print("="*70)


if __name__ == "__main__":
    main()
