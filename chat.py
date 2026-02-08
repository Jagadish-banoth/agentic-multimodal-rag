import yaml
from orchestrator.execution_engine import ExecutionEngine

def main():
    # Load config
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    engine = ExecutionEngine(config)

    print("\n=== Document Chat (Agentic RAG) ===")
    print("Ask questions about your documents.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You> ").strip()
        if query.lower() in ("exit", "quit"):
            print("Exiting chat.")
            break
        
        # Skip empty queries
        if not query:
            print("\n(Please enter a question)\n")
            continue

        output = engine.run(query)

        print("\nAssistant>\n")
        # Handle both "final_answer" and "answer" keys
        answer = output.get("final_answer") or output.get("answer", "I couldn't generate a response.")
        print(answer)
        
        # Display timing information
        if "timings" in output:
            print("\n⏱️  Timing breakdown:")
            for step, duration in output["timings"].items():
                print(f"    {step:12}: {duration:.2f}s")
            print(f"    {'─' * 20}")
            print(f"    {'Total':12}: {output.get('total_time', 0):.2f}s")
        
        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()
