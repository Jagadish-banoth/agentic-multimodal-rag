"""
chatbot_qwen.py
===============
Simple conversational chatbot using Local Ollama (llama3:8b)
Ask any general questions!
No API limits, no rate limiting, runs locally!
"""

import json
import logging

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaChatbot:
    """Simple chatbot using local Ollama (llama3:8b)."""
    
    def __init__(self):
        if not OLLAMA_AVAILABLE:
            raise ImportError("❌ ollama package not found! Install with: pip install ollama")
        
        self.model = "llama3:8b"
        self.conversation_history = []
        
        print("=" * 80)
        print("🤖 LOCAL OLLAMA CHATBOT - Powered by llama3:8b")
        print("=" * 80)
        print(f"Model: {self.model}")
        print(f"✓ Running locally (No API limits, instant response)")
        print("\nType 'exit', 'quit', or 'bye' to end the conversation")
        print("Type 'clear' to clear conversation history\n")
        print("=" * 80 + "\n")
    
    def chat(self, user_message):
        """Send a message and get response from Ollama."""
        
        # Add user message to history
        self.conversation_history.append(f"User: {user_message}")
        
        # Build conversation context (last 10 exchanges)
        context = "\n".join(self.conversation_history[-20:])
        prompt = f"{context}\nAssistant:"
        
        try:
            print("🤖 Thinking...", end="", flush=True)
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                stream=False,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": 500
                }
            )
            print("\r" + " " * 30 + "\r", end="", flush=True)  # Clear "thinking" message
            
            if response and "response" in response:
                bot_reply = response["response"].strip()
                
                # Clean up response
                if bot_reply.startswith("Assistant:"):
                    bot_reply = bot_reply[10:].strip()
                
                # Add to history
                self.conversation_history.append(f"Assistant: {bot_reply}")
                
                return bot_reply
            else:
                return "❌ Error: No response from Ollama"
        
        except Exception as e:
            return f"❌ Error: {e}"
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("🗑️  Conversation history cleared!\n")
    
    def run(self):
        """Run the interactive chatbot loop."""
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit commands
                if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                    print("\n👋 Goodbye! Thanks for chatting!\n")
                    break
                
                # Check for clear command
                if user_input.lower() == "clear":
                    self.clear_history()
                    continue
                
                # Get response from Ollama
                response = self.chat(user_input)
                
                # Print response
                print(f"\n🤖 Assistant: {response}\n")
                print("-" * 80 + "\n")
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for chatting!\n")
                break
            
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


def main():
    """Main entry point."""
    try:
        chatbot = OllamaChatbot()
        chatbot.run()
    except ImportError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ Failed to start chatbot: {e}")


if __name__ == "__main__":
    main()
