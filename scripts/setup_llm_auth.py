"""
scripts/setup_llm_auth.py
=========================
Quick setup script for LLM authentication.

Usage:
    python scripts/setup_llm_auth.py --method hf --token hf_xxxxx
    python scripts/setup_llm_auth.py --method ollama
    python scripts/setup_llm_auth.py --check
"""

import os
import sys
import argparse
from pathlib import Path


def setup_hf_token(token: str):
    """Set up HuggingFace token."""
    print("🔐 Setting HuggingFace API Token...\n")
    
    if not token.startswith("hf_"):
        print("❌ ERROR: Token must start with 'hf_'")
        print("Get token from: https://huggingface.co/settings/tokens")
        return False
    
    # Method 1: Environment variable (session-specific)
    os.environ["HF_TOKEN"] = token
    print("✅ Token set in environment (current session)")
    
    # Method 2: .env file (persistent)
    env_file = Path(__file__).parents[1] / ".env"
    
    # Read existing .env
    existing = {}
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, val = line.strip().split("=", 1)
                    existing[key] = val
    
    # Update token
    existing["HF_TOKEN"] = token
    
    # Write back
    with open(env_file, "w") as f:
        for key, val in existing.items():
            f.write(f"{key}={val}\n")
    
    print(f"✅ Token saved to .env file: {env_file}")
    
    # Verify token
    print("\n🧪 Verifying token...")
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=token)
        print("✅ Token is valid!")
        return True
    except Exception as e:
        print(f"❌ Token verification failed: {e}")
        return False


def setup_ollama():
    """Set up Ollama."""
    print("🚀 Setting up Ollama...\n")
    
    # Check if Ollama is installed
    import shutil
    if not shutil.which("ollama"):
        print("❌ Ollama is not installed!")
        print("\nTo install Ollama:")
        print("  Windows: Download from https://ollama.ai/download")
        print("  Or: choco install ollama")
        print("\nThen add Ollama to PATH")
        return False
    
    print("✅ Ollama is installed")
    
    # Update config
    config_file = Path(__file__).parents[1] / "config" / "settings.yaml"
    
    with open(config_file, "r") as f:
        content = f.read()
    
    # Replace settings
    content = content.replace("use_api: true", "use_api: false")
    
    with open(config_file, "w") as f:
        f.write(content)
    
    print("✅ Config updated to use local Ollama")
    print("\n📝 Next steps:")
    print("  1. Open another PowerShell window")
    print("  2. Run: ollama serve")
    print("  3. In first window, run: python -m planner.llm_planner")
    
    return True


def check_setup():
    """Check current LLM setup."""
    print("🔍 Checking LLM setup...\n")
    
    # Check config
    config_file = Path(__file__).parents[1] / "config" / "settings.yaml"
    with open(config_file, "r") as f:
        content = f.read()
    
    use_api = "use_api: true" in content
    
    if use_api:
        print("📍 Config: Using HuggingFace API")
        
        # Check HF token
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            print(f"✅ HF_TOKEN is set: {hf_token[:20]}...")
            
            # Verify token
            try:
                from huggingface_hub import InferenceClient
                client = InferenceClient(token=hf_token)
                print("✅ Token is valid and working!")
            except Exception as e:
                print(f"❌ Token is invalid: {e}")
        else:
            print("❌ HF_TOKEN is NOT set")
            print("\nSet it with:")
            print("  $env:HF_TOKEN = 'hf_xxxxx'")
    else:
        print("📍 Config: Using local Ollama")
        
        # Check Ollama
        import shutil
        if shutil.which("ollama"):
            print("✅ Ollama is installed")
            print("\nMake sure Ollama is running:")
            print("  ollama serve")
        else:
            print("❌ Ollama is not installed")
    
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    if use_api and not os.getenv("HF_TOKEN"):
        print("❌ HF API is configured but token is missing!")
        print("   Run: python scripts/setup_llm_auth.py --method hf --token hf_xxxxx")
    elif use_api and os.getenv("HF_TOKEN"):
        print("✅ Setup looks good! Ready to use planner.")
    else:
        print("✅ Ollama is configured. Make sure it's running.")


def main():
    parser = argparse.ArgumentParser(
        description="Setup LLM authentication for agentic planner"
    )
    parser.add_argument(
        "--method",
        choices=["hf", "ollama"],
        help="Setup method (HuggingFace API or local Ollama)"
    )
    parser.add_argument(
        "--token",
        help="HuggingFace API token (required with --method hf)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check current setup"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("LLM AUTHENTICATION SETUP")
    print("="*60)
    
    if args.check:
        check_setup()
    
    elif args.method == "hf":
        if not args.token:
            print("\n❌ ERROR: Token required with --method hf")
            print("\nUsage:")
            print("  python scripts/setup_llm_auth.py --method hf --token hf_xxxxx")
            print("\nGet token from: https://huggingface.co/settings/tokens")
            sys.exit(1)
        
        if setup_hf_token(args.token):
            print("\n✅ HuggingFace setup complete!")
            print("\nTest with:")
            print("  python -m planner.llm_planner")
        else:
            sys.exit(1)
    
    elif args.method == "ollama":
        if setup_ollama():
            print("\n✅ Ollama setup complete!")
        else:
            sys.exit(1)
    
    else:
        print("\n❌ Please specify --method hf or --method ollama or --check")
        print("\nExamples:")
        print("  python scripts/setup_llm_auth.py --check")
        print("  python scripts/setup_llm_auth.py --method hf --token hf_xxxxx")
        print("  python scripts/setup_llm_auth.py --method ollama")


if __name__ == "__main__":
    main()
