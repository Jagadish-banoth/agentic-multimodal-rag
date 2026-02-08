import subprocess
import os
from pathlib import Path
import sys


def run_embed_and_index():
    """Run the embedding and indexing script as a subprocess."""
    script_path = Path(__file__).parent.parent / "scripts" / "embed_and_index.py"
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode


def deduplicate_raw_files(raw_dir: Path):
    """
    Remove duplicate files in the raw_dir by name and size (keeps only one copy).
    """
    seen = {}
    for f in raw_dir.glob("*"):
        if not f.is_file():
            continue
        key = (f.name, f.stat().st_size)
        if key in seen:
            f.unlink()  # Remove duplicate
        else:
            seen[key] = f
    return len(seen)


def cleanup_temp_files(raw_dir: Path, keep_latest=1):
    """
    Remove all but the latest 'keep_latest' files in the raw_dir.
    """
    files = sorted([f for f in raw_dir.glob("*") if f.is_file()], key=lambda x: x.stat().st_mtime, reverse=True)
    for f in files[keep_latest:]:
        f.unlink()
    return len(files[:keep_latest])
