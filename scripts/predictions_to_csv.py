"""
Convert predictions.jsonl to CSV for demo purposes.
"""
import json
import csv
from pathlib import Path

INPUT_FILE = Path("artifacts/predictions.jsonl")
OUTPUT_FILE = Path("artifacts/predictions_demo.csv")

def truncate(text, max_len=500):
    if text and len(text) > max_len:
        return text[:max_len] + "..."
    return text or ""

def safe_get(d, *keys, default=""):
    """Safely navigate nested dict."""
    val = d
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k, default)
        else:
            return default
    return val if val is not None else default

def main():
    records = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"Loaded {len(records)} records from {INPUT_FILE}")
    
    # Extract fields for CSV
    rows = []
    for rec in records:
        gen_struct = rec.get("generated_structured", {})
        retrieved = rec.get("retrieved", [])
        
        # Get top sources from retrieved
        top_sources = []
        for r in retrieved[:3]:  # Top 3 sources
            source = r.get("source", "") or r.get("metadata", {}).get("source", "")
            if source:
                top_sources.append(Path(source).name)
        
        row = {
            "Query ID": rec.get("query_id", ""),
            "Query": truncate(rec.get("query", ""), 200),
            "Generated Answer": truncate(rec.get("generated", ""), 500),
            "Confidence": round(safe_get(gen_struct, "confidence", default=0), 4),
            "Model": safe_get(gen_struct, "model", default=""),
            "Tokens Used": safe_get(gen_struct, "tokens_used", default=0),
            "Num Sources Used": len(safe_get(gen_struct, "sources_used", default=[])),
            "Num Evidence Claims": len(safe_get(gen_struct, "evidence", default=[])),
            "Num Retrieved": len(retrieved),
            "Top Sources": "; ".join(top_sources) if top_sources else "",
            "Top Rerank Score": round(retrieved[0].get("rerank_score", 0), 4) if retrieved else 0,
        }
        rows.append(row)
    
    # Write CSV
    fieldnames = list(rows[0].keys()) if rows else []
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"CSV written to {OUTPUT_FILE}")
    print(f"Columns: {fieldnames}")

if __name__ == "__main__":
    main()
