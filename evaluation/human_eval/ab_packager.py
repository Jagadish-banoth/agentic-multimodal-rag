"""Create blind A/B human-eval packs.

This produces a JSONL file for annotation with randomized A/B ordering.
Annotators fill: faithfulness, relevance, style, winner.

Input JSONL expected fields:
- query_id (optional)
- query
- candidate_a
- candidate_b
- (optional) retrieved / sources for grounding

Output JSONL fields:
- item_id
- query
- a_text / b_text
- a_label / b_label (randomized)
- rubric (faithfulness, relevance, style)
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any, Dict, List, Optional, Sequence


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Package blind A/B annotation tasks")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    random.seed(args.seed)
    rows = load_jsonl(args.input)

    with open(args.out, "w", encoding="utf-8") as outf:
        for i, r in enumerate(rows):
            query = r.get("query") or r.get("question") or ""
            a = str(r.get("candidate_a") or "")
            b = str(r.get("candidate_b") or "")
            if not query or not a or not b:
                continue

            flip = random.random() < 0.5
            if flip:
                a_text, b_text = b, a
                a_label, b_label = "B", "A"
            else:
                a_text, b_text = a, b
                a_label, b_label = "A", "B"

            item = {
                "item_id": str(r.get("query_id") or f"item_{i}"),
                "query": query,
                "a_text": a_text,
                "b_text": b_text,
                "a_label": a_label,
                "b_label": b_label,
                "rubric": {
                    "faithfulness": "0-2 (0=hallucinated, 2=fully grounded)",
                    "relevance": "0-2 (0=off-topic, 2=answers question)",
                    "style": "0-2 (0=unclear, 2=clear/professional)",
                    "winner": "A or B",
                },
                "annotations": [],
            }
            outf.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
