"""Score human annotations + inter-annotator agreement.

Expected JSONL schema (output from ab_packager after annotators edit):
{
  "item_id": "...",
  "annotations": [
     {"annotator": "ann1", "faithfulness": 2, "relevance": 2, "style": 1, "winner": "A"},
     ...
  ]
}

Outputs summary metrics and simple agreement (Cohen's kappa) when 2 annotators exist.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _kappa(a: List[str], b: List[str]) -> float:
    # simple unweighted Cohen's kappa for categorical labels
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return 0.0
    labels = sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labels)}

    # confusion matrix
    m = [[0 for _ in labels] for _ in labels]
    for x, y in zip(a, b):
        m[idx[x]][idx[y]] += 1

    po = sum(m[i][i] for i in range(len(labels))) / n

    # expected agreement
    row = [sum(m[i]) for i in range(len(labels))]
    col = [sum(m[i][j] for i in range(len(labels))) for j in range(len(labels))]
    pe = sum((row[i] / n) * (col[i] / n) for i in range(len(labels)))

    denom = (1.0 - pe)
    return (po - pe) / denom if denom > 1e-12 else 0.0


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Score human A/B annotations")
    parser.add_argument("--input", required=True)
    args = parser.parse_args(argv)

    items = load_jsonl(args.input)

    faith, rel, style = [], [], []
    winners = []

    per_ann_winners: Dict[str, List[str]] = defaultdict(list)

    for it in items:
        anns = it.get("annotations") or []
        for a in anns:
            if "faithfulness" in a:
                faith.append(float(a["faithfulness"]))
            if "relevance" in a:
                rel.append(float(a["relevance"]))
            if "style" in a:
                style.append(float(a["style"]))
            w = str(a.get("winner") or "").strip().upper()
            if w in {"A", "B"}:
                winners.append(w)
                ann = str(a.get("annotator") or "unknown")
                per_ann_winners[ann].append(w)

    def avg(xs: List[float]) -> float:
        return sum(xs) / max(1, len(xs))

    summary = {
        "n_items": len(items),
        "n_annotations": sum(len(it.get("annotations") or []) for it in items),
        "faithfulness_mean": avg(faith),
        "relevance_mean": avg(rel),
        "style_mean": avg(style),
        "winner_A_rate": (sum(1 for w in winners if w == "A") / max(1, len(winners))),
    }

    # Agreement (only when exactly 2 annotators have full coverage)
    if len(per_ann_winners) == 2:
        ann_names = sorted(per_ann_winners.keys())
        a1, a2 = per_ann_winners[ann_names[0]], per_ann_winners[ann_names[1]]
        n = min(len(a1), len(a2))
        summary["cohen_kappa_winner"] = _kappa(a1[:n], a2[:n])
    else:
        summary["cohen_kappa_winner"] = 0.0

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
