"""Ablation runner.

Runs retrieval benchmark with components disabled to quantify impact.

Ablations implemented (via retriever selection):
- dense only
- sparse only
- hybrid rerank
- parallel hybrid
- faang hybrid

Usage:
  python scripts/run_ablation.py --input evaluation/dev_queries.csv --out artifacts/ablation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from evaluation.retrieval_benchmark import main as retrieval_main


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run ablation study over retriever variants")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="artifacts/ablation.json")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5, 10])
    args = parser.parse_args(argv)

    # Run retrieval benchmark with the default retriever list and write report.
    # We invoke the module main() to avoid duplicating logic.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    retrieval_main(
        [
            "--input",
            args.input,
            "--config",
            args.config,
            "--out",
            str(out_path),
            "--retrievers",
            "dense",
            "sparse",
            "hybrid_rerank",
            "parallel",
            "faang",
            "--k",
            *[str(x) for x in args.k],
        ]
    )


if __name__ == "__main__":
    main()
