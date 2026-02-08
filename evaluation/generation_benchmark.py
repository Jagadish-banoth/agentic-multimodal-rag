"""FAANG-style generation evaluation.

Computes automatic metrics over JSONL predictions:
- BLEU (sacrebleu)
- ROUGE (rouge_score)
- METEOR (nltk)
- BERTScore (bert_score)
- COMET (optional; heavy)

Input JSONL schema (one per line):
{
  "query_id": "q1",
  "query": "...",
  "generated": "...",
  "gold": {"answers": ["..."]}
}

You can also feed it outputs produced by scripts/run_ragas_pipeline.py if you add `gold`.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _safe_imports():
    out = {}
    try:
        import sacrebleu  # type: ignore

        out["sacrebleu"] = sacrebleu
    except Exception:
        out["sacrebleu"] = None

    try:
        from rouge_score import rouge_scorer  # type: ignore

        out["rouge_scorer"] = rouge_scorer
    except Exception:
        out["rouge_scorer"] = None

    try:
        from bert_score import score as bert_score  # type: ignore

        out["bert_score"] = bert_score
    except Exception:
        out["bert_score"] = None

    try:
        import nltk  # type: ignore
        from nltk.translate.meteor_score import meteor_score  # type: ignore

        # ensure punkt exists if possible
        try:
            _ = nltk.data.find("tokenizers/punkt")
        except Exception:
            pass

        out["nltk"] = nltk
        out["meteor_score"] = meteor_score
    except Exception:
        out["nltk"] = None
        out["meteor_score"] = None

    # COMET is optional and heavy; support if installed.
    # Package name is typically `unbabel-comet` and module `comet`.
    try:
        from comet import download_model, load_from_checkpoint  # type: ignore

        out["comet_download_model"] = download_model
        out["comet_load"] = load_from_checkpoint
    except Exception:
        out["comet_download_model"] = None
        out["comet_load"] = None

    return out


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"Empty input JSONL: {path}")
    return rows


def _extract_ref(item: Dict[str, Any]) -> str:
    gold = item.get("gold")
    if isinstance(gold, dict):
        answers = gold.get("answers")
        if isinstance(answers, list) and answers:
            return str(answers[0])
    if isinstance(gold, list) and gold:
        return str(gold[0])
    if isinstance(gold, str):
        return gold

    # fallbacks
    if "reference" in item:
        return str(item.get("reference") or "")
    if "expected" in item:
        return str(item.get("expected") or "")
    return ""


def evaluate(path: str, *, bert_lang: str = "en", comet_model: Optional[str] = None) -> Dict[str, float]:
    libs = _safe_imports()
    rows = load_jsonl(path)

    refs: List[str] = []
    hyps: List[str] = []
    for r in rows:
        refs.append(_extract_ref(r))
        hyps.append(str(r.get("generated") or r.get("answer") or ""))

    scores: Dict[str, float] = {}

    # BLEU
    sacrebleu = libs["sacrebleu"]
    if sacrebleu is not None:
        scores["bleu"] = float(sacrebleu.corpus_bleu(hyps, [refs]).score)
    else:
        scores["bleu"] = 0.0

    # ROUGE
    rouge_scorer = libs["rouge_scorer"]
    if rouge_scorer is not None:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1, r2, rl = [], [], []
        for ref, hyp in zip(refs, hyps):
            s = scorer.score(ref, hyp)
            r1.append(s["rouge1"].fmeasure)
            r2.append(s["rouge2"].fmeasure)
            rl.append(s["rougeL"].fmeasure)
        scores["rouge1"] = sum(r1) / max(1, len(r1))
        scores["rouge2"] = sum(r2) / max(1, len(r2))
        scores["rougeL"] = sum(rl) / max(1, len(rl))
    else:
        scores["rouge1"] = scores["rouge2"] = scores["rougeL"] = 0.0

    # METEOR
    meteor_fn = libs["meteor_score"]
    if meteor_fn is not None:
        vals = []
        for ref, hyp in zip(refs, hyps):
            try:
                vals.append(float(meteor_fn([ref], hyp)))
            except Exception:
                vals.append(0.0)
        scores["meteor"] = sum(vals) / max(1, len(vals))
    else:
        scores["meteor"] = 0.0

    # BERTScore
    bert_score = libs["bert_score"]
    if bert_score is not None:
        try:
            _P, _R, F = bert_score(hyps, refs, lang=bert_lang, rescale_with_baseline=True, verbose=False)
            scores["bertscore_f1"] = float(F.mean())
        except Exception:
            scores["bertscore_f1"] = 0.0
    else:
        scores["bertscore_f1"] = 0.0

    # COMET (optional)
    if comet_model and libs["comet_download_model"] is not None and libs["comet_load"] is not None:
        try:
            model_path = libs["comet_download_model"](comet_model)
            model = libs["comet_load"](model_path)
            data = [{"src": "", "mt": h, "ref": r} for h, r in zip(hyps, refs)]
            comet_out = model.predict(data, batch_size=8, gpus=0)
            scores["comet"] = float(sum(comet_out.scores) / max(1, len(comet_out.scores)))
        except Exception:
            scores["comet"] = 0.0
    else:
        scores["comet"] = 0.0

    return scores


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="FAANG-style generation benchmark")
    parser.add_argument("--input", required=True, help="Input JSONL with generated + gold")
    parser.add_argument("--out", default=None, help="Optional output JSON file")
    parser.add_argument("--bert_lang", default="en")
    parser.add_argument("--comet_model", default=None, help="Optional COMET model name (heavy)")
    args = parser.parse_args(argv)

    scores = evaluate(args.input, bert_lang=args.bert_lang, comet_model=args.comet_model)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2)

    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
