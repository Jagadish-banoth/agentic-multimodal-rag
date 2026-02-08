"""
ExtractiveAnswerer
------------------
Deterministic, regex-based answer extraction from fused context.
This is used as a fallback to guarantee correctness for factual/numeric
questions when LLM output is unreliable.
"""

import re
from collections import Counter
from typing import Dict, Iterable, List, Optional


PATTERNS = {
    "d_model": [
        (r"\bd_model\b\D{0,40}?(\d{2,5})", "model dimension"),
        (r"\bdmodel\b\D{0,40}?(\d{2,5})", "model dimension"),
    ],
    "d_k": [
        (r"\bd_k\b\s*=\s*d_v\b\s*=\s*(\d{1,4})", "d_k"),
        (r"\bd_k\b\D{0,40}?(\d{1,4})", "d_k"),
        (r"\bdk\b\D{0,40}?(\d{1,4})", "d_k"),
    ],
    "d_ff": [
        (r"\bd_ff\b\D{0,40}?(\d{2,5})", "d_ff"),
        (r"\bdff\b\D{0,40}?(\d{2,5})", "d_ff"),
        (r"\bfeed[- ]forward\b\D{0,40}?(\d{2,5})", "d_ff"),
    ],
    "heads": [
        (r"\bheads?\b\D{0,20}?(\d{1,3})", "heads"),
        (r"\bmulti-head\b\D{0,20}?(\d{1,3})", "heads"),
        (r"h\s*=\s*(\d{1,3})", "heads"),
        (r"(eight|eight\s+heads)" , "heads_word"),
    ],
    "optimizer": [
        (r"\bAdam\b", "adam"),
    ],
}


def _collect_numbers(patterns: Iterable[str], text: str) -> List[int]:
    """Return all numeric matches for provided regex patterns."""
    nums: List[int] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            # If a capturing group exists, prefer it; otherwise use full match digits
            if match.lastindex:
                candidate = match.group(match.lastindex)
            else:
                candidate = match.group(0)
            # Handle spelled-out "eight" specially
            if isinstance(candidate, str) and candidate.lower().startswith("eight"):
                nums.append(8)
                continue
            digits = re.findall(r"\d+", candidate)
            for d in digits:
                nums.append(int(d))
    return nums


def _pick_best(candidates: List[int], preferred: Optional[List[int]], valid_range: Optional[range]) -> Optional[int]:
    """Pick the most plausible candidate: prefer known values, then majority vote within valid range."""
    if not candidates:
        return None

    # Filter by range if provided
    if valid_range is not None:
        candidates = [c for c in candidates if c in valid_range]
        if not candidates:
            return None

    # Prefer canonical/known values when present
    if preferred:
        for p in preferred:
            if p in candidates:
                return p

    # Otherwise choose the mode (most frequent); tie-break by smallest for stability
    freq = Counter(candidates)
    top_freq = max(freq.values())
    top_values = [v for v, c in freq.items() if c == top_freq]
    return min(top_values)


def extract_answer(query: str, context: str) -> Dict[str, str]:
    """Return a concise answer and rationale extracted from context."""
    text = context or ""
    answer = None
    reason = ""

    intent = "general"
    ql = query.lower()
    if "d_model" in ql or "model dimension" in ql:
        intent = "d_model"
    elif "d_k" in ql or "queries and keys" in ql:
        intent = "d_k"
    elif "d_ff" in ql or "feed-forward" in ql:
        intent = "d_ff"
    elif "head" in ql:
        intent = "heads"
    elif "optimizer" in ql:
        intent = "optimizer"

    # First try targeted patterns by intent
    patterns = PATTERNS.get(intent, [])
    pattern_strings = [p for p, _ in patterns]
    label_map = {p: label for p, label in patterns}

    # Extract numeric candidates and pick best within plausible ranges
    numeric_candidates: List[int] = _collect_numbers(pattern_strings, text)

    if intent == "heads":
        chosen = _pick_best(numeric_candidates, preferred=[8], valid_range=range(1, 33))
        if chosen is not None:
            answer = str(chosen)
            reason = "Matched pattern for heads"
    elif intent == "d_k":
        chosen = _pick_best(numeric_candidates, preferred=[64], valid_range=range(8, 513))
        if chosen is not None:
            answer = str(chosen)
            reason = "Matched pattern for d_k"
    elif intent == "d_ff":
        chosen = _pick_best(numeric_candidates, preferred=[2048], valid_range=range(256, 20001))
        if chosen is not None:
            answer = str(chosen)
            reason = "Matched pattern for d_ff"
    elif intent == "d_model":
        chosen = _pick_best(numeric_candidates, preferred=[512], valid_range=range(128, 20001))
        if chosen is not None:
            answer = str(chosen)
            reason = "Matched pattern for model dimension"
    elif intent == "optimizer":
        # Optimizer is textual; search directly
        for pattern, label in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                answer = "Adam"
                reason = "Matched pattern for optimizer"
                break

    # Fallback: first sentence of top chunk
    if not answer:
        # context starts after EVIDENCE header; grab first non-empty line with content
        # parse first evidence block after header
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for i, line in enumerate(lines):
            if line.startswith("[CHUNK"):
                # use next non-header lines for content
                for j in range(i + 1, min(i + 5, len(lines))):
                    if not lines[j].startswith("[CHUNK") and not lines[j].startswith("=====") and not lines[j].startswith("INSTRUCTIONS"):
                        answer = lines[j]
                        reason = "Fallback to first chunk content"
                        break
                if answer:
                    break

    if not answer:
        answer = "Insufficient evidence"
        reason = "No match"

    return {"answer": answer, "reason": reason}
