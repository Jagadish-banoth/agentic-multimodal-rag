"""
verification/verifier.py
-------------------------
CONTROL PLANE component for SOTA answer verification and confidence scoring.

SOTA Features:
- Strict citation validation against chunk_map
- NLI-based claim verification per citation
- Evidence-grounded confidence scoring
- Machine-checkable citation coverage
- Trigger re-planning when grounding fails

Responsibilities:
- Check answer faithfulness to retrieved evidence
- Validate citations [C1], [C2], etc. against chunk_map
- Score confidence based on evidence support
- Detect hallucination / unsupported claims
- Trigger re-planning when confidence is low
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

import os
log_path = os.path.join(os.path.dirname(__file__), '../logs/verifier.log')
log_path = os.path.abspath(log_path)
logger = logging.getLogger("verifier")
if not logger.hasHandlers():
    handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class Verifier:
    """
    SOTA Answer verification with strict citation checking.

    Part of the CONTROL PLANE - decides whether to accept or retry.
    Citations are expected in [filename#chunk_id] format.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.confidence_threshold = config.get("agent", {}).get("confidence_threshold", 0.7)
        vcfg = config.get("verification", {})
        self.enabled = vcfg.get("enabled", True)
        self.nli_model = vcfg.get("faithfulness_model", "microsoft/deberta-v3-base")
        self.nli_threshold = vcfg.get("faithfulness_threshold", 0.65)
        self.nli_batch_size = vcfg.get("batch_size", 8)
        self.max_pairs = vcfg.get("max_pairs", 8)
        self.device = self._resolve_device(vcfg.get("device", "auto"))
        self._nli_pipeline = None
        
        # SOTA: Strict citation checking settings
        self.strict_citations = vcfg.get("strict_citations", True)
        self.min_citation_ratio = vcfg.get("min_citation_ratio", 0.3)  # At least 30% sentences cited
    
    def verify(
        self,
        query: str,
        answer: Dict,
        context_chunks: List[Dict],
        chunk_map: Optional[Dict[str, Dict]] = None
    ) -> Dict:
        """
        SOTA: Verify answer quality with strict citation validation.
        
        Args:
            query: Original user query
            answer: Generated answer dict from LLM (with evidence, sources_used)
            context_chunks: Retrieved evidence chunks
            chunk_map: Citation ID to chunk mapping from fusion layer
        
        Returns:
            Verification result with confidence score and decision
        """
        if not answer:
            return self._failed_result("Empty answer")
        
        # Extract answer text
        answer_text = self._extract_answer_text(answer)
        
        # Check for explicit abstention
        if self._is_abstention(answer_text):
            return {
                "verified": True,
                "confidence": 0.95,
                "level": "HIGH",
                "reason": "Properly abstained due to insufficient evidence",
                "should_retry": False,
                "citation_valid": True,
                "evidence_verified": []
            }
        
        # SOTA: Validate citations against chunk_map
        citation_result = self._verify_citations_strict(answer, chunk_map)
        
        # Build claim/evidence pairs for NLI
        pairs = self._build_claim_pairs_sota(answer, chunk_map)
        
        # Score with NLI (or fallback)
        evidence_score = self._score_with_nli(pairs) if pairs else self._compute_evidence_support(answer, context_chunks)
        
        # SOTA: Per-claim verification results
        evidence_verified = self._verify_evidence_claims(answer, chunk_map)
        
        # Check hedging
        hedging_penalty = self._check_hedging(answer_text)
        
        # Compute final confidence (SOTA: includes citation validity)
        confidence = self._compute_confidence_sota(
            evidence_score=evidence_score,
            citation_result=citation_result,
            hedging_penalty=hedging_penalty,
            answer=answer,
            evidence_verified=evidence_verified
        )
        
        # Determine level
        level = self._confidence_to_level(confidence)
        
        # Decision
        verified = confidence >= self.confidence_threshold
        should_retry = not verified and confidence > 0.25
        
        return {
            "verified": verified,
            "confidence": round(confidence, 3),
            "level": level,
            "evidence_score": round(evidence_score, 3),
            "citation_score": round(citation_result["score"], 3),
            "citation_valid": citation_result["all_valid"],
            "citations_used": citation_result["citations_used"],
            "invalid_citations": citation_result["invalid"],
            "reason": self._get_reason_sota(confidence, evidence_score, citation_result),
            "should_retry": should_retry,
            "pairs_evaluated": len(pairs),
            "evidence_verified": evidence_verified,
            "nli_model": self.nli_model if self.enabled else None
        }
    
    def _failed_result(self, reason: str) -> Dict:
        """Return a failed verification result."""
        return {
            "verified": False,
            "confidence": 0.0,
            "level": "LOW",
            "reason": reason,
            "should_retry": True,
            "citation_valid": False,
            "evidence_verified": []
        }

    # --------------------------------------------------
    # SOTA: STRICT CITATION VALIDATION
    # --------------------------------------------------
    def _verify_citations_strict(
        self,
        answer: Dict,
        chunk_map: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Any]:
        """
        SOTA: Strictly validate citations against chunk_map.
        
        Returns:
            {
                "all_valid": bool,
                "score": float,
                "citations_used": List[str],
                "valid": List[str],
                "invalid": List[str],
                "coverage_ratio": float
            }
        """
        # Get citations used from answer
        sources_used = answer.get("sources_used", [])
        evidence_list = answer.get("evidence", [])
        
        # Also extract from answer text directly
        answer_text = self._extract_answer_text(answer)
        text_tokens = re.findall(r"\[([^\[\]]+?)\]", answer_text)
        text_cite_ids = []
        for token in text_tokens:
            token = token.strip()
            if chunk_map and token in chunk_map:
                text_cite_ids.append(token)
            elif "#" in token:
                text_cite_ids.append(token)
        
        # Combine all citation sources
        all_citations = set(sources_used) | set(text_cite_ids)
        for ev in evidence_list:
            cite_ids = ev.get("cite_ids", [])
            all_citations.update(cite_ids)
        
        citations_used = list(all_citations)
        
        if not citations_used:
            # No citations at all - penalty
            return {
                "all_valid": False,
                "score": 0.2,
                "citations_used": [],
                "valid": [],
                "invalid": [],
                "coverage_ratio": 0.0
            }
        
        # Validate against chunk_map
        valid = []
        invalid = []
        
        if chunk_map:
            for cite_id in citations_used:
                if cite_id in chunk_map:
                    valid.append(cite_id)
                else:
                    invalid.append(cite_id)
        else:
            # No chunk_map - assume all citations are valid (backward compat)
            valid = citations_used
        
        all_valid = len(invalid) == 0 and len(valid) > 0
        
        # Calculate coverage (what % of answer sentences are cited)
        sentences = self._split_sentences(answer_text)
        cited_sentences = 0
        for sent in sentences:
            if re.search(r"\[[^\[\]]+\]", sent):
                cited_sentences += 1
        
        coverage_ratio = cited_sentences / max(len(sentences), 1)
        
        # Calculate score
        if not citations_used:
            score = 0.2
        else:
            validity_ratio = len(valid) / len(citations_used)
            score = 0.3 + (0.4 * validity_ratio) + (0.3 * min(1.0, coverage_ratio / self.min_citation_ratio))
        
        return {
            "all_valid": all_valid,
            "score": score,
            "citations_used": citations_used,
            "valid": valid,
            "invalid": invalid,
            "coverage_ratio": coverage_ratio
        }
    
    def _build_claim_pairs_sota(
        self,
        answer: Dict,
        chunk_map: Optional[Dict[str, Dict]] = None
    ) -> List[Tuple[str, str]]:
        """
        SOTA: Build claim/evidence pairs using structured evidence from generator.
        """
        pairs: List[Tuple[str, str]] = []
        
        evidence_list = answer.get("evidence", [])
        
        for ev in evidence_list:
            claim = ev.get("claim", "")
            cite_ids = ev.get("cite_ids", [])
            
            if not claim or not cite_ids:
                continue
            
            # Get evidence text from chunk_map
            for cite_id in cite_ids:
                if chunk_map and cite_id in chunk_map:
                    premise = chunk_map[cite_id].get("content", "")
                    if premise and claim:
                        pairs.append((premise, claim))
                        if len(pairs) >= self.max_pairs:
                            return pairs
        
        # Fallback: if no structured evidence, use answer text vs chunk_map
        if len(pairs) < 2 and chunk_map:
            answer_text = self._extract_answer_text(answer)
            sentences = self._split_sentences(answer_text)
            
            for sent in sentences[:self.max_pairs]:
                # Find citations in sentence
                cite_matches = re.findall(r"\[([^\[\]]+?)\]", sent)
                claim = re.sub(r"\[[^\[\]]+\]", "", sent).strip()

                for token in cite_matches:
                    cite_id = token.strip()
                    if cite_id in chunk_map:
                        premise = chunk_map[cite_id].get("content", "")
                        if premise and claim:
                            pairs.append((premise, claim))
                            if len(pairs) >= self.max_pairs:
                                return pairs
        
        return pairs
    
    def _verify_evidence_claims(
        self,
        answer: Dict,
        chunk_map: Optional[Dict[str, Dict]] = None
    ) -> List[Dict[str, Any]]:
        """
        SOTA: Verify each evidence claim against its cited chunks.
        
        Returns list of verification results per claim.
        """
        results = []
        evidence_list = answer.get("evidence", [])
        
        pipe = self._get_nli_pipeline()
        
        for ev in evidence_list:
            claim = ev.get("claim", "")
            cite_ids = ev.get("cite_ids", [])
            
            result = {
                "claim": claim,
                "cite_ids": cite_ids,
                "supported": False,
                "score": 0.0
            }
            
            if not claim or not cite_ids or not chunk_map:
                results.append(result)
                continue
            
            # Check if any cited chunk supports the claim
            max_score = 0.0
            for cite_id in cite_ids:
                if cite_id not in chunk_map:
                    continue
                
                premise = chunk_map[cite_id].get("content", "")
                if not premise:
                    continue
                
                # Use NLI if available
                if pipe:
                    try:
                        out = pipe({"text": premise, "text_pair": claim})
                        score = self._extract_entailment_score(out)
                        max_score = max(max_score, score)
                    except Exception:
                        # Fallback to overlap
                        score = self._compute_overlap_score(premise, claim)
                        max_score = max(max_score, score)
                else:
                    score = self._compute_overlap_score(premise, claim)
                    max_score = max(max_score, score)
            
            result["score"] = max_score
            result["supported"] = max_score >= self.nli_threshold
            results.append(result)
        
        return results
    
    def _compute_overlap_score(self, premise: str, hypothesis: str) -> float:
        """Compute term overlap score between premise and hypothesis."""
        p_terms = set(re.findall(r"\b\w{3,}\b", premise.lower()))
        h_terms = set(re.findall(r"\b\w{3,}\b", hypothesis.lower()))
        if not h_terms:
            return 0.0
        coverage = len(p_terms & h_terms) / max(len(h_terms), 1)
        return min(1.0, coverage * 1.3)
    
    def _compute_confidence_sota(
        self,
        evidence_score: float,
        citation_result: Dict,
        hedging_penalty: float,
        answer: Dict,
        evidence_verified: List[Dict]
    ) -> float:
        """
        SOTA: Compute confidence with citation validity weighting.
        
        Weights:
          - Evidence/NLI support: 35%
          - Citation validity & coverage: 30%
          - Per-claim verification: 20%
          - LLM confidence: 15%
        """
        # LLM's own confidence
        answer_confidence = 0.5
        if isinstance(answer, dict):
            conf = answer.get("confidence")
            if isinstance(conf, dict):
                answer_confidence = conf.get("score", 0.5)
            elif isinstance(conf, (int, float)):
                answer_confidence = float(conf)
        
        # Per-claim verification score
        claim_score = 0.5
        if evidence_verified:
            supported_count = sum(1 for ev in evidence_verified if ev.get("supported"))
            claim_score = supported_count / len(evidence_verified)
        
        # Weighted combination
        raw_score = (
            0.35 * evidence_score +
            0.30 * citation_result["score"] +
            0.20 * claim_score +
            0.15 * answer_confidence
        )
        
        # Apply penalties
        final_score = raw_score - hedging_penalty
        
        # Penalty for invalid citations
        if citation_result.get("invalid"):
            invalid_penalty = len(citation_result["invalid"]) * 0.1
            final_score -= invalid_penalty
        
        return max(0.0, min(1.0, final_score))
    
    def _get_reason_sota(
        self,
        confidence: float,
        evidence_score: float,
        citation_result: Dict
    ) -> str:
        """Generate detailed reason for verification result."""
        reasons = []
        
        if citation_result.get("invalid"):
            reasons.append(f"Invalid citations: {citation_result['invalid']}")
        
        if citation_result["coverage_ratio"] < self.min_citation_ratio:
            reasons.append(f"Low citation coverage ({citation_result['coverage_ratio']:.0%})")
        
        if evidence_score < 0.5:
            reasons.append("Weak evidence support from NLI")
        
        if not reasons:
            if confidence >= 0.8:
                return f"Strong grounding: {len(citation_result['valid'])} valid citations, {evidence_score:.0%} NLI support"
            elif confidence >= 0.6:
                return "Moderate grounding with acceptable citations"
            else:
                return "Limited grounding"
        
        return "; ".join(reasons)

    # --------------------------------------------------
    # HELPER METHODS (from original, kept for compatibility)
    # --------------------------------------------------
    def _resolve_device(self, device_pref: str) -> str:
        if device_pref == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_pref

    def _get_nli_pipeline(self):
        """Lazy-load NLI model; fallback to overlap heuristic on failure."""
        if not self.enabled:
            return None

        if self._nli_pipeline is None:
            try:
                device_index = 0 if self.device.startswith("cuda") and torch.cuda.is_available() else -1
                tokenizer = AutoTokenizer.from_pretrained(self.nli_model, use_fast=True, trust_remote_code=True)
                model = AutoModelForSequenceClassification.from_pretrained(self.nli_model, trust_remote_code=True)
                self._nli_pipeline = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device=device_index,
                    return_all_scores=True,
                    truncation=True,
                    max_length=512,
                )
                logger.info(f"Verifier: loaded NLI model {self.nli_model} on {self.device}")
            except Exception as exc:
                logger.warning(f"Verifier: failed to load NLI model ({exc}); using overlap heuristic")
                self.enabled = False
                self._nli_pipeline = None
        return self._nli_pipeline

    def _score_with_nli(self, pairs: List[Tuple[str, str]]) -> float:
        pipe = self._get_nli_pipeline()
        if not pipe:
            return self._compute_overlap_from_pairs(pairs)

        try:
            inputs = [
                {"text": prem, "text_pair": hyp}
                for prem, hyp in pairs[: self.max_pairs]
            ]
            outputs = pipe(inputs, batch_size=self.nli_batch_size)

            entail_scores: List[float] = []
            for out in outputs:
                entail_score = self._extract_entailment_score(out)
                entail_scores.append(entail_score)

            if not entail_scores:
                return 0.0
            return sum(entail_scores) / len(entail_scores)
        except Exception as exc:
            logger.warning(f"Verifier: NLI scoring failed ({exc}); falling back to overlap")
            return self._compute_overlap_from_pairs(pairs)

    def _extract_entailment_score(self, scores: List[Dict]) -> float:
        for item in scores:
            label = item.get("label", "").lower()
            if "entail" in label:
                return float(item.get("score", 0.0))
        return 0.0

    def _compute_overlap_from_pairs(self, pairs: List[Tuple[str, str]]) -> float:
        """Heuristic overlap if NLI unavailable."""
        overlaps = []
        for premise, hyp in pairs:
            p_terms = set(re.findall(r"\b\w{3,}\b", premise.lower()))
            h_terms = set(re.findall(r"\b\w{3,}\b", hyp.lower()))
            if not h_terms:
                continue
            coverage = len(p_terms & h_terms) / max(len(h_terms), 1)
            overlaps.append(min(1.0, coverage * 1.2))
        return sum(overlaps) / len(overlaps) if overlaps else 0.0

    def _extract_chunk_text(self, chunk: Optional[Dict]) -> Optional[str]:
        if not chunk:
            return None
        fields = [
            "content",
            "combined_content",
            "snippet",
            "frame_caption",
            "caption",
            "frame_ocr",
            "ocr_text",
            "transcript",
            "audio_transcript",
        ]
        for key in fields:
            val = chunk.get(key)
            if val:
                return str(val)
        return None

    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter."""
        raw = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in raw if len(s.strip()) > 12]
        return sentences[: max(1, self.max_pairs)]
    
    def _extract_answer_text(self, answer: Dict) -> str:
        """Extract the main answer text from response."""
        if isinstance(answer, str):
            return answer
        
        for key in ["concise_answer", "answer_long", "answer", "text", "content"]:
            if key in answer and answer[key]:
                return str(answer[key])
        
        return str(answer)
    
    def _is_abstention(self, answer_text: str) -> bool:
        """Check if answer is an explicit abstention."""
        abstention_phrases = [
            "insufficient context",
            "i don't know",
            "cannot answer",
            "not enough information",
            "no evidence",
            "unable to determine",
            "not found in the context"
        ]
        answer_lower = answer_text.lower()
        return any(phrase in answer_lower for phrase in abstention_phrases)
    
    def _compute_evidence_support(
        self,
        answer: Dict,
        context_chunks: List[Dict]
    ) -> float:
        """Compute term overlap based evidence support (fallback)."""
        if not context_chunks:
            return 0.0
        
        answer_text = self._extract_answer_text(answer).lower()
        answer_terms = set(re.findall(r'\b\w{3,}\b', answer_text))
        if not answer_terms:
            return 0.5
        
        evidence_text = " ".join(
            c.get("content", "") for c in context_chunks
        ).lower()
        evidence_terms = set(re.findall(r'\b\w{3,}\b', evidence_text))
        
        if not evidence_terms:
            return 0.0
        
        overlap = answer_terms & evidence_terms
        coverage = len(overlap) / len(answer_terms) if answer_terms else 0
        
        return min(1.0, coverage * 1.2)
    
    def _check_hedging(self, answer_text: str) -> float:
        """
        Detect hedging language that indicates uncertainty.
        Returns penalty (0.0 to 0.3)
        """
        hedging_phrases = [
            "might be", "could be", "possibly", "perhaps",
            "it seems", "it appears", "i think", "probably",
            "may have", "likely", "uncertain"
        ]
        
        answer_lower = answer_text.lower()
        hedge_count = sum(1 for phrase in hedging_phrases if phrase in answer_lower)
        
        return min(0.3, hedge_count * 0.1)
    
    def _confidence_to_level(self, confidence: float) -> str:
        """Convert numeric confidence to level."""
        if confidence >= 0.8:
            return "HIGH"
        elif confidence >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"


# ------------------------------------
# Standalone test
# ------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    config = {"agent": {"confidence_threshold": 0.7}, "verification": {"enabled": True}}
    verifier = Verifier(config)
    
    # SOTA test case with chunk_map
    chunk_map = {
        "paper.pdf#1": {"content": "The Transformer relies entirely on self-attention to compute representations."},
        "paper.pdf#2": {"content": "Self-attention allows the model to weigh the importance of different input tokens."}
    }
    
    answer = {
        "concise_answer": "The Transformer uses self-attention mechanisms [paper.pdf#1]. This allows it to weigh token importance [paper.pdf#2].",
        "evidence": [
            {"claim": "The Transformer uses self-attention mechanisms", "cite_ids": ["paper.pdf#1"]},
            {"claim": "This allows it to weigh token importance", "cite_ids": ["paper.pdf#2"]}
        ],
        "sources_used": ["paper.pdf#1", "paper.pdf#2"],
        "confidence": {"score": 0.85, "level": "HIGH"}
    }
    
    chunks = [
        {"chunk_id": "1", "content": "The Transformer relies entirely on self-attention to compute representations."},
        {"chunk_id": "2", "content": "Self-attention allows the model to weigh the importance of different input tokens."}
    ]
    
    result = verifier.verify("What is the Transformer?", answer, chunks, chunk_map=chunk_map)
    print("SOTA Verification result:")
    for k, v in result.items():
        print(f"  {k}: {v}")