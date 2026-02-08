"""RAG-AS evaluator

Core evaluator for Retrieval-Augmented Generation Answerability & Support (RAGAS)
Provides retrieval metrics, generation metrics, and grounding checks using both
custom NLI-based evaluation and the official RAGAS framework.
"""
import json
import logging
import math
import os
import re
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Optional imports for custom metrics
try:
    from transformers import pipeline
except Exception:  # pragma: no cover - optional
    pipeline = None

try:
    from rouge_score import rouge_scorer
except Exception:  # pragma: no cover - optional
    rouge_scorer = None

try:
    from bert_score import score as bert_score
except Exception:  # pragma: no cover - optional
    bert_score = None

try:
    import nltk
    _ = nltk.data.find("tokenizers/punkt")
except Exception:  # pragma: no cover - optional
    nltk = None

# BLEU score imports
BLEU_AVAILABLE = False
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    BLEU_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    sentence_bleu = None
    SmoothingFunction = None

# RAGAS Framework Imports
RAGAS_AVAILABLE = False
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
        answer_similarity,
    )
    from ragas import EvaluationDataset, SingleTurnSample
    from datasets import Dataset
    RAGAS_AVAILABLE = True
    logger.info("RAGAS framework loaded successfully")
except ImportError as e:
    logger.warning(f"RAGAS framework not available: {e}. Using custom metrics only.")
except Exception as e:
    logger.warning(f"RAGAS framework import error: {e}. Using custom metrics only.")


def _recall_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int) -> float:
    topk = set(retrieved_ids[:k])
    if not gold_ids:
        return 0.0
    hit = any(g in topk for g in gold_ids)
    return 1.0 if hit else 0.0


def _mrr(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in gold_ids:
            return 1.0 / i
    return 0.0


class RAGASEvaluator:
    """
    Unified RAGAS Evaluator that supports:
    1. Official RAGAS framework metrics (faithfulness, answer_relevancy, context_precision, etc.)
    2. Custom NLI-based grounding checks
    3. Standard retrieval metrics (Recall@K, MRR)
    4. Standard generation metrics (ROUGE, BERTScore)
    """
    
    def __init__(
        self,
        nli_model: str = "roberta-large-mnli",
        device: Optional[int] = None,
        entailment_threshold: float = 0.7,
        use_ragas_framework: bool = True,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """Initialize evaluator.

        Args:
            nli_model: model name for NLI (transformers) - for custom grounding checks
            device: device index for transformers pipeline (None => CPU)
            entailment_threshold: threshold to consider a sentence as supporting
            use_ragas_framework: whether to use official RAGAS metrics (requires API key)
            llm_model: LLM model for RAGAS evaluation (defaults to gpt-3.5-turbo if API available)
        """
        # 0.9 is often too strict for MNLI probabilities on real-world text;
        # default to a more usable threshold, but allow overrides.
        self.entailment_threshold = float(os.getenv("ENTAILMENT_THRESHOLD", str(entailment_threshold)))
        self.use_ragas_framework = use_ragas_framework and RAGAS_AVAILABLE
        self.llm_model = llm_model
        self.embedding_model = embedding_model

        self._ragas_llm = None
        self._ragas_embeddings = None

        # Initialize NLI pipeline for custom grounding
        if pipeline is not None:
            try:
                self.nli = pipeline("text-classification", model=nli_model, device=device, top_k=1)
                logger.info(f"NLI pipeline loaded with model: {nli_model}")
            except Exception:
                logger.exception("Failed to load NLI pipeline; falling back to None")
                self.nli = None
        else:
            self.nli = None

        # Initialize ROUGE scorer
        if rouge_scorer is not None:
            self._rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        else:
            self._rouge = None

        # Initialize BLEU smoothing function
        if BLEU_AVAILABLE and SmoothingFunction is not None:
            self._bleu_smoothing = SmoothingFunction().method1
        else:
            self._bleu_smoothing = None

        # Check RAGAS availability
        if self.use_ragas_framework:
            logger.info("RAGAS framework enabled for evaluation")

            # Build LLM + embeddings for RAGAS metrics.
            # RAGAS metrics like faithfulness / answer relevancy require an LLM,
            # and several metrics require embeddings.
            self._ragas_llm = self._build_ragas_llm()
            self._ragas_embeddings = self._build_ragas_embeddings()

            if self._ragas_llm is None or self._ragas_embeddings is None:
                logger.warning(
                    "RAGAS framework enabled, but missing LLM/embeddings; disabling official RAGAS metrics. "
                    "(Will still compute custom retrieval/generation/grounding metrics.)"
                )
                self.use_ragas_framework = False
        else:
            logger.info("Using custom metrics only (RAGAS framework disabled or unavailable)")

    def _extract_label_scores(self, out: Any) -> List[Dict[str, Any]]:
        """Normalize HF pipeline outputs into a list of {label, score} dicts."""
        if out is None:
            return []
        if isinstance(out, dict):
            return [out]
        if isinstance(out, list):
            # Some pipeline calls return [[{label,score},...]] for single input
            if out and isinstance(out[0], list):
                return [d for d in out[0] if isinstance(d, dict)]
            return [d for d in out if isinstance(d, dict)]
        return []

    def _entailment_score(self, out: Any) -> Optional[float]:
        """Return entailment probability from NLI pipeline output, if available."""
        label_scores = self._extract_label_scores(out)
        if not label_scores:
            return None

        # First try explicit labels
        for d in label_scores:
            label = d.get("label")
            if not isinstance(label, str):
                continue
            label_u = label.upper()
            if label_u in {"ENTAILMENT", "ENTAILED", "SUPPORTS"}:
                try:
                    return float(d.get("score", 0.0))
                except Exception:
                    return None

        # Common fallback: roberta-large-mnli sometimes exposes LABEL_0/1/2
        # where LABEL_2 is entailment.
        for d in label_scores:
            label = d.get("label")
            if isinstance(label, str) and label.upper() == "LABEL_2":
                try:
                    return float(d.get("score", 0.0))
                except Exception:
                    return None

        # As a last resort, try to map via model config if accessible
        try:
            id2label = getattr(getattr(self.nli, "model", None), "config", None)
            id2label = getattr(id2label, "id2label", None)
            if isinstance(id2label, dict):
                entail_ids = [i for i, name in id2label.items() if str(name).upper() in {"ENTAILMENT", "ENTAILED", "SUPPORTS"}]
                if entail_ids:
                    target = f"LABEL_{entail_ids[0]}".upper()
                    for d in label_scores:
                        label = d.get("label")
                        if isinstance(label, str) and label.upper() == target:
                            return float(d.get("score", 0.0))
        except Exception:
            return None

        return None

    def _build_ragas_llm(self):
        """Create a LangChain-compatible LLM instance for RAGAS evaluation.

        Prefer local Ollama to avoid external API keys/rate limits.
        """
        model_name = self.llm_model or os.getenv("RAGAS_LLM_MODEL") or "llama3:8b"

        # Prefer local Ollama
        try:
            from langchain_community.chat_models import ChatOllama

            return ChatOllama(model=model_name, temperature=0)
        except Exception:
            pass

        # Fallback to OpenAI if API key exists
        if os.getenv("OPENAI_API_KEY"):
            try:
                from langchain_openai import ChatOpenAI

                return ChatOpenAI(model=os.getenv("RAGAS_OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
            except Exception:
                pass

        return None

    def _build_ragas_embeddings(self):
        """Create LangChain-compatible embeddings for RAGAS evaluation."""
        model_name = (
            self.embedding_model
            or os.getenv("RAGAS_EMBEDDING_MODEL")
            or "sentence-transformers/all-MiniLM-L6-v2"
        )
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(model_name=model_name)
        except Exception:
            # Older import paths
            try:
                from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

                return HuggingFaceEmbeddings(model_name=model_name)
            except Exception:
                return None

    # ====================================================================================
    # OFFICIAL RAGAS FRAMEWORK METHODS
    # ====================================================================================
    
    def evaluate_with_ragas(
        self,
        queries: List[str],
        generated_answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
        metrics: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate using the official RAGAS framework.
        
        Args:
            queries: List of user queries
            generated_answers: List of generated answers
            contexts: List of retrieved contexts (each is a list of context strings)
            ground_truths: List of ground truth answers
            metrics: List of RAGAS metrics to compute (defaults to all)
            
        Returns:
            Dictionary with RAGAS scores
        """
        if not RAGAS_AVAILABLE:
            logger.warning("RAGAS framework not available. Returning empty results.")
            return {"error": "RAGAS framework not installed"}
        
        # Default metrics
        if metrics is None:
            metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness,
                answer_similarity,
            ]
        
        try:
            # Create RAGAS dataset
            samples = []
            for i in range(len(queries)):
                sample = SingleTurnSample(
                    user_input=queries[i],
                    response=generated_answers[i],
                    retrieved_contexts=contexts[i] if i < len(contexts) else [],
                    reference=ground_truths[i] if i < len(ground_truths) else "",
                )
                samples.append(sample)
            
            eval_dataset = EvaluationDataset(samples=samples)
            
            # Run RAGAS evaluation
            results = ragas_evaluate(
                dataset=eval_dataset,
                metrics=metrics,
                llm=self._ragas_llm,
                embeddings=self._ragas_embeddings,
            )
            
            # Convert to dict
            scores = results.to_pandas().to_dict('records') if hasattr(results, 'to_pandas') else {}
            
            # Compute aggregates
            aggregate = {}
            if isinstance(scores, list) and scores:
                for key in scores[0].keys():
                    if isinstance(scores[0][key], (int, float)):
                        values = [s[key] for s in scores if isinstance(s.get(key), (int, float))]
                        aggregate[f"avg_{key}"] = sum(values) / len(values) if values else 0.0
            
            return {
                "per_sample": scores,
                "aggregate": aggregate,
                "success": True
            }
            
        except Exception as e:
            logger.exception(f"RAGAS evaluation failed: {e}")
            return {"error": str(e), "success": False}

    def evaluate_single_with_ragas(
        self,
        query: str,
        generated: str,
        contexts: List[str],
        ground_truth: str,
    ) -> Dict[str, float]:
        """
        Evaluate a single query using RAGAS framework.
        
        Returns dict with metric scores.
        """
        result = self.evaluate_with_ragas(
            queries=[query],
            generated_answers=[generated],
            contexts=[contexts],
            ground_truths=[ground_truth],
        )
        
        if result.get("success") and result.get("per_sample"):
            return result["per_sample"][0]
        return {}

    # ====================================================================================
    # CUSTOM RETRIEVAL METRICS
    # ====================================================================================
    
    def evaluate_retrieval(
        self,
        retrieved: List[Dict[str, Any]],
        gold_ids: List[str],
        ks: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """Compute recall@k and MRR for one query.

        retrieved: list of {'doc_id': ..., 'score': ..., 'excerpt': ...}
        gold_ids: list of gold doc ids
        """
        ids = [r.get("doc_id", r.get("chunk_id", "")) for r in retrieved]
        out = {}
        for k in ks:
            out[f"recall@{k}"] = _recall_at_k(ids, gold_ids, k)
        out["mrr"] = _mrr(ids, gold_ids)
        return out

    # ====================================================================================
    # CUSTOM GENERATION METRICS
    # ====================================================================================
    
    def evaluate_generation(self, generated: str, gold: str) -> Dict[str, float]:
        """Compute BLEU, ROUGE, and BERTScore for generated vs gold answer."""
        out: Dict[str, float] = {}
        
        # ===== BLEU Scores =====
        out.update(self._compute_bleu(gold, generated))
        
        # ===== ROUGE Scores =====
        out.update(self._compute_rouge(gold, generated))
        
        # ===== BERTScore =====
        out.update(self._compute_bertscore(gold, generated))

        return out

    def _compute_bleu(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Compute BLEU scores (1-4 grams) with smoothing."""
        default_scores = {
            "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0, "bleu_avg": 0.0
        }
        
        if not BLEU_AVAILABLE or sentence_bleu is None:
            return default_scores
        
        if not reference or not hypothesis:
            return default_scores
        
        try:
            ref_tokens = reference.lower().split()
            hyp_tokens = hypothesis.lower().split()
            
            if len(hyp_tokens) < 1 or len(ref_tokens) < 1:
                return default_scores
            
            # Compute individual n-gram BLEU scores with smoothing
            bleu1 = sentence_bleu(
                [ref_tokens], hyp_tokens, 
                weights=(1, 0, 0, 0), 
                smoothing_function=self._bleu_smoothing
            )
            bleu2 = sentence_bleu(
                [ref_tokens], hyp_tokens, 
                weights=(0.5, 0.5, 0, 0), 
                smoothing_function=self._bleu_smoothing
            )
            bleu3 = sentence_bleu(
                [ref_tokens], hyp_tokens, 
                weights=(0.33, 0.33, 0.34, 0), 
                smoothing_function=self._bleu_smoothing
            )
            bleu4 = sentence_bleu(
                [ref_tokens], hyp_tokens, 
                weights=(0.25, 0.25, 0.25, 0.25), 
                smoothing_function=self._bleu_smoothing
            )
            
            return {
                "bleu1": round(bleu1, 4),
                "bleu2": round(bleu2, 4),
                "bleu3": round(bleu3, 4),
                "bleu4": round(bleu4, 4),
                "bleu_avg": round((bleu1 + bleu2 + bleu3 + bleu4) / 4, 4)
            }
        except Exception as e:
            logger.debug(f"BLEU computation failed: {e}")
            return default_scores

    def _compute_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Compute ROUGE scores (1, 2, L)."""
        default_scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        if self._rouge is None or not reference or not hypothesis:
            return default_scores
        
        try:
            sc = self._rouge.score(reference, hypothesis)
            return {
                "rouge1": round(sc["rouge1"].fmeasure, 4),
                "rouge2": round(sc["rouge2"].fmeasure, 4),
                "rougeL": round(sc["rougeL"].fmeasure, 4)
            }
        except Exception as e:
            logger.debug(f"ROUGE computation failed: {e}")
            return default_scores

    def _compute_bertscore(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Compute BERTScore (precision, recall, F1)."""
        default_scores = {
            "bertscore_precision": 0.0, 
            "bertscore_recall": 0.0, 
            "bertscore_f1": 0.0
        }
        
        if bert_score is None or os.getenv("BERTSCORE_DISABLED", "0") == "1":
            return default_scores
        
        if not reference or not hypothesis:
            return default_scores
        
        try:
            model_type = os.getenv("BERTSCORE_MODEL", "distilbert-base-uncased")
            device = os.getenv("BERTSCORE_DEVICE", "cpu")
            
            # NOTE: rescale_with_baseline=False to avoid negative scores
            P, R, F = bert_score(
                [hypothesis],
                [reference],
                lang="en",
                rescale_with_baseline=False,  # Changed: avoid negative scores
                model_type=model_type,
                device=device,
                verbose=False,
            )
            return {
                "bertscore_precision": round(float(P.mean()), 4),
                "bertscore_recall": round(float(R.mean()), 4),
                "bertscore_f1": round(float(F.mean()), 4)
            }
        except Exception as e:
            logger.debug(f"BERTScore computation failed: {e}")
            # Try fallback with lighter model
            try:
                P, R, F = bert_score(
                    [hypothesis],
                    [reference],
                    lang="en",
                    rescale_with_baseline=False,
                    model_type="distilbert-base-uncased",
                    device="cpu",
                    verbose=False,
                )
                return {
                    "bertscore_precision": round(float(P.mean()), 4),
                    "bertscore_recall": round(float(R.mean()), 4),
                    "bertscore_f1": round(float(F.mean()), 4)
                }
            except Exception:
                return default_scores

    # ====================================================================================
    # CUSTOM NLI-BASED GROUNDING (Faithfulness)
    # ====================================================================================
    def evidence_support_score(
        self,
        claim: str,
        docs: List[Dict[str, Any]],
        max_sentences: int = 5
    ) -> Dict[str, Any]:
        """Compute grounding support for an answer/claim against provided docs.

        Important: checking the full multi-sentence answer as a single NLI hypothesis
        often yields near-zero support. For multi-sentence answers we compute per-claim
        support and use that ratio as the grounding score.
        """

        # Multi-sentence answer: use per-claim grounding ratio.
        try:
            claims = self.extract_claims(claim)
        except Exception:
            claims = [claim] if claim else []

        if len(claims) > 1:
            cg = self.evidence_support_per_claim(claim, docs)
            return {
                "support_ratio": float(cg.get("claim_support_ratio", 0.0)),
                "mode": "per_claim",
                "per_claim": cg.get("per_claim", []),
            }

        results = []
        supported = 0
        for doc in docs:
            text = doc.get("excerpt") or doc.get("text") or doc.get("content") or ""
            # sentence-splitting fallback
            sents = [text]
            if nltk is not None:
                try:
                    sents = nltk.tokenize.sent_tokenize(text)[:max_sentences]
                except Exception:
                    sents = [text]

            max_score = 0.0
            doc_supported = False
            for s in sents:
                if not self.nli:
                    # cannot compute; skip
                    continue
                try:
                    # NLI text-classification pipeline expects concatenated premise + hypothesis
                    # Format: "premise </s></s> hypothesis" for roberta-mnli or similar
                    nli_input = f"{s} </s></s> {claim}"
                    out = self.nli(nli_input, top_k=None)
                    ent = self._entailment_score(out)
                    if ent is None:
                        continue
                    max_score = max(max_score, float(ent))
                    if float(ent) >= self.entailment_threshold:
                        doc_supported = True
                except Exception as e:
                    logger.debug(f"NLI step failed for one sentence: {e}", exc_info=True)

            if doc_supported:
                supported += 1
            results.append({
                "doc_id": doc.get("doc_id", doc.get("chunk_id", "")),
                "supported": doc_supported,
                "max_score": max_score
            })

        support_ratio = supported / len(docs) if docs else 0.0
        return {"support_ratio": support_ratio, "per_doc": results}

    # ====================================================================================
    # CLAIM EXTRACTION AND PER-CLAIM GROUNDING
    # ====================================================================================
    
    def extract_claims(self, text: str, min_len: int = 2) -> List[str]:
        """Extract candidate claims from generated text. Simple sentence-based splitter with length filter."""
        if not text:
            return []

        # Remove citation markers and normalize whitespace to improve NLI matching.
        text = re.sub(r"\[[^\]]+\]", "", text)
        text = text.replace("ANSWER:", " ").replace("Answer:", " ")
        text = re.sub(r"\s+", " ", text).strip()

        sents = [text]
        if nltk is not None:
            try:
                sents = nltk.tokenize.sent_tokenize(text)
            except Exception:
                sents = [text]
        # Basic cleaning: strip and remove very short sentences
        claims = [s.strip() for s in sents if len(s.strip().split()) >= min_len]
        return claims

    def evidence_support_per_claim(
        self,
        generated: str,
        docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess support for each extracted claim across provided docs.

        Returns:
            {"claim_support_ratio": float, "per_claim": [{"claim":str, "supported":bool, "per_doc":[...]}]}
        """
        claims = self.extract_claims(generated)
        per_claim_results = []
        supported_count = 0
        for claim in claims:
            # For each claim, check docs (use evidence_support_score internally but treat claim as hypothesis)
            doc_res = []
            claim_supported = False
            for doc in docs:
                # reuse same sentence-level logic but we want per-doc supported flag and max score
                text = doc.get("excerpt") or doc.get("text") or doc.get("content") or ""
                sents = [text]
                if nltk is not None:
                    try:
                        sents = nltk.tokenize.sent_tokenize(text)[:5]
                    except Exception:
                        sents = [text]
                max_score = 0.0
                doc_supported = False
                for s in sents:
                    if not self.nli:
                        continue
                    try:
                        # NLI text-classification pipeline expects concatenated premise + hypothesis
                        nli_input = f"{s} </s></s> {claim}"
                        out = self.nli(nli_input, top_k=None)
                        ent = self._entailment_score(out)
                        if ent is None:
                            continue
                        max_score = max(max_score, float(ent))
                        if float(ent) >= self.entailment_threshold:
                            doc_supported = True
                    except Exception as e:
                        logger.debug(f"NLI per-claim step failed: {e}", exc_info=True)
                if doc_supported:
                    claim_supported = True
                doc_res.append({
                    "doc_id": doc.get("doc_id", doc.get("chunk_id", "")),
                    "supported": doc_supported,
                    "max_score": max_score
                })

            if claim_supported:
                supported_count += 1

            per_claim_results.append({
                "claim": claim,
                "supported": claim_supported,
                "per_doc": doc_res
            })

        claim_support_ratio = supported_count / len(per_claim_results) if per_claim_results else 0.0
        return {"claim_support_ratio": claim_support_ratio, "per_claim": per_claim_results}

    # ====================================================================================
    # FULL QUERY EVALUATION (Combined)
    # ====================================================================================
    
    def evaluate_query(
        self,
        query: str,
        generated: str,
        retrieved: List[Dict[str, Any]],
        gold: Dict[str, Any],
        use_ragas: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate a single query and return per-query structured metrics/result.

        gold: {'answers':[...], 'evidence_ids':[...]} where answers is list of gold answer strings
        retrieved: list of {'doc_id','score','excerpt'}
        use_ragas: whether to include official RAGAS metrics (requires API key)
        """
        gold_answers = gold.get("answers", [])
        gold_evidence = gold.get("evidence_ids", [])

        # Retrieval metrics
        retrieval_metrics = self.evaluate_retrieval(retrieved, gold_evidence)

        # Generation metrics - choose best gold as reference (first)
        ref = gold_answers[0] if gold_answers else ""
        gen_metrics = self.evaluate_generation(generated, ref)

        # Grounding: prefer checking grounding over gold evidence docs when available
        if gold_evidence:
            docs_to_check = [r for r in retrieved if r.get("doc_id", r.get("chunk_id", "")) in gold_evidence]
            if not docs_to_check:
                docs_to_check = retrieved
        else:
            docs_to_check = retrieved

        grounding = self.evidence_support_score(generated, docs_to_check)

        # Claim-level grounding decomposition
        claim_grounding = self.evidence_support_per_claim(generated, docs_to_check)

        # Improved correctness calculation using multiple signals
        correctness = self._compute_correctness(generated, ref, gen_metrics, grounding, claim_grounding)

        result = {
            "query": query,
            "generated": generated,
            "reference": ref,
            "retrieval": retrieval_metrics,
            "generation": gen_metrics,
            "grounding": grounding,
            "claim_grounding": claim_grounding,
            "correct": correctness,
        }

        # Optionally add official RAGAS metrics
        if use_ragas and self.use_ragas_framework:
            contexts = [r.get("excerpt") or r.get("text") or r.get("content", "") for r in retrieved[:5]]
            ragas_result = self.evaluate_single_with_ragas(query, generated, contexts, ref)
            result["ragas_metrics"] = ragas_result

            # Prefer RAGAS answer_correctness to set correctness when available (but not NaN)
            ac = ragas_result.get("answer_correctness") if isinstance(ragas_result, dict) else None
            if isinstance(ac, (int, float)) and not (isinstance(ac, float) and math.isnan(ac)):
                # RAGAS answer_correctness: use threshold 0.4
                correctness = bool(float(ac) >= 0.4)
                result["correct"] = correctness

        return result

    def _compute_correctness(
        self,
        generated: str,
        reference: str,
        gen_metrics: Dict[str, float],
        grounding: Dict[str, Any],
        claim_grounding: Dict[str, Any],
    ) -> bool:
        """
        Compute correctness using multiple signals:
        1. Exact match (case-insensitive)
        2. High semantic similarity (BERTScore F1 > 0.7)
        3. Good ROUGE-L overlap (> 0.5)
        4. Answer is grounded in evidence (claim_support_ratio > 0.5)
        """
        if not reference:
            # No reference to compare - check grounding instead
            grounding_ratio = claim_grounding.get("claim_support_ratio", 0.0)
            return grounding_ratio >= 0.5
        
        # 1. Exact match (normalized)
        gen_norm = generated.strip().lower()
        ref_norm = reference.strip().lower()
        if gen_norm == ref_norm:
            return True
        
        # 2. Check if reference is contained in generated (common for RAG)
        if ref_norm in gen_norm:
            return True
        
        # 3. Semantic similarity via BERTScore
        bertscore_f1 = gen_metrics.get("bertscore_f1", 0.0)
        if bertscore_f1 >= 0.7:
            return True
        
        # 4. Lexical overlap via ROUGE-L
        rouge_l = gen_metrics.get("rougeL", 0.0)
        if rouge_l >= 0.5:
            return True
        
        # 5. Weighted combination: if multiple signals are moderately good
        score = 0.0
        if bertscore_f1 >= 0.5:
            score += 0.3
        if rouge_l >= 0.3:
            score += 0.3
        if claim_grounding.get("claim_support_ratio", 0.0) >= 0.5:
            score += 0.4
        
        return score >= 0.5

    # ====================================================================================
    # BATCH EVALUATION
    # ====================================================================================
    
    def evaluate_batch(
        self,
        data: List[Dict[str, Any]],
        use_ragas: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Evaluate a batch of queries.
        
        Args:
            data: List of dicts with keys: query, generated, retrieved, gold
            use_ragas: whether to include official RAGAS metrics
            
        Returns:
            Tuple of (per_query_results, aggregate_metrics)
        """
        results = []
        for item in data:
            result = self.evaluate_query(
                query=item.get("query", ""),
                generated=item.get("generated", ""),
                retrieved=item.get("retrieved", []),
                gold=item.get("gold", {}),
                use_ragas=use_ragas,
            )
            results.append(result)
        
        aggregate = aggregate_results(results)
        return results, aggregate


# ====================================================================================
# HELPER FUNCTIONS
# ====================================================================================

def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate per-query results into summary metrics."""
    agg = {
        # Retrieval metrics
        "recall@1": 0.0,
        "recall@5": 0.0,
        "recall@10": 0.0,
        "mrr": 0.0,
        # BLEU metrics
        "bleu1": 0.0,
        "bleu2": 0.0,
        "bleu3": 0.0,
        "bleu4": 0.0,
        "bleu_avg": 0.0,
        # ROUGE metrics
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0,
        # BERTScore metrics
        "bertscore_precision": 0.0,
        "bertscore_recall": 0.0,
        "bertscore_f1": 0.0,
        # Grounding metrics
        "grounding_ratio": 0.0,
        "claim_support_ratio": 0.0,
        # Accuracy
        "accuracy": 0.0,
        # Official RAGAS metrics
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "answer_correctness": 0.0,
        "answer_similarity": 0.0,
    }
    n = len(results)
    if n == 0:
        return {k: 0.0 for k in agg}

    ragas_counts = {
        "faithfulness": 0,
        "answer_relevancy": 0,
        "context_precision": 0,
        "context_recall": 0,
        "answer_correctness": 0,
        "answer_similarity": 0,
    }

    for r in results:
        # Retrieval metrics
        agg["recall@1"] += r.get("retrieval", {}).get("recall@1", 0.0)
        agg["recall@5"] += r.get("retrieval", {}).get("recall@5", 0.0)
        agg["recall@10"] += r.get("retrieval", {}).get("recall@10", 0.0)
        agg["mrr"] += r.get("retrieval", {}).get("mrr", 0.0)
        
        # Generation metrics (BLEU, ROUGE, BERTScore)
        gen = r.get("generation", {})
        agg["bleu1"] += gen.get("bleu1", 0.0)
        agg["bleu2"] += gen.get("bleu2", 0.0)
        agg["bleu3"] += gen.get("bleu3", 0.0)
        agg["bleu4"] += gen.get("bleu4", 0.0)
        agg["bleu_avg"] += gen.get("bleu_avg", 0.0)
        agg["rouge1"] += gen.get("rouge1", 0.0)
        agg["rouge2"] += gen.get("rouge2", 0.0)
        agg["rougeL"] += gen.get("rougeL", 0.0)
        agg["bertscore_precision"] += gen.get("bertscore_precision", 0.0)
        agg["bertscore_recall"] += gen.get("bertscore_recall", 0.0)
        agg["bertscore_f1"] += gen.get("bertscore_f1", 0.0)
        
        # Grounding metrics
        agg["grounding_ratio"] += r.get("grounding", {}).get("support_ratio", 0.0)
        agg["claim_support_ratio"] += r.get("claim_grounding", {}).get("claim_support_ratio", 0.0)
        
        # Accuracy
        agg["accuracy"] += 1.0 if r.get("correct") else 0.0

        # Official RAGAS metrics
        ragas_metrics = r.get("ragas_metrics") or r.get("ragas", {})
        if isinstance(ragas_metrics, dict):
            for key in ragas_counts.keys():
                val = ragas_metrics.get(key)
                if isinstance(val, (int, float)):
                    agg[key] += float(val)
                    ragas_counts[key] += 1

    # Average core metrics over all samples
    core_metrics = [
        "recall@1", "recall@5", "recall@10", "mrr",
        "bleu1", "bleu2", "bleu3", "bleu4", "bleu_avg",
        "rouge1", "rouge2", "rougeL",
        "bertscore_precision", "bertscore_recall", "bertscore_f1",
        "grounding_ratio", "claim_support_ratio", "accuracy",
    ]
    for k in core_metrics:
        agg[k] = round(agg[k] / n, 4)

    # Average RAGAS metrics over available samples (may be < n if disabled/errors)
    for k, c in ragas_counts.items():
        agg[k] = round((agg[k] / c) if c else 0.0, 4)

    return agg


def run_ragas_evaluation(
    input_file: str,
    output_file: str,
    use_ragas_framework: bool = False,
) -> Dict[str, Any]:
    """
    Run RAGAS evaluation on a JSONL file.
    
    Args:
        input_file: Path to input JSONL with query, generated, retrieved, gold fields
        output_file: Path to save results
        use_ragas_framework: Whether to use official RAGAS metrics
        
    Returns:
        Aggregate metrics
    """
    import json
    
    # Load data
    data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    logger.info(f"Loaded {len(data)} samples for evaluation")
    
    # Initialize evaluator
    evaluator = RAGASEvaluator(use_ragas_framework=use_ragas_framework)
    
    # Run evaluation
    results, aggregate = evaluator.evaluate_batch(data, use_ragas=use_ragas_framework)
    
    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    
    # Save summary
    summary_file = output_file.replace(".jsonl", "_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    logger.info(f"Summary saved to {summary_file}")
    
    return aggregate


# ====================================================================================
# CLI ENTRY POINT
# ====================================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAGAS Evaluation")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--ragas", action="store_true", help="Use official RAGAS framework")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    aggregate = run_ragas_evaluation(
        input_file=args.input,
        output_file=args.output,
        use_ragas_framework=args.ragas,
    )
    
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION SUMMARY")
    print("=" * 60)
    for k, v in aggregate.items():
        print(f"  {k}: {v:.4f}")
    print("=" * 60)
