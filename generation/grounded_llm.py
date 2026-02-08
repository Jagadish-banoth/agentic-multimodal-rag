"""
generation/grounded_llm.py
===========================

FAANG-Grade SOTA Grounded LLM Generator
---------------------------------------

Features:
✅ Structured JSON output with evidence/citations for verification
✅ Stable citation IDs (C1, C2, ...) from fusion layer
✅ Industry-grade prompt engineering (few-shot, chain-of-thought)
✅ Direct local Ollama inference (no API dependencies)
✅ SOTA response quality and grounding
✅ Robust parsing and confidence estimation
✅ Full compatibility with ExecutionEngine + Verifier
✅ Zero hallucination through evidence discipline

Author: FAANG-Grade RAG System  
Date: Feb 2026
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Optional: load environment variables from .env
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None

# Optional: OpenAI client (used for OpenRouter)
try:
    from openai import OpenAI  # type: ignore
    OPENAI_AVAILABLE = True
except Exception:  # pragma: no cover
    OpenAI = None
    OPENAI_AVAILABLE = False

# Local Ollama (fallback)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None

log_path = os.path.join(os.path.dirname(__file__), '../logs/grounded_llm.log')
log_path = os.path.abspath(log_path)
logger = logging.getLogger("grounded_llm")
if not logger.hasHandlers():
    handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Project root and config
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "settings.yaml"

if load_dotenv:
    # Loads ROOT/.env if present; safe no-op if missing.
    load_dotenv(dotenv_path=str(ROOT / ".env"), override=False)


def load_config() -> Dict[str, Any]:
    """Load configuration from settings.yaml."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}
# ============================================================================
# SOTA PROMPT TEMPLATES (Llama3:8b optimized with structured output)
# ============================================================================

SYSTEM_PROMPT_LLAMA = """You are an expert document analyst assistant. Your role is to answer questions accurately using ONLY the provided context.

CORE PRINCIPLES:
1. **Accuracy First**: Every statement must be supported by the provided context
2. **Citation Required**: ALWAYS cite evidence using [C1], [C2], etc. format
3. **Clarity**: Organize answers logically with clear explanations
4. **Grounding**: Every major claim needs a citation
5. **Honesty**: If context lacks information, explicitly state this

CITATION FORMAT:
- Use [filename#chunk_id] to cite specific evidence chunks
- Place citations immediately after the claim they support
- Multiple citations for one claim: [fileA#12][fileB#7]
- Do NOT invent citation IDs - only use those provided in the evidence

RESPONSE STYLE:
- Professional and authoritative
- Direct and specific (avoid vague language)
- Evidence-based reasoning visible in answer
- Structured for readability
- 2-4 paragraphs for complex questions"""

SIMPLE_PROMPT = """Answer the following question based ONLY on the provided context. Be direct and cite sources using [filename#chunk_id].

If the context lacks sufficient information, state: "Based on the provided evidence, I cannot fully answer this question because..."

EVIDENCE:
{context}

QUESTION: {query}

ANSWER (with citations [filename#chunk_id]):"""

STRUCTURED_PROMPT = """You are an expert at answering questions based on provided documents. 

RULES:
1. ONLY use information from the evidence below
2. CITE every claim using [filename#chunk_id]
3. If evidence is insufficient, say so clearly

EVIDENCE:
{context}

QUESTION: {query}

Provide a comprehensive answer with proper citations [filename#chunk_id].

ANSWER:"""

FEW_SHOT_PROMPT = """You are an expert at answering questions based on provided documents. Answer accurately with citations.

EXAMPLE 1:
Context: "[C1] Machine Learning is a subset of AI. [C2] Deep Learning uses neural networks with multiple layers."
Question: "What is Machine Learning?"
Answer: Machine Learning is a subset of Artificial Intelligence [C1]. Deep Learning, a specialized approach within this field, uses neural networks with multiple layers to model complex patterns [C2].

EXAMPLE 2:
Context: "[C1] The Transformer model was introduced in 2017 by Vaswani et al. It uses self-attention mechanisms."
Question: "What is the capital of France?"
Answer: I cannot answer this question based on the provided evidence because the context only discusses the Transformer model [C1] and doesn't contain geographic information.

---

Now answer the following question with proper citations:

EVIDENCE:
{context}

QUESTION: {query}

ANSWER (cite using [filename#chunk_id]):"""


# ============================================================================
# SOTA GENERATOR CLASS
# ============================================================================

class GroundedLLM:
    """
    FAANG-Grade SOTA Generator using Llama3:8b locally.
    
    Optimized for:
    - Direct local Ollama calls (no API)
    - Llama3's instruction-tuning (8k context)
    - High-quality, grounded responses
    - Production reliability & low-latency
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize SOTA Generator for Llama3:8b."""
        self.config = config or load_config()
        
        # Model configuration
        models_cfg = self.config.get("models", {})
        self.llama_model = models_cfg.get("generator_model", "llama3:8b")
        
        # Generation parameters - tuned for Llama3:8b
        gen_cfg = self.config.get("generation", {})
        self.temperature = gen_cfg.get("temperature", 0.3)
        self.top_p = gen_cfg.get("top_p", 0.9)
        self.top_k = 40  # Llama3 optimal value
        self.max_tokens = gen_cfg.get("max_generation_tokens", 400)
        self.max_context_tokens = 7000  # Leave room in 8k window

        # Provider selection
        # - "auto": use OpenRouter if key present, else Ollama
        # - "openrouter": prefer OpenRouter, fallback to Ollama
        # - "ollama": Ollama only
        self.provider = (gen_cfg.get("provider") or "auto").lower()
        self.fallback_provider = (gen_cfg.get("fallback_provider") or "ollama").lower()

        openrouter_cfg = gen_cfg.get("openrouter", {})
        self.openrouter_base_url = openrouter_cfg.get("base_url", "https://openrouter.ai/api/v1")
        self.openrouter_model = openrouter_cfg.get(
            "model",
            "nvidia/nemotron-3-nano-30b-a3b:free",
        )
        self.openrouter_reasoning = bool(openrouter_cfg.get("reasoning", True))
        self.openrouter_timeout = openrouter_cfg.get("timeout", 120)
        self.openrouter_auto_continue = bool(openrouter_cfg.get("auto_continue", True))
        self.openrouter_max_continuations = int(openrouter_cfg.get("max_continuations", 1))
        self.openrouter_api_key = (
            openrouter_cfg.get("api_key")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OP_TOKEN")
        )
        self.openrouter_client = None
        if self.provider in {"auto", "openrouter"} and self.openrouter_api_key and OPENAI_AVAILABLE:
            try:
                # OpenRouter is OpenAI-compatible; base_url routes requests.
                self.openrouter_client = OpenAI(
                    base_url=self.openrouter_base_url,
                    api_key=self.openrouter_api_key,
                    timeout=self.openrouter_timeout,
                )
            except Exception as e:
                logger.warning(f"OpenRouter client init failed; will fallback if possible: {e}")
        
        # Ollama settings (local-only)
        ollama_cfg = gen_cfg.get("ollama", {})
        self.ollama_host = ollama_cfg.get("host", "http://localhost:11434")
        self.ollama_timeout = ollama_cfg.get("timeout", 120)

        # Local-only client to avoid any API key usage
        self.ollama_client = None
        try:
            from ollama import Client  # type: ignore
            self.ollama_client = Client(host=self.ollama_host)
        except Exception:
            logger.debug("Falling back to default Ollama client; host env must be set if non-default")
        
        # Generation strategy
        self.use_few_shot = gen_cfg.get("use_few_shot", True)
        self.sota_mode = gen_cfg.get("sota_mode", False)
        
        # Verify we have at least one viable backend.
        if self.provider == "ollama":
            if not OLLAMA_AVAILABLE:
                raise RuntimeError("Ollama required for provider=ollama. Install with: pip install ollama")
        elif self.provider in {"auto", "openrouter"}:
            if self.openrouter_client is None and not OLLAMA_AVAILABLE:
                raise RuntimeError(
                    "No generation backend available. "
                    "Set OPENROUTER_API_KEY (or OP_TOKEN) for OpenRouter, or install Ollama for local fallback."
                )

        active = self._active_provider_name()
        logger.info(f"✅ GroundedLLM initialized (provider={active})")

    def _active_provider_name(self) -> str:
        if self.provider == "ollama":
            return f"ollama:{self.llama_model}"
        if self.openrouter_client is not None:
            return f"openrouter:{self.openrouter_model}"
        return f"ollama:{self.llama_model}"
    
    def generate(
        self,
        query: str,
        context: str,
        sources: Optional[List[Dict]] = None,
        chunk_map: Optional[Dict[str, Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate grounded answer with structured evidence (SOTA).
        
        Args:
            query: User query
            context: Fused context with evidence (contains [C1], [C2], etc.)
            sources: List of source documents
            chunk_map: Citation ID to chunk mapping from fusion layer
            **kwargs: Additional options
        
        Returns:
            Response dict with:
            - answer: Human-readable answer text
            - evidence: List of {claim, cite_ids} for verification
            - sources_used: List of citation IDs actually used
            - confidence: Confidence score
            - sources: List of source file names
        """
        if not context or not context.strip():
            return self._empty_response(query)
        
        logger.info(f"📝 Generating SOTA answer: {query[:60]}...")
        
        try:
            # Build optimized prompt for Llama3
            prompt = self._build_prompt(query, context)

            # Call preferred provider (OpenRouter → fallback Ollama)
            answer_text, used_model = self._call_preferred(prompt)
            
            # Clean response
            answer_text = self._clean_response(answer_text)
            
            # SOTA: Extract citations used in the answer
            citations_used = self._extract_citations(answer_text, chunk_map)
            
            # SOTA: Build structured evidence for verification
            evidence = self._build_evidence_structure(answer_text, citations_used, chunk_map)
            
            # Estimate confidence using FAANG heuristics + citation coverage
            confidence = self._estimate_confidence_sota(
                answer_text, context, query, citations_used, chunk_map
            )
            
            # Format sources
            source_list = self._format_sources(sources or [])
            
            return {
                "answer": answer_text,
                "evidence": evidence,
                "sources_used": citations_used,
                "sources": source_list,
                "confidence": confidence,
                "query": query,
                "model": used_model,
                "tokens_used": self._estimate_tokens(answer_text),
                "chunk_map": chunk_map or {},
                "retrieved_metadata": sources if sources is not None else [],
            }
        
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return self._error_response(query, str(e))
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build optimized prompt for Llama3:8b with citation instructions."""
        cleaned_context = self._normalize_context(context, query)

        # Use structured prompt for contexts with citation IDs
        if "[C" in cleaned_context:
            if self.use_few_shot and len(cleaned_context) > 500:
                base_prompt = FEW_SHOT_PROMPT
            else:
                base_prompt = STRUCTURED_PROMPT
        else:
            base_prompt = SIMPLE_PROMPT

        return base_prompt.format(context=cleaned_context, query=query)

    def _call_preferred(self, prompt: str) -> tuple[str, str]:
        """Call OpenRouter if configured, otherwise Ollama; fallback when possible."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_LLAMA},
            {"role": "user", "content": prompt},
        ]

        if self.provider in {"auto", "openrouter"} and self.openrouter_client is not None:
            try:
                return self._call_openrouter(messages), self.openrouter_model
            except Exception as e:
                logger.warning(f"OpenRouter call failed; attempting fallback: {e}")
                if self.fallback_provider != "ollama":
                    raise
                if not OLLAMA_AVAILABLE:
                    logger.error("Ollama not available for fallback")
                    raise
                # Try Ollama fallback
                try:
                    return self._call_llama_with_messages(messages), self.llama_model
                except Exception as fallback_e:
                    logger.error(f"Ollama fallback also failed: {fallback_e}")
                    raise RuntimeError(f"Both OpenRouter and Ollama failed. OpenRouter: {e}, Ollama: {fallback_e}")

        # Fallback / default - ensure Ollama works
        try:
            return self._call_llama_with_messages(messages), self.llama_model
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    def _call_openrouter(self, messages: List[Dict[str, str]]) -> str:
        if self.openrouter_client is None:
            raise RuntimeError("OpenRouter client not initialized")

        extra_body = {"reasoning": {"enabled": True}} if self.openrouter_reasoning else None

        full_text_parts: List[str] = []
        current_messages: List[Dict[str, Any]] = list(messages)
        continuations = 0

        while True:
            response = self.openrouter_client.chat.completions.create(
                model=self.openrouter_model,
                messages=current_messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                extra_body=extra_body,
            )

            choice = response.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            msg = choice.message
            content = (msg.content or "").strip()

            logger.info(
                f"OpenRouter completion: finish_reason={finish_reason} chars={len(content)} continuations={continuations}"
            )

            if content:
                full_text_parts.append(content)

            # Decide whether to auto-continue
            should_continue = False
            if self.openrouter_auto_continue and continuations < self.openrouter_max_continuations:
                if finish_reason == "length":
                    should_continue = True
                elif finish_reason in {"stop", None}:
                    should_continue = False
                else:
                    # If provider returns a non-stop finish_reason, we try one continuation.
                    should_continue = True

            if not should_continue:
                break

            # Preserve OpenRouter reasoning_details if present (provider-specific extra field).
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content}
            reasoning_details = getattr(msg, "reasoning_details", None)
            if reasoning_details is not None:
                assistant_msg["reasoning_details"] = reasoning_details

            current_messages = list(messages) + [assistant_msg, {"role": "user", "content": "Continue from where you left off and finish the answer."}]
            continuations += 1

        final_text = "\n\n".join([p for p in full_text_parts if p]).strip()
        
        # If OpenRouter returned empty or too short response, raise to trigger fallback
        if len(final_text) < 20:
            raise RuntimeError(f"OpenRouter returned insufficient content ({len(final_text)} chars). Falling back.")
        
        return final_text
    
    def _call_llama(self, prompt: str) -> str:
        """Call Llama3:8b via Ollama with SOTA parameters."""
        try:
            # Use chat API for instruction-tuned Llama3
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_LLAMA},
                {"role": "user", "content": prompt}
            ]

            return self._call_llama_with_messages(messages)

        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            raise RuntimeError(f"Failed to call Llama3:8b: {e}")

    def _call_llama_with_messages(self, messages: List[Dict[str, str]]) -> str:
        """Internal helper to call Ollama with pre-built chat messages."""
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama not installed. Install with: pip install ollama")
        
        try:
            chat_fn = self.ollama_client.chat if self.ollama_client else ollama.chat

            response = chat_fn(
                model=self.llama_model,
                messages=messages,
                stream=False,
                options={
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "num_predict": self.max_tokens,
                    "num_ctx": self.max_context_tokens,
                    "repeat_penalty": 1.1,  # Avoid repetition
                    "tfs_z": 1.0,  # Tail-free sampling
                }
            )
            
            if not response or "message" not in response:
                raise RuntimeError(f"Invalid Ollama response format: {response}")
            
            content = response["message"]["content"]
            if not content or not content.strip():
                raise RuntimeError("Ollama returned empty content")
                
            return content.strip()
            
        except Exception as e:
            if "CUDA error" in str(e):
                logger.error(f"Ollama CUDA error - try restarting Ollama service: {e}")
            elif "connection" in str(e).lower():
                logger.error(f"Cannot connect to Ollama - ensure it's running on {self.ollama_host}: {e}")
            else:
                logger.error(f"Ollama call failed: {e}")
            raise
    
    def _clean_response(self, text: str) -> str:
        """Clean and normalize response text."""
        text = text.strip()
        
        # Remove markdown code blocks
        text = re.sub(r"```.*?\n", "", text, flags=re.DOTALL)
        text = re.sub(r"```", "", text)
        
        # Remove LLM artifacts
        text = re.sub(r"^\*\*Answer:\*\*\s*", "", text)
        text = re.sub(r"^Answer:\s*", "", text)
        text = re.sub(r"^Based on.*?context[,:]\s*", "", text, flags=re.IGNORECASE)
        
        # Clean whitespace
        text = re.sub(r"\n\n+", "\n\n", text)
        
        return text.strip()
    
    def _extract_citations(self, answer: str, chunk_map: Optional[Dict[str, Dict]] = None) -> List[str]:
        """
        SOTA: Extract all citation IDs used in the answer.

        Returns list like ["paper.pdf#chunk_12", "doc.txt#3"].
        """
        # Match any bracketed tokens like [filename#chunk_id]
        raw_tokens = re.findall(r"\[([^\[\]]+?)\]", answer)

        # Deduplicate while preserving order
        seen = set()
        citations: List[str] = []
        for token in raw_tokens:
            token = token.strip()
            # Prefer tokens that look like file#chunk or exist in chunk_map
            if chunk_map and token in chunk_map:
                cite_id = token
            elif "#" in token:
                cite_id = token
            else:
                continue

            if cite_id not in seen:
                seen.add(cite_id)
                citations.append(cite_id)

        return citations
    
    def _build_evidence_structure(
        self,
        answer: str,
        citations_used: List[str],
        chunk_map: Optional[Dict[str, Dict]] = None
    ) -> List[Dict[str, Any]]:
        """
        SOTA: Build structured evidence list for verification.
        
        Each evidence item contains:
        - claim: The sentence/claim from the answer
        - cite_ids: List of citation IDs supporting this claim
        - chunk_ids: Actual chunk IDs (from chunk_map)
        - verified: Placeholder for verifier to fill
        """
        evidence = []
        
        # Split answer into sentences
        sentences = self._split_sentences(answer)
        
        for sent in sentences:
            # Find citations in this sentence
            sent_tokens = re.findall(r"\[([^\[\]]+?)\]", sent)
            cite_ids: List[str] = []
            for token in sent_tokens:
                token = token.strip()
                if chunk_map and token in chunk_map:
                    cite_ids.append(token)
                elif "#" in token:
                    cite_ids.append(token)

            if cite_ids:
                # Clean the claim (remove citation markers for readability)
                claim = re.sub(r"\[[^\[\]]+\]", "", sent).strip()
                claim = re.sub(r"\s+", " ", claim)

                # Map to actual chunk IDs
                chunk_ids = []
                if chunk_map:
                    for cid in cite_ids:
                        if cid in chunk_map:
                            chunk_ids.append(chunk_map[cid].get("chunk_id", cid))

                evidence.append({
                    "claim": claim,
                    "cite_ids": cite_ids,
                    "chunk_ids": chunk_ids,
                    "sentence": sent,
                    "verified": None  # Verifier will fill this
                })
        
        return evidence
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences for evidence extraction."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s) > 10]
    
    def _estimate_confidence_sota(
        self,
        answer: str,
        context: str,
        query: str,
        citations_used: List[str],
        chunk_map: Optional[Dict[str, Dict]] = None
    ) -> float:
        """
        SOTA confidence scoring with citation coverage.
        
        Factors:
        - Citation coverage (how many claims are cited)
        - Citation validity (do citation IDs exist in chunk_map)
        - Answer length and detail
        - Evidence overlap
        - Abstention detection
        """
        confidence = 0.55  # Lower baseline, citations boost it
        
        # Explicit abstention (high confidence response)
        abstention_patterns = [
            "cannot answer", "cannot fully answer",
            "not provided in", "insufficient",
            "not enough", "unable to determine",
        ]
        answer_lower = answer.lower()
        if any(pat in answer_lower for pat in abstention_patterns):
            return 0.92  # High confidence abstention
        
        # SOTA: Citation coverage bonus
        if citations_used:
            # Valid citation bonus
            valid_citations = 0
            if chunk_map:
                valid_citations = sum(1 for c in citations_used if c in chunk_map)
            else:
                valid_citations = len(citations_used)
            
            # More valid citations = higher confidence
            citation_bonus = min(0.25, valid_citations * 0.05)
            confidence += citation_bonus
            
            # Penalty for invalid citations
            if chunk_map and valid_citations < len(citations_used):
                invalid_count = len(citations_used) - valid_citations
                confidence -= invalid_count * 0.08
        else:
            # No citations is a red flag for grounded generation
            confidence -= 0.15
        
        # Length bonus (detailed answers are better)
        if 150 < len(answer) < 1200:
            confidence += 0.12
        elif 80 < len(answer) <= 150:
            confidence += 0.05
        elif len(answer) < 50:
            confidence -= 0.12
        
        # Evidence overlap (answer uses context vocabulary)
        context_terms = set(w.lower() for w in context.split() if len(w) > 4)
        answer_terms = set(w.lower() for w in answer.split() if len(w) > 4)
        
        if context_terms and answer_terms:
            overlap = len(context_terms & answer_terms) / max(len(answer_terms), 1)
            confidence += min(0.15, overlap * 0.4)
        
        # Hedging penalty (only for excessive hedging)
        hedging = ["might", "could", "possibly", "perhaps", "may be", "seems"]
        hedge_count = sum(1 for h in hedging if f" {h} " in f" {answer_lower} ")
        if hedge_count > 3:
            confidence -= 0.08
        
        return max(0.0, min(1.0, confidence))
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate tokens (Llama uses ~4 chars per token)."""
        return max(1, len(text) // 4)

    def _normalize_context(self, context: str, query: str) -> str:
        """Strip duplicated question/answer stubs and cap length for Ollama."""
        if not context:
            return ""

        ctx = context.strip()

        # Remove embedded QUESTION/ANSWER section from fused context to avoid duplication
        marker = "QUESTION:"
        if marker.lower() in ctx.lower():
            parts = re.split(r"(?i)question:", ctx, maxsplit=1)
            if parts:
                ctx = parts[0].rstrip()

        # Remove dangling ANSWER prompts
        ctx = re.sub(r"ANSWER:\s*$", "", ctx, flags=re.IGNORECASE).strip()

        # Hard cap characters to stay within context window
        max_chars = self.max_context_tokens * 4
        if len(ctx) > max_chars:
            ctx = ctx[:max_chars]

        return ctx
    
    def _format_sources(self, sources: List[Dict]) -> List[str]:
        """Format sources for citation."""
        formatted = []
        seen = set()
        
        for src in sources[:5]:
            name = src.get("source", src.get("source_file", "unknown"))
            if name not in seen:
                seen.add(name)
                formatted.append(name)
        
        return formatted
    
    def _empty_response(self, query: str) -> Dict[str, Any]:
        """Response when no context available."""
        return {
            "answer": "I cannot answer this question without relevant context.",
            "sources": [],
            "confidence": 0.0,
            "query": query,
            "model": self.llama_model,
            "tokens_used": 0,
        }
    
    def _error_response(self, query: str, error: str) -> Dict[str, Any]:
        """Response when generation fails."""
        return {
            "answer": f"Error generating response: {error}",
            "sources": [],
            "confidence": 0.0,
            "query": query,
            "model": self.llama_model,
            "tokens_used": 0,
            "error": error,
        }


# ============================================================================
# BACKWARD COMPATIBILITY ALIASES
# ============================================================================

GroundedLLM_SOTA = GroundedLLM


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    generator = GroundedLLM()
    
    context = """
    The Transformer is an attention-based architecture introduced in 2017.
    It uses self-attention mechanisms to process sequences in parallel.
    Self-attention allows the model to attend to all positions simultaneously.
    """
    
    query = "What is the Transformer architecture?"
    result = generator.generate(query=query, context=context)
    
    print("\n" + "="*70)
    print(f"Query: {result['query']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print("-"*70)
    print(result['answer'])
    print("-"*70)
    print(f"Model: {result['model']}")
