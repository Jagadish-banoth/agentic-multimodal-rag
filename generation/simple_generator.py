"""
generation/simple_generator.py
==============================
Lightweight generator optimized for smaller models (Gemma 2B, Phi-3, etc.)
Uses simple prompts instead of complex JSON schemas.
"""

import logging
import ollama
from typing import Dict, Any

logger = logging.getLogger(__name__)


class SimpleGenerator:
    """
    Simple, efficient generator for smaller local models.
    Returns plain text answers with citations.
    """
    
    def __init__(self, model_name: str = "gemma2:2b"):
        self.model_name = model_name
        self.temperature = 0.1
        self.max_tokens = 500
        logger.info(f"Initialized SimpleGenerator | model={model_name}")
    
    def generate(self, query: str, fused_context: str) -> Dict[str, Any]:
        """
        Generate grounded answer from context.
        
        Args:
            query: User question
            fused_context: Retrieved and fused context
        
        Returns:
            Dict with answer, confidence, and metadata
        """
        if not fused_context or not fused_context.strip():
            return {
                "answer": "I don't have enough context to answer this question.",
                "confidence": 0.0,
                "sources": [],
                "reasoning": "No context provided"
            }
        
        # Build simple prompt
        prompt = self._build_prompt(query, fused_context)
        
        try:
            # Generate answer
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                }
            )
            
            answer_text = response["message"]["content"].strip()
            
            # Extract citations if present
            citations = self._extract_citations(answer_text)
            
            # Estimate confidence
            confidence = self._estimate_confidence(answer_text, fused_context)
            
            return {
                "answer": answer_text,
                "concise_answer": self._make_concise(answer_text),
                "answer_long": answer_text,
                "confidence": {"score": confidence, "level": self._conf_level(confidence)},
                "sources_used": citations,
                "reasoning_summary": "Generated from retrieved context",
                "evidence": []
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "answer": f"Generation error: {str(e)[:100]}",
                "confidence": {"score": 0.0, "level": "LOW"},
                "sources_used": [],
                "error": str(e)
            }
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build optimized prompt for smaller models."""
        return f"""You are a helpful AI assistant. Answer the question based ONLY on the provided context.

RULES:
- Be concise and accurate
- Use information from the context only
- If the answer isn't in the context, say "I don't have enough information"
- Include relevant details but keep it focused

CONTEXT:
{context[:2000]}

QUESTION: {query}

ANSWER:"""
    
    def _extract_citations(self, answer: str) -> list:
        """Extract source references from answer."""
        import re
        # Look for patterns like [1], [source: ...], etc.
        citations = re.findall(r'\[([^\]]+)\]', answer)
        return list(set(citations)) if citations else []
    
    def _estimate_confidence(self, answer: str, context: str) -> float:
        """Estimate answer confidence based on content."""
        # Simple heuristic
        if "don't have" in answer.lower() or "insufficient" in answer.lower():
            return 0.2
        if "unsure" in answer.lower() or "not clear" in answer.lower():
            return 0.4
        
        # Check overlap with context
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        overlap = len(answer_words & context_words) / max(len(answer_words), 1)
        
        # Base confidence + overlap bonus
        return min(0.7 + (overlap * 0.2), 0.95)
    
    def _conf_level(self, score: float) -> str:
        """Convert confidence score to level."""
        if score >= 0.7:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        return "LOW"
    
    def _make_concise(self, text: str) -> str:
        """Extract first 2-3 sentences as concise answer."""
        sentences = text.split('. ')
        concise = '. '.join(sentences[:2])
        if not concise.endswith('.'):
            concise += '.'
        return concise


if __name__ == "__main__":
    # Quick test
    gen = SimpleGenerator("gemma2:2b")
    
    context = """
    Self-attention is a mechanism in transformers that computes relationships between 
    all positions in a sequence. It uses Query, Key, and Value matrices to determine 
    attention weights. The formula is: Attention(Q,K,V) = softmax(QK^T/√d_k)V
    """
    
    result = gen.generate("What is self-attention?", context)
    print("\nAnswer:", result["answer"])
    print("Confidence:", result["confidence"])
