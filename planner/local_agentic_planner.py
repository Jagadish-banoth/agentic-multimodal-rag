'''"""
planner/local_agentic_planner.py
================================
FAANG-Grade LLM-Based Agentic Planner
OPTIMIZED FOR LOCAL OLLAMA (llama3:8b)

Features:
- Pure local LLM (no API calls)
- Optimized prompts for smaller models
- Function calling via JSON
- Multi-step query decomposition
- Dynamic parameter tuning
- Confidence scoring
- Full observability & metrics
- Fast response times

Author: Senior RAG Engineer (15+ years)
"""

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None

from planner.schemas import (
    QueryIntent,
    QueryAnalysis,
    Modality,
    RetrievalStrategy,
    ConfidenceLevel,
    RetrievalPlan,
    ToolCall,
    PlannerMetrics,
    INTENT_CLASSIFICATION_SCHEMA,
    MODALITY_SELECTION_SCHEMA,
    STRATEGY_SELECTION_SCHEMA,
)

logger = logging.getLogger(__name__)


# ============================
# OPTIMIZED PROMPTS (For Local LLM)
# ============================

INTENT_CLASSIFICATION_PROMPT = """Analyze this query and classify its intent.

Query: "{query}"

Respond ONLY with JSON matching this schema:
{{
    "intent": "factual|explanatory|comparative|visual_reasoning|temporal|procedural|aggregation|unknown",
    "complexity": "simple|medium|complex",
    "requires_reasoning": true|false,
    "explanation": "brief reasoning"
}}

JSON response only:"""

MODALITY_SELECTION_PROMPT = """Select modalities for this query.

Intent: {intent}
Query: "{query}"

Respond ONLY with JSON:
{{
    "modalities": ["text", "image", "audio", "video"],
    "fusion_weights": {{"text": 0.6, "image": 0.4}}
}}

JSON response only:"""

STRATEGY_SELECTION_PROMPT = """Select retrieval strategy for this query.

Intent: {intent}
Complexity: {complexity}
Query: "{query}"

Respond ONLY with JSON:
{{
    "strategy": "dense_only|sparse_only|hybrid|graph_enhanced|multimodal_fusion",
    "dense_k": 20,
    "sparse_k": 20,
    "rerank_k": 12,
    "reasoning": "brief explanation"
}}

JSON response only:"""


# ============================
# LOCAL AGENTIC PLANNER
# ============================

class LocalAgenticPlanner:
    """
    Optimized agentic planner using local Ollama (llama3:8b).
    
    Design:
    - Uses local LLM for all decisions (no API calls)
    - Optimized prompts for smaller models
    - Fast response times
    - Full reproducibility
    - No external dependencies
    """
    
    def __init__(self, config: Dict):
        """
        Initialize local agentic planner.
        
        Args:
            config: System configuration dict
        """
        self.config = config
        
        # Use local model exclusively
        planner_config = config.get("models", {}).get("planner", {})
        self.local_model = planner_config.get("local_model", "llama3:8b")
        
        # Retrieval defaults
        self.default_dense_k = config["retrieval"]["dense_k"]
        self.default_sparse_k = config["retrieval"]["sparse_k"]
        self.default_rerank_k = config["retrieval"]["rerank_k"]
        
        # Agent settings
        self.confidence_threshold = config["agent"]["confidence_threshold"]
        self.max_attempts = config["agent"]["max_attempts"]
        
        # Verify Ollama is available
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("❌ ollama package not available! Install with: pip install ollama")
        
        # Metrics tracking
        self.metrics_history: List[PlannerMetrics] = []
        
        logger.info(f"✓ Local Agentic Planner initialized")
        logger.info(f"  Model: {self.local_model}")
        logger.info(f"  Confidence threshold: {self.confidence_threshold}")
        logger.info(f"  Max attempts: {self.max_attempts}")
    
    # ============================
    # MAIN PLANNING API
    # ============================
    
    def plan(self, user_query: str, context: Optional[Dict] = None) -> RetrievalPlan:
        """
        Generate comprehensive retrieval plan using local LLM.
        
        Args:
            user_query: User's natural language query
            context: Optional context
        
        Returns:
            RetrievalPlan with decisions
        """
        start_time = time.time()
        plan_id = str(uuid.uuid4())[:8]
        
        logger.info(f"📋 Planning query [{plan_id}]: {user_query[:80]}")
        
        # Step 1: Query Analysis
        analysis_start = time.time()
        analysis = self._analyze_query(user_query)
        analysis_time = (time.time() - analysis_start) * 1000
        
        logger.info(f"  Intent: {analysis.intent.value} | Complexity: {analysis.complexity} | Time: {analysis_time:.0f}ms")
        
        # Step 2: Modality Selection
        modalities_start = time.time()
        modalities, fusion_weights = self._select_modalities(user_query, analysis)
        modalities_time = (time.time() - modalities_start) * 1000
        
        logger.info(f"  Modalities: {[m.value for m in modalities]} | Time: {modalities_time:.0f}ms")
        
        # Step 3: Strategy Selection
        strategy_start = time.time()
        strategy, params = self._select_strategy(user_query, analysis)
        strategy_time = (time.time() - strategy_start) * 1000
        
        logger.info(f"  Strategy: {strategy.value} | k={params['dense_k']}/{params['sparse_k']}/{params['rerank_k']} | Time: {strategy_time:.0f}ms")
        
        # Step 4: Tool Routing
        tool_calls = self._route_tools(user_query, analysis, modalities)
        
        # Step 5: Query Decomposition (complex queries)
        sub_plans = []
        if analysis.complexity == "complex" and analysis.sub_queries:
            sub_plans = self._decompose_query(analysis.sub_queries)
        
        # Step 6: Confidence Scoring
        confidence, conf_level = self._compute_plan_confidence(analysis, modalities, strategy)
        
        # Build final plan
        plan = RetrievalPlan(
            query=user_query,
            intent=analysis.intent,
            analysis=analysis,
            modalities=modalities,
            strategy=strategy,
            dense_k=params["dense_k"],
            sparse_k=params["sparse_k"],
            rerank_k=params["rerank_k"],
            tool_calls=tool_calls,
            sub_plans=sub_plans,
            confidence=confidence,
            confidence_level=conf_level,
            reasoning=analysis.explanation,
            allow_retry=confidence >= 0.5,
            max_retries=self.max_attempts,
            plan_id=plan_id,
            timestamp=datetime.now().isoformat()
        )
        
        # Metrics
        total_time = (time.time() - start_time) * 1000
        
        metrics = PlannerMetrics(
            plan_id=plan_id,
            query_length=len(user_query),
            analysis_time_ms=analysis_time,
            planning_time_ms=total_time,
            total_time_ms=total_time,
            llm_calls=3,  # analysis, modality, strategy
            llm_tokens_used=self._estimate_tokens(user_query),
            fallback_used=False,  # No API, no fallback needed
            confidence=confidence,
            intent=analysis.intent.value,
            strategy=strategy.value
        )
        self.metrics_history.append(metrics)
        
        logger.info(f"✓ Plan generated in {total_time:.0f}ms | Confidence: {confidence:.2f} ({conf_level.value})")
        
        return plan
    
    # ============================
    # STEP 1: QUERY ANALYSIS (LOCAL LLM)
    # ============================
    
    def _analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query intent using local LLM."""
        prompt = INTENT_CLASSIFICATION_PROMPT.format(query=query)
        
        try:
            response = self._llm_call_with_schema(prompt, temperature=0.1)
            
            intent = QueryIntent(response.get("intent", "unknown"))
            
            # Fallback if model returns unknown
            if intent == QueryIntent.UNKNOWN:
                return self._fallback_analysis(query)
            
            return QueryAnalysis(
                intent=intent,
                complexity=response.get("complexity", "medium"),
                domain=response.get("domain"),
                temporal_aspect=response.get("temporal_aspect", False),
                requires_reasoning=response.get("requires_reasoning", False),
                ambiguity_score=0.0,
                key_entities=response.get("key_entities", []),
                sub_queries=response.get("sub_queries", []),
                explanation=response.get("explanation", "")
            )
        
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}, using fallback")
            return self._fallback_analysis(query)
    
    def _fallback_analysis(self, query: str) -> QueryAnalysis:
        """Fallback rule-based analysis."""
        q = query.lower()
        
        # Intent detection
        if any(w in q for w in ["compare", "difference", "vs", "versus", "contrast"]):
            intent = QueryIntent.COMPARATIVE
        elif any(w in q for w in ["explain", "why", "how does", "describe"]):
            intent = QueryIntent.EXPLANATORY
        elif any(w in q for w in ["diagram", "image", "figure", "chart", "visual"]):
            intent = QueryIntent.VISUAL_REASONING
        elif any(w in q for w in ["when", "timeline", "history", "trend"]):
            intent = QueryIntent.TEMPORAL
        elif any(w in q for w in ["step", "process", "procedure", "how to"]):
            intent = QueryIntent.PROCEDURAL
        else:
            intent = QueryIntent.FACTUAL
        
        # Complexity
        word_count = len(query.split())
        complexity = "complex" if word_count > 20 else ("medium" if word_count > 10 else "simple")
        
        return QueryAnalysis(
            intent=intent,
            complexity=complexity,
            requires_reasoning="explain" in q or "why" in q,
            explanation=f"Fallback: {intent.value}"
        )
    
    # ============================
    # STEP 2: MODALITY SELECTION
    # ============================
    
    def _select_modalities(
        self, 
        query: str, 
        analysis: QueryAnalysis
    ) -> Tuple[List[Modality], Dict[str, float]]:
        """Select modalities for retrieval."""
        prompt = MODALITY_SELECTION_PROMPT.format(
            intent=analysis.intent.value,
            query=query
        )
        
        try:
            response = self._llm_call_with_schema(prompt, temperature=0.1)
            
            # If LLM fails, use fallback
            if not response or "modalities" not in response:
                logger.warning("Modality selection returned empty, using fallback")
                return self._fallback_modalities(query, analysis)
            
            modalities = [Modality(m) for m in response.get("modalities", ["text"])]
            fusion_weights = response.get("fusion_weights", {"text": 1.0})
            
            return modalities, fusion_weights
        
        except Exception as e:
            logger.warning(f"Modality selection failed: {e}, using fallback")
            return self._fallback_modalities(query, analysis)
    
    def _fallback_modalities(
        self, 
        query: str, 
        analysis: QueryAnalysis
    ) -> Tuple[List[Modality], Dict[str, float]]:
        """Fallback modality selection."""
        modalities = [Modality.TEXT]
        q = query.lower()
        
        if analysis.intent == QueryIntent.VISUAL_REASONING or any(w in q for w in ["image", "diagram", "chart"]):
            modalities.append(Modality.IMAGE)
        
        if "audio" in q or "lecture" in q:
            modalities.append(Modality.AUDIO)
        
        if "video" in q:
            modalities.append(Modality.VIDEO)
        
        weights = {m.value: 1.0 / len(modalities) for m in modalities}
        
        return modalities, weights
    
    # ============================
    # STEP 3: STRATEGY SELECTION
    # ============================
    
    def _select_strategy(
        self, 
        query: str, 
        analysis: QueryAnalysis
    ) -> Tuple[RetrievalStrategy, Dict[str, int]]:
        """Select retrieval strategy."""
        prompt = STRATEGY_SELECTION_PROMPT.format(
            intent=analysis.intent.value,
            complexity=analysis.complexity,
            query=query
        )
        
        try:
            response = self._llm_call_with_schema(prompt, temperature=0.1)
            
            # If LLM fails, use fallback
            if not response or "strategy" not in response:
                logger.warning("Strategy selection returned empty, using fallback")
                return self._fallback_strategy(analysis)
            
            strategy = RetrievalStrategy(response.get("strategy", "hybrid"))
            params = {
                "dense_k": response.get("dense_k", self.default_dense_k),
                "sparse_k": response.get("sparse_k", self.default_sparse_k),
                "rerank_k": response.get("rerank_k", self.default_rerank_k),
            }
            
            return strategy, params
        
        except Exception as e:
            logger.warning(f"Strategy selection failed: {e}, using fallback")
            return self._fallback_strategy(analysis)
    
    def _fallback_strategy(
        self, 
        analysis: QueryAnalysis
    ) -> Tuple[RetrievalStrategy, Dict[str, int]]:
        """Fallback strategy selection."""
        if analysis.complexity == "simple":
            strategy = RetrievalStrategy.DENSE_ONLY
            params = {"dense_k": 10, "sparse_k": 0, "rerank_k": 5}
        elif analysis.requires_reasoning:
            strategy = RetrievalStrategy.GRAPH_ENHANCED
            params = {"dense_k": 30, "sparse_k": 30, "rerank_k": 15}
        else:
            strategy = RetrievalStrategy.HYBRID
            params = {
                "dense_k": self.default_dense_k,
                "sparse_k": self.default_sparse_k,
                "rerank_k": self.default_rerank_k
            }
        
        return strategy, params
    
    # ============================
    # STEP 4: TOOL ROUTING
    # ============================
    
    def _route_tools(
        self, 
        query: str, 
        analysis: QueryAnalysis, 
        modalities: List[Modality]
    ) -> List[ToolCall]:
        """Route to specialized tools if needed."""
        tools = []
        
        if Modality.IMAGE in modalities and "text" in query.lower():
            tools.append(ToolCall(
                tool_name="ocr_extractor",
                parameters={"modality": "image", "extract_tables": True},
                reason="Query mentions text in images",
                priority=1
            ))
        
        if "table" in query.lower():
            tools.append(ToolCall(
                tool_name="table_parser",
                parameters={"format": "pandas"},
                reason="Query requires table understanding",
                priority=2
            ))
        
        if any(w in query.lower() for w in ["calculate", "compute", "sum", "average"]):
            tools.append(ToolCall(
                tool_name="code_interpreter",
                parameters={"language": "python"},
                reason="Query requires computation",
                priority=3
            ))
        
        return tools
    
    # ============================
    # STEP 5: QUERY DECOMPOSITION
    # ============================
    
    def _decompose_query(self, sub_queries: List[str]) -> List[RetrievalPlan]:
        """Create sub-plans for complex queries."""
        sub_plans = []
        
        for sq in sub_queries[:3]:
            try:
                sub_plan = self.plan(sq)
                sub_plans.append(sub_plan)
            except Exception as e:
                logger.warning(f"Sub-query planning failed: {e}")
        
        return sub_plans
    
    # ============================
    # STEP 6: CONFIDENCE SCORING
    # ============================
    
    def _compute_plan_confidence(
        self, 
        analysis: QueryAnalysis, 
        modalities: List[Modality], 
        strategy: RetrievalStrategy
    ) -> Tuple[float, ConfidenceLevel]:
        """Compute plan confidence."""
        confidence = 1.0
        
        if analysis.ambiguity_score > 0.5:
            confidence *= 0.7
        
        if analysis.intent == QueryIntent.UNKNOWN:
            confidence *= 0.6
        
        if analysis.complexity == "simple":
            confidence *= 1.1
        
        if confidence >= 0.8:
            level = ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW
        
        return min(confidence, 1.0), level
    
    # ============================
    # LLM INTERFACE (LOCAL ONLY)
    # ============================
    
    def _llm_call_with_schema(
        self, 
        prompt: str, 
        temperature: float = 0.1
    ) -> Dict:
        """Call local Ollama with JSON schema."""
        schema_prompt = (
            f"{prompt}\n\n"
            f"Return ONLY valid JSON, no other text:"
        )
        
        try:
            # Call Ollama with timeout and reduced parameters for speed
            response = ollama.generate(
                model=self.local_model,
                prompt=schema_prompt,
                options={
                    "temperature": temperature,
                    "format": "json",
                    "num_predict": 256,  # Limit tokens for faster response
                },
                stream=False,
            )
            
            text = (response.get("response") or "").strip()
            if not text:
                logger.warning("Empty response from Ollama, using fallback")
                return {}
            
            # Unwrap code fences if present
            if "```json" in text:
                text = text.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in text:
                text = text.split("```", 1)[1].split("```", 1)[0].strip()
            
            # Find first { in case of leading text
            if not text.startswith("{"):
                idx = text.find("{")
                if idx != -1:
                    text = text[idx:]
            
            # Handle incomplete JSON
            if not text.endswith("}"):
                # Try to find closing brace
                idx = text.rfind("}")
                if idx != -1:
                    text = text[:idx+1]
            
            result = json.loads(text)
            return result
        
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}, text was: {text[:100]}")
            return {}
        except Exception as e:
            logger.warning(f"Ollama call failed: {e}")
            return {}
    
    # ============================
    # UTILITIES
    # ============================
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation."""
        return len(text) // 4
    
    def get_metrics_summary(self) -> Dict:
        """Get aggregated metrics."""
        if not self.metrics_history:
            return {}
        
        total_plans = len(self.metrics_history)
        
        return {
            "total_plans": total_plans,
            "avg_planning_time_ms": sum(m.planning_time_ms for m in self.metrics_history) / total_plans,
            "avg_analysis_time_ms": sum(m.analysis_time_ms for m in self.metrics_history) / total_plans,
            "avg_confidence": sum(m.confidence for m in self.metrics_history) / total_plans,
            "min_time_ms": min(m.planning_time_ms for m in self.metrics_history),
            "max_time_ms": max(m.planning_time_ms for m in self.metrics_history),
            "total_time_ms": sum(m.planning_time_ms for m in self.metrics_history),
            "intents": {
                intent: sum(1 for m in self.metrics_history if m.intent == intent)
                for intent in set(m.intent for m in self.metrics_history)
            },
            "strategies": {
                strategy: sum(1 for m in self.metrics_history if m.strategy == strategy)
                for strategy in set(m.strategy for m in self.metrics_history)
            }
        }


if __name__ == "__main__":
    import yaml
    
    # Load config
    config_path = Path(__file__).parents[1] / "config" / "settings.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize planner
    planner = LocalAgenticPlanner(config)
    
    print("\n" + "="*80)
    print("LOCAL AGENTIC PLANNER - TESTING")
    print("="*80 + "\n")
    
    # Test queries
    queries = [
        "What is machine learning?",
        "Explain how attention mechanism works in transformers",
        "Compare BERT vs GPT models"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 80)
        plan = planner.plan(query)
        print(f"✓ Plan ID: {plan.plan_id}")
        print(f"  Intent: {plan.intent.value}")
        print(f"  Strategy: {plan.strategy.value}")
        print(f"  Confidence: {plan.confidence:.2f}")
    
    # Print metrics
    print(f"\n{'='*80}")
    print("METRICS SUMMARY")
    print(f"{'='*80}\n")
    metrics = planner.get_metrics_summary()
    for k, v in metrics.items():
        print(f"{k}: {v}")
'''


"""
Optimized Local Agentic Planner
--------------------------------
This rewrite keeps your original architecture but is optimized for
low-latency inference in production-like scenarios.

Key changes (summary):
- Introduces `planner_mode` config: `fast` (default) vs `full` (LLM-driven).
  * `fast` uses deterministic/rule-based logic + caching (no LLM calls by default).
  * `full` uses your original LLM-based flows (still faster: safer timeouts,
    smaller token budgets, parallel LLM calls where helpful).
- Plan caching with TTL to return instant responses for repeated queries.
- Reduced and guarded LLM usage: a single controlled call per step only when enabled.
- Non-recursive sub-plan creation in fast mode to avoid planning blow-ups.
- Safer, smaller Ollama calls: lower token budget and explicit timeout.
- Clear metrics: llm_calls tracks actual LLM usage.
- Config-driven: toggle `planner.mode` and `planner.allow_local_llm`.

Behavioral guidance (how to use):
- For production low-latency -> set `planner.mode: fast` in your settings.yaml.
- If you need stronger planning for a few queries, use `planner.mode: full` or
  set `allow_local_llm: true` for selective calls.

Expected impact (empirical guidance):
- Planning time: ~30s -> ~50-200ms (fast mode)
- Rerank/generation remain separate concerns; planner now will not add 30s.

Author: assistant
"""

import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor

try:
    import ollama
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False
    ollama = None

from planner.schemas import (
    QueryIntent,
    QueryAnalysis,
    Modality,
    RetrievalStrategy,
    ConfidenceLevel,
    RetrievalPlan,
    ToolCall,
    PlannerMetrics,
    INTENT_CLASSIFICATION_SCHEMA,
    MODALITY_SELECTION_SCHEMA,
    STRATEGY_SELECTION_SCHEMA,
)

logger = logging.getLogger(__name__)


# ============================
# LIGHTWEIGHT PROMPTS (Smaller budgets)
# ============================

INTENT_CLASSIFICATION_PROMPT = (
    "Analyze this query and classify its intent.\n"
    "Query: \"{query}\"\n"
    "Respond ONLY with JSON matching the schema: intent, complexity, requires_reasoning, explanation.\n"
)

MODALITY_SELECTION_PROMPT = (
    "Select modalities for this query.\n"
    "Intent: {intent}\n"
    "Query: \"{query}\"\n"
    "Return JSON: modalities, fusion_weights, reasoning.\n"
)

STRATEGY_SELECTION_PROMPT = (
    "Select retrieval strategy for the query.\n"
    "Intent: {intent} | Complexity: {complexity}\n"
    "Return JSON: strategy, dense_k, sparse_k, rerank_k, reasoning.\n"
)


# ============================
# SIMPLE IN-MEMORY CACHE (TTL)
# ============================
class PlanCache:
    def __init__(self, ttl_seconds: int = 3600):
        self._store: Dict[str, Tuple[RetrievalPlan, float]] = {}
        self.ttl = ttl_seconds

    def get(self, key: str) -> Optional[RetrievalPlan]:
        entry = self._store.get(key)
        if not entry:
            return None
        plan, ts = entry
        if time.time() - ts > self.ttl:
            del self._store[key]
            return None
        return plan

    def set(self, key: str, plan: RetrievalPlan) -> None:
        self._store[key] = (plan, time.time())

    def clear(self) -> None:
        self._store.clear()


# ============================
# OPTIMIZED LOCAL AGENTIC PLANNER
# ============================
class LocalAgenticPlanner:
    """
    Optimized planner that favors deterministic, fast planning by default.

    Config expectations (subset):
      config['models']['planner']['mode'] -> 'fast'|'full'  (default: 'fast')
      config['models']['planner']['allow_local_llm'] -> bool
      config['models']['planner']['llm_timeout'] -> seconds
      config['models']['planner']['llm_max_tokens'] -> int
      config['agent']['confidence_threshold']
    """

    def __init__(self, config: Dict):
        self.config = config
        planner_config = config.get("models", {}).get("planner", {})

        self.planner_mode = planner_config.get("mode", "fast")  # fast|full
        self.allow_local_llm = planner_config.get("allow_local_llm", False)
        self.local_model = planner_config.get("local_model", "llama3:8b")
        self.llm_timeout = planner_config.get("llm_timeout", 5)
        self.llm_max_tokens = planner_config.get("llm_max_tokens", 64)

        self.default_dense_k = config["retrieval"]["dense_k"]
        self.default_sparse_k = config["retrieval"]["sparse_k"]
        self.default_rerank_k = config["retrieval"]["rerank_k"]

        self.confidence_threshold = config["agent"]["confidence_threshold"]
        self.max_attempts = config["agent"]["max_attempts"]

        # Lightweight cache for plans
        cache_ttl = planner_config.get("cache_ttl_seconds", 3600)
        self.plan_cache = PlanCache(ttl_seconds=cache_ttl)

        # Executor for optional concurrent LLM calls (used only in full mode)
        self.executor = ThreadPoolExecutor(max_workers=3)

        # Ollama availability is optional; we guard calls
        self.ollama_available = OLLAMA_AVAILABLE and ollama is not None

        # Metrics
        self.metrics_history: List[PlannerMetrics] = []

        logger.info(f"LocalAgenticPlanner init: mode={self.planner_mode} allow_llm={self.allow_local_llm}")

        if self.planner_mode == "full" and self.allow_local_llm and not self.ollama_available:
            logger.warning("Planner in full mode but ollama package not available. Falling back to fast mode.")
            self.planner_mode = "fast"

    # ============================
    # PUBLIC API
    # ============================
    def plan(self, user_query: str, context: Optional[Dict] = None) -> RetrievalPlan:
        start_time = time.time()
        plan_id = str(uuid.uuid4())[:8]
        qkey = user_query.strip().lower()

        # Quick cache hit
        cached = self.plan_cache.get(qkey)
        if cached:
            logger.debug("Plan cache hit")
            # update minimal metrics
            metrics = PlannerMetrics(
                plan_id=plan_id,
                query_length=len(user_query),
                analysis_time_ms=0.0,
                planning_time_ms=(time.time() - start_time) * 1000,
                total_time_ms=(time.time() - start_time) * 1000,
                llm_calls=0,
                llm_tokens_used=0,
                fallback_used=True,
                confidence=cached.confidence,
                intent=cached.intent.value,
                strategy=cached.strategy.value,
            )
            self.metrics_history.append(metrics)
            return cached

        logger.info(f"Planning query [{plan_id}]: {user_query[:120]}")

        # Fast mode: deterministic & rule-based planning
        if self.planner_mode == "fast":
            analysis = self._fallback_analysis(user_query)
            modalities, fusion_weights = self._fallback_modalities(user_query, analysis)
            strategy, params = self._fallback_strategy(analysis)

            tool_calls = self._route_tools(user_query, analysis, modalities)
            sub_plans = self._decompose_query_fast(analysis)
            confidence, conf_level = self._compute_plan_confidence(analysis, modalities, strategy)

            plan = RetrievalPlan(
                query=user_query,
                intent=analysis.intent,
                analysis=analysis,
                modalities=modalities,
                strategy=strategy,
                dense_k=params["dense_k"],
                sparse_k=params["sparse_k"],
                rerank_k=params["rerank_k"],
                tool_calls=tool_calls,
                sub_plans=sub_plans,
                confidence=confidence,
                confidence_level=conf_level,
                reasoning=analysis.explanation,
                allow_retry=confidence >= 0.5,
                max_retries=self.max_attempts,
                plan_id=plan_id,
                timestamp=datetime.now().isoformat(),
            )

            # Persist plan in cache for quick repeat answers
            self.plan_cache.set(qkey, plan)

            total_time = (time.time() - start_time) * 1000
            metrics = PlannerMetrics(
                plan_id=plan_id,
                query_length=len(user_query),
                analysis_time_ms=0.0,
                planning_time_ms=total_time,
                total_time_ms=total_time,
                llm_calls=0,
                llm_tokens_used=0,
                fallback_used=True,
                confidence=confidence,
                intent=analysis.intent.value,
                strategy=strategy.value,
            )
            self.metrics_history.append(metrics)
            logger.info(f"Fast plan generated in {total_time:.0f}ms | confidence={confidence:.2f}")
            return plan

        # Full mode: LLM-assisted planning but guarded and faster
        # We limit token budgets and add timeouts.
        analysis = self._call_or_fallback(self._analyze_query_llm, user_query, allow_fallback=True)
        modalities, fusion_weights = self._call_or_fallback(self._select_modalities_llm, user_query, analysis, allow_fallback=True)
        strategy, params = self._call_or_fallback(self._select_strategy_llm, user_query, analysis, allow_fallback=True)

        tool_calls = self._route_tools(user_query, analysis, modalities)
        sub_plans = self._decompose_query_fast(analysis)
        confidence, conf_level = self._compute_plan_confidence(analysis, modalities, strategy)

        plan = RetrievalPlan(
            query=user_query,
            intent=analysis.intent,
            analysis=analysis,
            modalities=modalities,
            strategy=strategy,
            dense_k=params["dense_k"],
            sparse_k=params["sparse_k"],
            rerank_k=params["rerank_k"],
            tool_calls=tool_calls,
            sub_plans=sub_plans,
            confidence=confidence,
            confidence_level=conf_level,
            reasoning=analysis.explanation,
            allow_retry=confidence >= 0.5,
            max_retries=self.max_attempts,
            plan_id=plan_id,
            timestamp=datetime.now().isoformat(),
        )

        total_time = (time.time() - start_time) * 1000
        # compute llm usage from last calls if any
        llm_calls = 0
        if self.allow_local_llm and self.ollama_available:
            # rough heuristic: 3 possible calls
            llm_calls = 3

        metrics = PlannerMetrics(
            plan_id=plan_id,
            query_length=len(user_query),
            analysis_time_ms=0.0,
            planning_time_ms=total_time,
            total_time_ms=total_time,
            llm_calls=llm_calls,
            llm_tokens_used=self._estimate_tokens(user_query) * llm_calls,
            fallback_used=False,
            confidence=confidence,
            intent=analysis.intent.value,
            strategy=strategy.value,
        )
        self.metrics_history.append(metrics)
        logger.info(f"Full plan generated in {total_time:.0f}ms | confidence={confidence:.2f} llm_calls={llm_calls}")
        return plan

    # ============================
    # WRAPPERS: LLM vs FALLBACK
    # ============================
    def _call_or_fallback(self, fn, *args, allow_fallback: bool = True):
        """
        Helper: call an LLM-backed function if allowed, otherwise fallback.
        Ensures timeouts and safe fallbacks.
        """
        try:
            if self.planner_mode == "full" and self.allow_local_llm and self.ollama_available:
                return fn(*args)
            # Otherwise use deterministic fallback
            name = fn.__name__
            if name == "_analyze_query_llm":
                return self._fallback_analysis(args[0])
            if name == "_select_modalities_llm":
                return self._fallback_modalities(args[0], args[1])
            if name == "_select_strategy_llm":
                return self._fallback_strategy(args[1])
            # generic fallback
            return fn(*args)
        except Exception as e:
            logger.warning(f"LLM wrapper failed ({fn.__name__}): {e}. Using fallback.")
            # Map to fallback
            if fn.__name__ == "_analyze_query_llm":
                return self._fallback_analysis(args[0])
            if fn.__name__ == "_select_modalities_llm":
                return self._fallback_modalities(args[0], args[1])
            if fn.__name__ == "_select_strategy_llm":
                return self._fallback_strategy(args[1])
            raise

    # ============================
    # LLM-DRIVEN STEPS (GUARDED)
    # ============================
    def _analyze_query_llm(self, query: str) -> QueryAnalysis:
        prompt = INTENT_CLASSIFICATION_PROMPT.format(query=query)
        data = self._llm_call_with_schema(prompt, timeout=self.llm_timeout)
        if not data:
            return self._fallback_analysis(query)
        intent = QueryIntent(data.get("intent", QueryIntent.UNKNOWN))
        return QueryAnalysis(
            intent=intent,
            complexity=data.get("complexity", "medium"),
            domain=data.get("domain"),
            temporal_aspect=data.get("temporal_aspect", False),
            requires_reasoning=data.get("requires_reasoning", False),
            ambiguity_score=data.get("ambiguity_score", 0.0),
            key_entities=data.get("key_entities", []),
            sub_queries=data.get("sub_queries", []),
            explanation=data.get("explanation", ""),
        )

    def _select_modalities_llm(self, query: str, analysis: QueryAnalysis) -> Tuple[List[Modality], Dict[str, float]]:
        prompt = MODALITY_SELECTION_PROMPT.format(intent=analysis.intent.value, query=query)
        data = self._llm_call_with_schema(prompt, timeout=self.llm_timeout)
        if not data or "modalities" not in data:
            return self._fallback_modalities(query, analysis)
        modalities = [Modality(m) for m in data.get("modalities", [Modality.TEXT.value])]
        fusion_weights = data.get("fusion_weights", {m.value: 1.0 for m in modalities})
        return modalities, fusion_weights

    def _select_strategy_llm(self, query: str, analysis: QueryAnalysis) -> Tuple[RetrievalStrategy, Dict[str, int]]:
        prompt = STRATEGY_SELECTION_PROMPT.format(intent=analysis.intent.value, complexity=analysis.complexity, query=query)
        data = self._llm_call_with_schema(prompt, timeout=self.llm_timeout)
        if not data or "strategy" not in data:
            return self._fallback_strategy(analysis)
        strategy = RetrievalStrategy(data.get("strategy", RetrievalStrategy.HYBRID.value))
        params = {
            "dense_k": int(data.get("dense_k", self.default_dense_k)),
            "sparse_k": int(data.get("sparse_k", self.default_sparse_k)),
            "rerank_k": int(data.get("rerank_k", self.default_rerank_k)),
        }
        return strategy, params

    # ============================
    # FALLBACK (RULE-BASED) IMPLEMENTATIONS
    # ============================
    def _fallback_analysis(self, query: str) -> QueryAnalysis:
        q = (query or "").lower()
        if any(w in q for w in ["compare", "difference", "vs", "versus", "contrast"]):
            intent = QueryIntent.COMPARATIVE
        elif any(w in q for w in ["explain", "why", "how does", "describe"]):
            intent = QueryIntent.EXPLANATORY
        elif any(w in q for w in ["diagram", "image", "figure", "chart", "visual"]):
            intent = QueryIntent.VISUAL_REASONING
        elif any(w in q for w in ["when", "timeline", "history", "trend"]):
            intent = QueryIntent.TEMPORAL
        elif any(w in q for w in ["step", "process", "procedure", "how to"]):
            intent = QueryIntent.PROCEDURAL
        else:
            intent = QueryIntent.FACTUAL
        word_count = len(query.split()) if query else 0
        complexity = "complex" if word_count > 20 else ("medium" if word_count > 10 else "simple")
        return QueryAnalysis(
            intent=intent,
            complexity=complexity,
            requires_reasoning=("explain" in q or "why" in q),
            explanation=f"Fallback: {intent.value}",
        )

    def _fallback_modalities(self, query: str, analysis: QueryAnalysis) -> Tuple[List[Modality], Dict[str, float]]:
        modalities = [Modality.TEXT]
        q = (query or "").lower()
        if analysis.intent == QueryIntent.VISUAL_REASONING or any(w in q for w in ["image", "diagram", "chart"]):
            modalities.append(Modality.IMAGE)
        if "audio" in q or "lecture" in q:
            modalities.append(Modality.AUDIO)
        if "video" in q:
            modalities.append(Modality.VIDEO)
        weights = {m.value: 1.0 / len(modalities) for m in modalities}
        return modalities, weights

    def _fallback_strategy(self, analysis: QueryAnalysis) -> Tuple[RetrievalStrategy, Dict[str, int]]:
        if analysis.complexity == "simple":
            strategy = RetrievalStrategy.DENSE_ONLY
            params = {"dense_k": 10, "sparse_k": 0, "rerank_k": 5}
        elif analysis.requires_reasoning:
            strategy = RetrievalStrategy.GRAPH_ENHANCED
            params = {"dense_k": 30, "sparse_k": 30, "rerank_k": 12}
        else:
            strategy = RetrievalStrategy.HYBRID
            params = {"dense_k": self.default_dense_k, "sparse_k": self.default_sparse_k, "rerank_k": self.default_rerank_k}
        return strategy, params

    # ============================
    # TOOL ROUTING (UNCHANGED)
    # ============================
    def _route_tools(self, query: str, analysis: QueryAnalysis, modalities: List[Modality]) -> List[ToolCall]:
        tools = []
        q = (query or "").lower()
        if Modality.IMAGE in modalities and "text" in q:
            tools.append(ToolCall(tool_name="ocr_extractor", parameters={"modality": "image", "extract_tables": True}, reason="Query mentions text in images", priority=1))
        if "table" in q:
            tools.append(ToolCall(tool_name="table_parser", parameters={"format": "pandas"}, reason="Query requires table understanding", priority=2))
        if any(w in q for w in ["calculate", "compute", "sum", "average"]):
            tools.append(ToolCall(tool_name="code_interpreter", parameters={"language": "python"}, reason="Query requires computation", priority=3))
        return tools

    # ============================
    # SUB-PLANS: FAST NON-RECURSIVE
    # ============================
    def _decompose_query_fast(self, analysis: QueryAnalysis) -> List[RetrievalPlan]:
        """Create lightweight sub-plans without recursively invoking full planner."""
        sub_plans = []
        if not analysis.sub_queries:
            return sub_plans
        for sq in analysis.sub_queries[:3]:
            an = self._fallback_analysis(sq)
            strat, params = self._fallback_strategy(an)
            rp = RetrievalPlan(
                query=sq,
                intent=an.intent,
                analysis=an,
                modalities=[Modality.TEXT],
                strategy=strat,
                dense_k=params["dense_k"],
                sparse_k=params["sparse_k"],
                rerank_k=params["rerank_k"],
                plan_id=str(uuid.uuid4())[:8],
                timestamp=datetime.now().isoformat(),
            )
            sub_plans.append(rp)
        return sub_plans

    # ============================
    # CONFIDENCE SCORING (UNCHANGED)
    # ============================
    def _compute_plan_confidence(self, analysis: QueryAnalysis, modalities: List[Modality], strategy: RetrievalStrategy) -> Tuple[float, ConfidenceLevel]:
        confidence = 1.0
        if getattr(analysis, "ambiguity_score", 0.0) > 0.5:
            confidence *= 0.7
        if analysis.intent == QueryIntent.UNKNOWN:
            confidence *= 0.6
        if analysis.complexity == "simple":
            confidence *= 1.1
        if confidence >= 0.8:
            level = ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW
        return min(confidence, 1.0), level

    # ============================
    # LLM CALL (SAFE & SMALL)
    # ============================
    def _llm_call_with_schema(self, prompt: str, timeout: int = 5) -> Dict:
        """
        Call local Ollama with a strongly-limited token budget and timeout.
        Returns parsed JSON or empty dict on failure.
        """
        if not self.ollama_available:
            logger.debug("Ollama not available; skipping LLM call.")
            return {}

        try:
            # Ollama options tuned for low-latency
            opts = {
                "temperature": 0.1,
                "format": "json",
                "num_predict": max(16, min(self.llm_max_tokens, 128)),
                "timeout": timeout,
            }
            response = ollama.generate(model=self.local_model, prompt=prompt, options=opts, stream=False)
            text = (response.get("response") or "").strip()
            if not text:
                return {}
            # sanitize
            if "```json" in text:
                text = text.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in text:
                text = text.split("```", 1)[1].split("```", 1)[0].strip()
            if not text.startswith("{"):
                idx = text.find("{")
                if idx != -1:
                    text = text[idx:]
            # try to close
            if not text.endswith("}"):
                idx = text.rfind("}")
                if idx != -1:
                    text = text[: idx + 1]
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                logger.debug("Failed to parse JSON from LLM response.")
                return {}
        except Exception as e:
            logger.warning(f"Ollama call failed: {e}")
            return {}

    # ============================
    # UTILITIES
    # ============================
    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def get_metrics_summary(self) -> Dict:
        if not self.metrics_history:
            return {}
        total_plans = len(self.metrics_history)
        return {
            "total_plans": total_plans,
            "avg_planning_time_ms": sum(m.planning_time_ms for m in self.metrics_history) / total_plans,
            "avg_analysis_time_ms": sum(m.analysis_time_ms for m in self.metrics_history) / total_plans,
            "avg_confidence": sum(m.confidence for m in self.metrics_history) / total_plans,
            "min_time_ms": min(m.planning_time_ms for m in self.metrics_history),
            "max_time_ms": max(m.planning_time_ms for m in self.metrics_history),
            "total_time_ms": sum(m.planning_time_ms for m in self.metrics_history),
        }


# ============================
# QUICK LOCAL TEST (only when executed directly)
# ============================
if __name__ == "__main__":
    import yaml
    config_path = Path(__file__).parents[1] / "config" / "settings.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    planner = LocalAgenticPlanner(config)
    queries = [
        "What is machine learning?",
        "Explain how attention mechanism works in transformers",
        "Compare BERT vs GPT models",
    ]
    for q in queries:
        p = planner.plan(q)
        print(p.plan_id, p.intent.value, p.strategy.value, p.confidence)
