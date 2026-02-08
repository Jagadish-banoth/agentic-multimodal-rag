"""
planner/agentic_planner.py
==========================
PRODUCTION-GRADE AGENTIC PLANNER WITH GEMMA 2B + GPU ACCELERATION

Features:
- Gemma 2B model for fast, intelligent planning (GPU-accelerated)
- Rule-based fallback when LLM unavailable or fails
- In-memory caching with TTL for instant repeated queries
- Optimized prompts for small LLMs
- Full observability and metrics
- Thread-safe and production-ready

Author: Expert RAG Engineer
"""

import json
import logging
import time
import uuid
import hashlib
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

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
)

logger = logging.getLogger(__name__)


# ============================
# OPTIMIZED PROMPTS FOR GEMMA 2B
# ============================

GEMMA_INTENT_PROMPT = '''Classify this query intent.
Query: "{query}"

Return JSON only:
{{"intent":"factual|explanatory|comparative|visual_reasoning|temporal|procedural","complexity":"simple|medium|complex","requires_reasoning":true|false}}'''

GEMMA_MODALITY_PROMPT = '''Select modalities for query.
Intent: {intent}
Query: "{query}"

Return JSON only:
{{"modalities":["text"],"weights":{{"text":1.0}}}}'''

GEMMA_STRATEGY_PROMPT = '''Select retrieval strategy.
Intent: {intent}, Complexity: {complexity}

Return JSON only:
{{"strategy":"hybrid|dense_only|sparse_only","dense_k":20,"sparse_k":20,"rerank_k":10}}'''


# ============================
# THREAD-SAFE LRU CACHE
# ============================

class ThreadSafePlanCache:
    """Thread-safe plan cache with TTL and LRU eviction."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self._store: Dict[str, Tuple[RetrievalPlan, float]] = {}
        self._lock = threading.RLock()
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._hits = 0
        self._misses = 0
    
    def _hash_key(self, query: str) -> str:
        """Create stable hash for query."""
        normalized = query.strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def get(self, query: str) -> Optional[RetrievalPlan]:
        key = self._hash_key(query)
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                self._misses += 1
                return None
            plan, ts = entry
            if time.time() - ts > self.ttl:
                del self._store[key]
                self._misses += 1
                return None
            self._hits += 1
            return plan
    
    def set(self, query: str, plan: RetrievalPlan) -> None:
        key = self._hash_key(query)
        with self._lock:
            # Evict oldest if at capacity
            if len(self._store) >= self.max_size:
                oldest_key = min(self._store.keys(), key=lambda k: self._store[k][1])
                del self._store[oldest_key]
            self._store[key] = (plan, time.time())
    
    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0
    
    @property
    def stats(self) -> Dict[str, int]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
                "size": len(self._store)
            }


# ============================
# GEMMA 2B AGENTIC PLANNER
# ============================

class AgenticPlanner:
    """
    Production-grade agentic planner using Gemma 2B with GPU acceleration.
    
    Design Philosophy:
    - Fast by default: Uses rule-based for simple queries
    - Smart when needed: Uses Gemma 2B for complex queries
    - Resilient: Automatic fallback to rule-based on any failure
    - Observable: Full metrics and logging
    
    Config (settings.yaml):
        planner:
            mode: "agentic"           # agentic | fast | rule_based
            local_model: "gemma2:2b"  # Primary model
            fallback_model: "gemma3:4b"  # Fallback model
            use_gpu: true             # GPU acceleration
            llm_timeout: 5            # Timeout in seconds
            llm_max_tokens: 64        # Max tokens for response
            cache_ttl_seconds: 3600   # Cache TTL
            min_complexity_for_llm: "medium"  # Only use LLM for medium+ complexity
    """
    
    def __init__(self, config: Dict):
        """Initialize agentic planner with config."""
        self.config = config
        
        # Extract planner config
        planner_cfg = config.get("planner", config.get("models", {}).get("planner", {}))
        
        # Mode: agentic (Gemma+fallback), fast (rule-based only)
        self.mode = planner_cfg.get("mode", "fast")
        
        # Model settings
        self.primary_model = planner_cfg.get("local_model", "gemma2:2b")
        self.fallback_model = planner_cfg.get("fallback_model", "gemma3:4b")
        self.use_gpu = planner_cfg.get("use_gpu", True)
        
        # Performance settings
        self.llm_timeout = planner_cfg.get("llm_timeout", 5)
        self.llm_max_tokens = planner_cfg.get("llm_max_tokens", 64)
        self.min_complexity_for_llm = planner_cfg.get("min_complexity_for_llm", "medium")
        
        # Retrieval defaults
        self.default_dense_k = config.get("retrieval", {}).get("dense_k", 100)
        self.default_sparse_k = config.get("retrieval", {}).get("sparse_k", 100)
        self.default_rerank_k = config.get("retrieval", {}).get("rerank_k", 50)
        
        # Agent settings
        agent_cfg = config.get("agent", {})
        self.confidence_threshold = agent_cfg.get("confidence_threshold", 0.7)
        self.max_attempts = agent_cfg.get("max_attempts", 2)
        
        # Cache
        cache_ttl = planner_cfg.get("cache_ttl_seconds", 3600)
        self.cache = ThreadSafePlanCache(max_size=1000, ttl_seconds=cache_ttl)
        
        # Thread pool for async LLM calls
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="planner")
        
        # Ollama availability check
        self.ollama_available = OLLAMA_AVAILABLE and ollama is not None
        self._model_available = False
        self._active_model = None
        
        if self.ollama_available:
            self._check_model_availability()
        
        # Metrics
        self.metrics_history: List[PlannerMetrics] = []
        self._lock = threading.Lock()
        
        logger.info(f"✓ AgenticPlanner initialized")
        logger.info(f"  Mode: {self.mode}")
        logger.info(f"  Primary model: {self.primary_model} (available: {self._model_available})")
        logger.info(f"  Active model: {self._active_model}")
        logger.info(f"  GPU: {self.use_gpu}")
        logger.info(f"  Timeout: {self.llm_timeout}s")
    
    def _auto_pull_model(self, model_name: str) -> bool:
        """Auto-pull a model from Ollama if not available."""
        try:
            logger.info(f"  ⬇️  Auto-pulling model {model_name}... (this may take a few minutes)")
            print(f"\n⬇️  Auto-pulling {model_name}... (first-time setup, please wait)")
            ollama.pull(model_name)
            logger.info(f"  ✓ Successfully pulled {model_name}")
            print(f"✓ Successfully pulled {model_name}\n")
            return True
        except Exception as e:
            logger.warning(f"  ✗ Failed to pull {model_name}: {e}")
            print(f"✗ Failed to pull {model_name}: {e}")
            return False

    def _check_model_availability(self) -> None:
        """Check if models are available and auto-pull if needed."""
        try:
            models = ollama.list()
            available_models = [m.get("name", m.get("model", "")) for m in models.get("models", [])]
            
            # Normalize model names (ollama list may show with/without :latest)
            available_normalized = []
            for m in available_models:
                available_normalized.append(m)
                if ":" in m:
                    available_normalized.append(m.split(":")[0])
            
            # Check primary model
            if self.primary_model in available_normalized or self.primary_model.split(":")[0] in available_normalized:
                self._model_available = True
                self._active_model = self.primary_model
                logger.info(f"  ✓ Primary model {self.primary_model} available")
                return
            
            # Check fallback model
            if self.fallback_model in available_normalized or self.fallback_model.split(":")[0] in available_normalized:
                self._model_available = True
                self._active_model = self.fallback_model
                logger.info(f"  ✓ Fallback model {self.fallback_model} available")
                return
            
            # Check for any gemma model
            for m in available_models:
                if "gemma" in m.lower():
                    self._model_available = True
                    self._active_model = m
                    logger.info(f"  ✓ Found gemma model: {m}")
                    return
            
            # No model found - try to auto-pull primary model
            logger.info(f"  ⚠️  No planner model found. Attempting auto-pull...")
            if self._auto_pull_model(self.primary_model):
                self._model_available = True
                self._active_model = self.primary_model
                return
            
            # If primary fails, try fallback
            if self._auto_pull_model(self.fallback_model):
                self._model_available = True
                self._active_model = self.fallback_model
                return
            
            logger.warning(f"  ✗ No suitable model found and auto-pull failed. Using rule-based fallback.")
            
        except Exception as e:
            logger.warning(f"  ✗ Could not check model availability: {e}")
    
    # ============================
    # PUBLIC API
    # ============================
    
    def plan(self, user_query: str, context: Optional[Dict] = None) -> RetrievalPlan:
        """
        Generate retrieval plan for a query.
        
        Flow:
        1. Check cache for instant response
        2. Quick rule-based analysis for complexity
        3. If simple query or LLM unavailable -> rule-based plan
        4. If complex query and LLM available -> Gemma 2B plan
        5. Fallback to rule-based on any failure
        
        Args:
            user_query: User's natural language query
            context: Optional context dict
        
        Returns:
            RetrievalPlan with all planning decisions
        """
        start_time = time.time()
        plan_id = str(uuid.uuid4())[:8]
        llm_calls = 0
        fallback_used = False
        
        # Step 1: Cache check
        cached = self.cache.get(user_query)
        if cached:
            logger.debug(f"[{plan_id}] Cache hit")
            self._record_metrics(plan_id, user_query, cached, start_time, 0, True)
            return cached
        
        logger.info(f"📋 Planning [{plan_id}]: {user_query[:80]}...")
        
        # Step 2: Quick complexity check (always rule-based)
        quick_analysis = self._rule_based_analysis(user_query)
        
        # Step 3: Decide planning approach
        use_llm = (
            self.mode == "agentic" and
            self._model_available and
            self.ollama_available and
            quick_analysis.complexity in ["medium", "complex"]
        )
        
        if use_llm:
            # Step 4: Gemma 2B planning with fallback
            try:
                analysis = self._llm_analysis(user_query)
                llm_calls += 1
                
                modalities, weights = self._llm_modalities(user_query, analysis)
                llm_calls += 1
                
                strategy, params = self._llm_strategy(user_query, analysis)
                llm_calls += 1
                
            except Exception as e:
                logger.warning(f"[{plan_id}] LLM planning failed: {e}. Using rule-based fallback.")
                analysis = quick_analysis
                modalities, weights = self._rule_based_modalities(user_query, analysis)
                strategy, params = self._rule_based_strategy(analysis)
                fallback_used = True
        else:
            # Step 5: Pure rule-based planning
            analysis = quick_analysis
            modalities, weights = self._rule_based_modalities(user_query, analysis)
            strategy, params = self._rule_based_strategy(analysis)
            fallback_used = True
        
        # Build tool calls
        tool_calls = self._build_tool_calls(user_query, analysis, modalities)
        
        # Compute confidence
        confidence, conf_level = self._compute_confidence(analysis, modalities, strategy, fallback_used)
        
        # Create plan
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
            sub_plans=[],
            confidence=confidence,
            confidence_level=conf_level,
            reasoning=analysis.explanation,
            allow_retry=confidence >= 0.5,
            max_retries=self.max_attempts,
            plan_id=plan_id,
            timestamp=datetime.now().isoformat(),
        )
        
        # Cache the plan
        self.cache.set(user_query, plan)
        
        # Record metrics
        self._record_metrics(plan_id, user_query, plan, start_time, llm_calls, fallback_used)
        
        total_ms = (time.time() - start_time) * 1000
        logger.info(f"✓ Plan [{plan_id}] in {total_ms:.0f}ms | "
                   f"intent={analysis.intent.value} | strategy={strategy.value} | "
                   f"confidence={confidence:.2f} | llm_calls={llm_calls}")
        
        return plan
    
    # ============================
    # LLM-BASED PLANNING (GEMMA 2B)
    # ============================
    
    def _llm_analysis(self, query: str) -> QueryAnalysis:
        """Analyze query using Gemma 2B."""
        prompt = GEMMA_INTENT_PROMPT.format(query=query)
        
        data = self._call_gemma(prompt)
        if not data:
            return self._rule_based_analysis(query)
        
        try:
            intent_str = data.get("intent", "factual")
            intent = QueryIntent(intent_str) if intent_str in [e.value for e in QueryIntent] else QueryIntent.FACTUAL
            
            return QueryAnalysis(
                intent=intent,
                complexity=data.get("complexity", "medium"),
                requires_reasoning=data.get("requires_reasoning", False),
                explanation=f"Gemma: {intent.value}",
            )
        except Exception as e:
            logger.warning(f"LLM analysis parse failed: {e}")
            return self._rule_based_analysis(query)
    
    def _llm_modalities(self, query: str, analysis: QueryAnalysis) -> Tuple[List[Modality], Dict[str, float]]:
        """Select modalities using Gemma 2B."""
        prompt = GEMMA_MODALITY_PROMPT.format(intent=analysis.intent.value, query=query)
        
        data = self._call_gemma(prompt)
        if not data or "modalities" not in data:
            return self._rule_based_modalities(query, analysis)
        
        try:
            mods = [Modality(m) for m in data.get("modalities", ["text"]) if m in [e.value for e in Modality]]
            if not mods:
                mods = [Modality.TEXT]
            weights = data.get("weights", {m.value: 1.0/len(mods) for m in mods})
            return mods, weights
        except Exception as e:
            logger.warning(f"LLM modality parse failed: {e}")
            return self._rule_based_modalities(query, analysis)
    
    def _llm_strategy(self, query: str, analysis: QueryAnalysis) -> Tuple[RetrievalStrategy, Dict[str, int]]:
        """Select strategy using Gemma 2B."""
        prompt = GEMMA_STRATEGY_PROMPT.format(intent=analysis.intent.value, complexity=analysis.complexity)
        
        data = self._call_gemma(prompt)
        if not data or "strategy" not in data:
            return self._rule_based_strategy(analysis)
        
        try:
            strat_str = data.get("strategy", "hybrid")
            strategy = RetrievalStrategy(strat_str) if strat_str in [e.value for e in RetrievalStrategy] else RetrievalStrategy.HYBRID
            
            params = {
                "dense_k": int(data.get("dense_k", self.default_dense_k)),
                "sparse_k": int(data.get("sparse_k", self.default_sparse_k)),
                "rerank_k": int(data.get("rerank_k", self.default_rerank_k)),
            }
            return strategy, params
        except Exception as e:
            logger.warning(f"LLM strategy parse failed: {e}")
            return self._rule_based_strategy(analysis)
    
    def _call_gemma(self, prompt: str) -> Dict:
        """
        Call Gemma 2B with GPU acceleration and timeout.
        
        Returns parsed JSON dict or empty dict on failure.
        """
        if not self.ollama_available or not self._model_available:
            return {}
        
        try:
            # Submit to thread pool with timeout
            future = self._executor.submit(self._gemma_generate, prompt)
            text = future.result(timeout=self.llm_timeout)
            
            if not text:
                return {}
            
            # Parse JSON from response
            return self._parse_json_response(text)
            
        except FuturesTimeoutError:
            logger.warning(f"Gemma call timed out after {self.llm_timeout}s")
            return {}
        except Exception as e:
            logger.warning(f"Gemma call failed: {e}")
            return {}
    
    def _gemma_generate(self, prompt: str) -> str:
        """Execute Gemma generation (runs in thread pool)."""
        try:
            response = ollama.generate(
                model=self._active_model,
                prompt=prompt,
                options={
                    "temperature": 0.1,
                    "format": "json",
                    "num_predict": self.llm_max_tokens,
                    "num_gpu": 99 if self.use_gpu else 0,  # Use all GPU layers
                    "num_thread": 4,
                },
                stream=False,
            )
            return (response.get("response") or "").strip()
        except Exception as e:
            logger.warning(f"Gemma generate error: {e}")
            return ""
    
    def _parse_json_response(self, text: str) -> Dict:
        """Parse JSON from LLM response with robust handling."""
        if not text:
            return {}
        
        # Remove code fences
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()
        
        # Find JSON object
        if not text.startswith("{"):
            idx = text.find("{")
            if idx != -1:
                text = text[idx:]
            else:
                return {}
        
        # Find closing brace
        if not text.endswith("}"):
            idx = text.rfind("}")
            if idx != -1:
                text = text[:idx + 1]
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}
    
    # ============================
    # RULE-BASED PLANNING (FALLBACK)
    # ============================
    
    def _rule_based_analysis(self, query: str) -> QueryAnalysis:
        """Fast rule-based query analysis."""
        q = query.lower()
        
        # Intent classification
        if any(w in q for w in ["compare", "difference", "vs", "versus", "contrast", "better"]):
            intent = QueryIntent.COMPARATIVE
        elif any(w in q for w in ["explain", "why", "how does", "describe", "what is"]):
            intent = QueryIntent.EXPLANATORY
        elif any(w in q for w in ["diagram", "image", "figure", "chart", "visual", "picture", "photo"]):
            intent = QueryIntent.VISUAL_REASONING
        elif any(w in q for w in ["when", "timeline", "history", "date", "year"]):
            intent = QueryIntent.TEMPORAL
        elif any(w in q for w in ["step", "process", "procedure", "how to", "guide", "tutorial"]):
            intent = QueryIntent.PROCEDURAL
        elif any(w in q for w in ["list", "all", "every", "summarize", "aggregate"]):
            intent = QueryIntent.AGGREGATION
        else:
            intent = QueryIntent.FACTUAL
        
        # Complexity estimation
        word_count = len(query.split())
        has_multiple_questions = query.count("?") > 1
        has_conjunctions = any(w in q for w in [" and ", " or ", " but ", " also "])
        
        if word_count > 25 or has_multiple_questions:
            complexity = "complex"
        elif word_count > 12 or has_conjunctions:
            complexity = "medium"
        else:
            complexity = "simple"
        
        return QueryAnalysis(
            intent=intent,
            complexity=complexity,
            requires_reasoning=intent in [QueryIntent.EXPLANATORY, QueryIntent.COMPARATIVE],
            explanation=f"Rule-based: {intent.value}",
        )
    
    def _rule_based_modalities(self, query: str, analysis: QueryAnalysis) -> Tuple[List[Modality], Dict[str, float]]:
        """Rule-based modality selection."""
        modalities = [Modality.TEXT]
        q = query.lower()
        
        if analysis.intent == QueryIntent.VISUAL_REASONING:
            modalities.append(Modality.IMAGE)
        elif any(w in q for w in ["image", "diagram", "chart", "figure", "picture", "photo", "screenshot"]):
            modalities.append(Modality.IMAGE)
        
        if any(w in q for w in ["audio", "sound", "voice", "speech", "podcast", "lecture"]):
            modalities.append(Modality.AUDIO)
        
        if any(w in q for w in ["video", "movie", "clip", "recording", "watch"]):
            modalities.append(Modality.VIDEO)
        
        weights = {m.value: 1.0 / len(modalities) for m in modalities}
        return modalities, weights
    
    def _rule_based_strategy(self, analysis: QueryAnalysis) -> Tuple[RetrievalStrategy, Dict[str, int]]:
        """Rule-based strategy selection."""
        if analysis.complexity == "simple":
            return RetrievalStrategy.DENSE_ONLY, {
                "dense_k": 20,
                "sparse_k": 0,
                "rerank_k": 10,
            }
        
        if analysis.requires_reasoning or analysis.intent == QueryIntent.COMPARATIVE:
            return RetrievalStrategy.HYBRID, {
                "dense_k": self.default_dense_k,
                "sparse_k": self.default_sparse_k,
                "rerank_k": self.default_rerank_k,
            }
        
        if analysis.intent in [QueryIntent.VISUAL_REASONING, QueryIntent.AUDIO_REASONING, QueryIntent.VIDEO_REASONING]:
            return RetrievalStrategy.MULTIMODAL_FUSION, {
                "dense_k": self.default_dense_k,
                "sparse_k": self.default_sparse_k // 2,
                "rerank_k": self.default_rerank_k,
            }
        
        # Default: hybrid
        return RetrievalStrategy.HYBRID, {
            "dense_k": self.default_dense_k,
            "sparse_k": self.default_sparse_k,
            "rerank_k": self.default_rerank_k,
        }
    
    # ============================
    # TOOL ROUTING
    # ============================
    
    def _build_tool_calls(self, query: str, analysis: QueryAnalysis, modalities: List[Modality]) -> List[ToolCall]:
        """Build tool calls based on query and analysis."""
        tools = []
        q = query.lower()
        
        if Modality.IMAGE in modalities and any(w in q for w in ["text", "ocr", "read"]):
            tools.append(ToolCall(
                tool_name="ocr_extractor",
                parameters={"modality": "image", "extract_tables": True},
                reason="Query mentions text in images",
                priority=1
            ))
        
        if any(w in q for w in ["table", "row", "column", "cell"]):
            tools.append(ToolCall(
                tool_name="table_parser",
                parameters={"format": "pandas"},
                reason="Query requires table understanding",
                priority=2
            ))
        
        if any(w in q for w in ["calculate", "compute", "sum", "average", "count", "total"]):
            tools.append(ToolCall(
                tool_name="code_interpreter",
                parameters={"language": "python"},
                reason="Query requires computation",
                priority=3
            ))
        
        return tools
    
    # ============================
    # CONFIDENCE SCORING
    # ============================
    
    def _compute_confidence(
        self,
        analysis: QueryAnalysis,
        modalities: List[Modality],
        strategy: RetrievalStrategy,
        fallback_used: bool
    ) -> Tuple[float, ConfidenceLevel]:
        """Compute plan confidence score."""
        confidence = 1.0
        
        # Penalize unknown intent
        if analysis.intent == QueryIntent.UNKNOWN:
            confidence *= 0.6
        
        # Boost simple queries
        if analysis.complexity == "simple":
            confidence *= 1.1
        
        # Slight penalty for fallback
        if fallback_used:
            confidence *= 0.95
        
        # Penalize complex multi-modal
        if len(modalities) > 2:
            confidence *= 0.9
        
        confidence = min(max(confidence, 0.0), 1.0)
        
        if confidence >= 0.8:
            level = ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW
        
        return confidence, level
    
    # ============================
    # METRICS
    # ============================
    
    def _record_metrics(
        self,
        plan_id: str,
        query: str,
        plan: RetrievalPlan,
        start_time: float,
        llm_calls: int,
        fallback_used: bool
    ) -> None:
        """Record planning metrics."""
        total_time = (time.time() - start_time) * 1000
        
        metrics = PlannerMetrics(
            plan_id=plan_id,
            query_length=len(query),
            analysis_time_ms=0.0,
            planning_time_ms=total_time,
            total_time_ms=total_time,
            llm_calls=llm_calls,
            llm_tokens_used=len(query) // 4 * max(llm_calls, 1),
            fallback_used=fallback_used,
            confidence=plan.confidence,
            intent=plan.intent.value,
            strategy=plan.strategy.value,
        )
        
        with self._lock:
            self.metrics_history.append(metrics)
            # Keep only last 1000 metrics
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
    
    def get_metrics_summary(self) -> Dict:
        """Get aggregated metrics summary."""
        with self._lock:
            if not self.metrics_history:
                return {"total_plans": 0}
            
            total = len(self.metrics_history)
            return {
                "total_plans": total,
                "avg_planning_time_ms": sum(m.planning_time_ms for m in self.metrics_history) / total,
                "avg_confidence": sum(m.confidence for m in self.metrics_history) / total,
                "min_time_ms": min(m.planning_time_ms for m in self.metrics_history),
                "max_time_ms": max(m.planning_time_ms for m in self.metrics_history),
                "llm_call_rate": sum(m.llm_calls for m in self.metrics_history) / total,
                "fallback_rate": sum(1 for m in self.metrics_history if m.fallback_used) / total,
                "cache_stats": self.cache.stats,
            }
    
    def clear_cache(self) -> None:
        """Clear the plan cache."""
        self.cache.clear()
        logger.info("Plan cache cleared")
    
    def shutdown(self) -> None:
        """Shutdown the planner cleanly."""
        self._executor.shutdown(wait=False)
        logger.info("AgenticPlanner shutdown complete")


# ============================
# FACTORY FUNCTION
# ============================

def create_planner(config: Dict) -> AgenticPlanner:
    """
    Factory function to create the appropriate planner.
    
    Args:
        config: System configuration dict
    
    Returns:
        AgenticPlanner instance
    """
    return AgenticPlanner(config)


# ============================
# BACKWARD COMPATIBILITY ALIAS
# ============================

LocalAgenticPlanner = AgenticPlanner


# ============================
# MAIN (TESTING)
# ============================

if __name__ == "__main__":
    import yaml
    
    # Load config
    config_path = Path(__file__).parents[1] / "config" / "settings.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Override to agentic mode for testing
    if "planner" not in config:
        config["planner"] = {}
    config["planner"]["mode"] = "agentic"
    config["planner"]["local_model"] = "gemma2:2b"
    config["planner"]["use_gpu"] = True
    
    # Create planner
    planner = AgenticPlanner(config)
    
    print("\n" + "=" * 80)
    print("AGENTIC PLANNER TEST (Gemma 2B + GPU)")
    print("=" * 80)
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "Explain how attention mechanism works in transformers",
        "Compare BERT vs GPT models in terms of architecture and use cases",
        "Show me the diagram of neural network layers",
        "How does fiverr protect freelancers' personal information?",
        "What are the steps to train a computer vision model?",
        "List all pre-trained models available for natural language processing",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}] Query: {query}")
        print("-" * 60)
        
        plan = planner.plan(query)
        
        print(f"  Plan ID: {plan.plan_id}")
        print(f"  Intent: {plan.intent.value}")
        print(f"  Complexity: {plan.analysis.complexity}")
        print(f"  Strategy: {plan.strategy.value}")
        print(f"  Modalities: {[m.value for m in plan.modalities]}")
        print(f"  Confidence: {plan.confidence:.2f} ({plan.confidence_level.value})")
        print(f"  Reasoning: {plan.reasoning}")
    
    # Print metrics
    print(f"\n{'=' * 80}")
    print("METRICS SUMMARY")
    print(f"{'=' * 80}")
    metrics = planner.get_metrics_summary()
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    planner.shutdown()
