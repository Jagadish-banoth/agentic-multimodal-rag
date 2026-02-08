"""
planner/schemas.py
==================
Structured schemas for LLM-based agentic planning.

Used for function calling and type-safe plan generation.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Literal
from enum import Enum


# ============================
# ENUMS
# ============================

class QueryIntent(str, Enum):
    """Structured query intent classification."""
    FACTUAL = "factual"
    EXPLANATORY = "explanatory"
    COMPARATIVE = "comparative"
    VISUAL_REASONING = "visual_reasoning"
    AUDIO_REASONING = "audio_reasoning"
    VIDEO_REASONING = "video_reasoning"
    PROCEDURAL = "procedural"
    AGGREGATION = "aggregation"
    TEMPORAL = "temporal"
    UNKNOWN = "unknown"


class Modality(str, Enum):
    """Available modalities for retrieval."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class RetrievalStrategy(str, Enum):
    """Retrieval strategy selection."""
    DENSE_ONLY = "dense_only"
    SPARSE_ONLY = "sparse_only"
    HYBRID = "hybrid"
    GRAPH_ENHANCED = "graph_enhanced"
    MULTIMODAL_FUSION = "multimodal_fusion"


class ConfidenceLevel(str, Enum):
    """Confidence level for plan quality."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ============================
# DATACLASSES
# ============================

@dataclass
class QueryAnalysis:
    """
    Deep query analysis output from LLM.
    
    Used for understanding query characteristics before planning.
    """
    intent: QueryIntent
    complexity: Literal["simple", "medium", "complex"]
    domain: Optional[str] = None
    temporal_aspect: bool = False
    requires_reasoning: bool = False
    ambiguity_score: float = 0.0
    key_entities: List[str] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class ToolCall:
    """
    Represents a tool/function call decision.
    
    Used for dynamic routing to specialized retrievers.
    """
    tool_name: str
    parameters: Dict
    reason: str
    priority: int = 1


@dataclass
class RetrievalPlan:
    """
    Enhanced retrieval plan with LLM-generated decisions.
    
    Replaces the old keyword-based planner.
    """
    # Original query
    query: str
    
    # LLM-analyzed intent
    intent: QueryIntent
    analysis: QueryAnalysis
    
    # Modality selection
    modalities: List[Modality]
    
    # Strategy
    strategy: RetrievalStrategy
    
    # Dynamic parameter tuning
    dense_k: int
    sparse_k: int
    rerank_k: int
    
    # Tool calls
    tool_calls: List[ToolCall] = field(default_factory=list)
    
    # Multi-step decomposition
    sub_plans: List["RetrievalPlan"] = field(default_factory=list)
    
    # Confidence and reasoning
    confidence: float = 1.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH
    reasoning: str = ""
    
    # Control flow
    allow_retry: bool = True
    max_retries: int = 2
    timeout_seconds: int = 30
    
    # Observability
    plan_id: Optional[str] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dict for logging/storage."""
        data = asdict(self)
        # Convert enums to strings
        data["intent"] = self.intent.value
        data["modalities"] = [m.value for m in self.modalities]
        data["strategy"] = self.strategy.value
        data["confidence_level"] = self.confidence_level.value
        if self.analysis:
            data["analysis"]["intent"] = self.analysis.intent.value
        return data


@dataclass
class PlannerMetrics:
    """
    Observability metrics for planner performance.
    
    Used for monitoring and debugging.
    """
    plan_id: str
    query_length: int
    analysis_time_ms: float
    planning_time_ms: float
    total_time_ms: float
    llm_calls: int
    llm_tokens_used: int
    fallback_used: bool
    confidence: float
    intent: str
    strategy: str


# ============================
# FUNCTION CALLING SCHEMAS
# ============================

INTENT_CLASSIFICATION_SCHEMA = {
    "name": "classify_query_intent",
    "description": "Classify user query intent for optimal retrieval strategy",
    "parameters": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": [i.value for i in QueryIntent],
                "description": "Primary intent of the user query"
            },
            "complexity": {
                "type": "string",
                "enum": ["simple", "medium", "complex"],
                "description": "Query complexity level"
            },
            "domain": {
                "type": "string",
                "description": "Domain or topic area (e.g., 'medical', 'legal', 'technical')"
            },
            "temporal_aspect": {
                "type": "boolean",
                "description": "Whether query has temporal/time-based aspects"
            },
            "requires_reasoning": {
                "type": "boolean",
                "description": "Whether query requires multi-step reasoning"
            },
            "key_entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key entities mentioned in query"
            },
            "sub_queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Decomposed sub-queries for complex queries"
            },
            "explanation": {
                "type": "string",
                "description": "Brief explanation of the classification"
            }
        },
        "required": ["intent", "complexity", "explanation"]
    }
}

MODALITY_SELECTION_SCHEMA = {
    "name": "select_modalities",
    "description": "Select which modalities to use for retrieval",
    "parameters": {
        "type": "object",
        "properties": {
            "modalities": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [m.value for m in Modality]
                },
                "description": "List of modalities to retrieve from"
            },
            "primary_modality": {
                "type": "string",
                "enum": [m.value for m in Modality],
                "description": "Primary modality for this query"
            },
            "fusion_weights": {
                "type": "object",
                "description": "Weights for multimodal fusion (e.g., {'text': 0.6, 'image': 0.4})"
            },
            "reasoning": {
                "type": "string",
                "description": "Explanation for modality selection"
            }
        },
        "required": ["modalities", "reasoning"]
    }
}

STRATEGY_SELECTION_SCHEMA = {
    "name": "select_retrieval_strategy",
    "description": "Choose optimal retrieval strategy based on query analysis",
    "parameters": {
        "type": "object",
        "properties": {
            "strategy": {
                "type": "string",
                "enum": [s.value for s in RetrievalStrategy],
                "description": "Retrieval strategy to use"
            },
            "dense_k": {
                "type": "integer",
                "description": "Number of dense retrieval results",
                "minimum": 5,
                "maximum": 100
            },
            "sparse_k": {
                "type": "integer",
                "description": "Number of sparse retrieval results",
                "minimum": 5,
                "maximum": 100
            },
            "rerank_k": {
                "type": "integer",
                "description": "Number of results after reranking",
                "minimum": 3,
                "maximum": 50
            },
            "use_graph": {
                "type": "boolean",
                "description": "Whether to use knowledge graph traversal"
            },
            "reasoning": {
                "type": "string",
                "description": "Explanation for strategy selection"
            }
        },
        "required": ["strategy", "dense_k", "sparse_k", "rerank_k", "reasoning"]
    }
}
