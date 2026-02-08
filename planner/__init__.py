"""
Planner module for agentic RAG pipeline.

Provides intelligent query planning with Gemma 2B + GPU acceleration
and rule-based fallback for production reliability.
"""

from planner.agentic_planner import AgenticPlanner, create_planner, LocalAgenticPlanner
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

__all__ = [
    "AgenticPlanner",
    "LocalAgenticPlanner",
    "create_planner",
    "QueryIntent",
    "QueryAnalysis",
    "Modality",
    "RetrievalStrategy",
    "ConfidenceLevel",
    "RetrievalPlan",
    "ToolCall",
    "PlannerMetrics",
]
