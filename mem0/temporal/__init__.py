"""
Temporal Memory Extension for mem0.

Provides XTDB-like bitemporal capabilities:
- Immutable versioned memories (never mutate, always supersede)
- Bitemporal queries (valid time + transaction time)
- Auto-generated ontology that evolves with data
- Reasoning trace storage
- Situation embeddings for intuition-like retrieval
"""

from mem0.temporal.temporal_memory import TemporalMemory
from mem0.temporal.config import TemporalMemoryConfig, TemporalStoreConfig
from mem0.temporal.stores.models import (
    MemoryVersion,
    RelationshipVersion,
    TemporalGraph,
    SituationMatch,
)
from mem0.temporal.ontology.types import EntityTypeVersion, RelationTypeVersion
from mem0.temporal.reasoning.models import ReasoningTrace, ReasoningStep

__all__ = [
    "TemporalMemory",
    "TemporalMemoryConfig",
    "TemporalStoreConfig",
    "MemoryVersion",
    "RelationshipVersion",
    "TemporalGraph",
    "SituationMatch",
    "EntityTypeVersion",
    "RelationTypeVersion",
    "ReasoningTrace",
    "ReasoningStep",
]
