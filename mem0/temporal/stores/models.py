"""
Pydantic models for bitemporal memory storage.

These models represent immutable versioned objects with bitemporal coordinates:
- valid_from/valid_to: When the fact was true in the world
- transaction_time/superseded_at: When we recorded/replaced this version
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class MemoryVersion(BaseModel):
    """
    An immutable version of a memory.

    Memories are never mutated - updates create new versions and supersede old ones.
    This enables point-in-time queries: "What did we believe at time T?"
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    version_id: UUID = Field(default_factory=uuid4, description="Unique ID for this version")
    memory_id: UUID = Field(..., description="Stable identity across versions")

    # Content
    content: str = Field(..., description="The memory content")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding")

    # Bitemporal coordinates
    valid_from: datetime = Field(..., description="When this fact became true in the world")
    valid_to: Optional[datetime] = Field(default=None, description="When this fact stopped being true (None = still valid)")
    transaction_time: datetime = Field(default_factory=datetime.now, description="When we recorded this version")
    superseded_at: Optional[datetime] = Field(default=None, description="When a newer version replaced this (None = current)")

    # Ontology link
    entity_type_id: Optional[UUID] = Field(default=None, description="Link to entity type in ontology")

    # Provenance
    previous_version_id: Optional[UUID] = Field(default=None, description="Link to the version this superseded")
    reasoning_trace_id: Optional[UUID] = Field(default=None, description="Link to reasoning that created/modified this")
    context_snapshot: Optional[Dict[str, Any]] = Field(default=None, description="Context available when created")

    # Identity scopes (for multi-tenant and session tracking)
    user_id: Optional[str] = Field(default=None, description="User-level scope")
    organization_id: Optional[str] = Field(default=None, description="Organization-level scope")
    session_id: Optional[str] = Field(default=None, description="Session-level scope")
    agent_id: Optional[str] = Field(default=None, description="Agent identifier")
    run_id: Optional[str] = Field(default=None, description="Run/conversation identifier")

    # Scope level for this memory
    scope_level: Optional[str] = Field(
        default="user",
        description="Memory scope: 'user', 'organization', 'session', or 'global'"
    )

    # Additional metadata
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class RelationshipVersion(BaseModel):
    """
    An immutable version of a relationship between memories.

    Like memories, relationships are versioned and never mutated.
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    version_id: UUID = Field(default_factory=uuid4, description="Unique ID for this version")
    relationship_id: UUID = Field(..., description="Stable identity across versions")

    # Relationship data
    source_memory_id: UUID = Field(..., description="Source memory in the relationship")
    target_memory_id: UUID = Field(..., description="Target memory in the relationship")
    relation_type_id: Optional[UUID] = Field(default=None, description="Link to relation type in ontology")
    relation_name: str = Field(..., description="Name/type of the relationship")

    # Attributes
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Relationship attributes")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")

    # Bitemporal coordinates
    valid_from: datetime = Field(..., description="When this relationship became true")
    valid_to: Optional[datetime] = Field(default=None, description="When this relationship ended")
    transaction_time: datetime = Field(default_factory=datetime.now, description="When we recorded this")
    superseded_at: Optional[datetime] = Field(default=None, description="When superseded by newer version")

    # Provenance
    previous_version_id: Optional[UUID] = Field(default=None)
    reasoning_trace_id: Optional[UUID] = Field(default=None)


class TemporalGraph(BaseModel):
    """
    A snapshot of the knowledge graph at a specific point in time.

    Used for graph reconstruction queries: "What did the graph look like at time T?"
    """

    # The point in time this graph represents
    as_of_transaction_time: datetime = Field(..., description="Transaction time this graph represents")
    as_of_valid_time: Optional[datetime] = Field(default=None, description="Valid time filter (if any)")

    # Graph contents
    memories: Dict[UUID, MemoryVersion] = Field(default_factory=dict, description="Memories keyed by memory_id")
    relationships: List[RelationshipVersion] = Field(default_factory=list, description="All relationships")

    # Metadata
    memory_count: int = Field(default=0)
    relationship_count: int = Field(default=0)

    def get_memory(self, memory_id: UUID) -> Optional[MemoryVersion]:
        """Get a memory by its stable ID."""
        return self.memories.get(memory_id)

    def get_relationships_for(self, memory_id: UUID) -> List[RelationshipVersion]:
        """Get all relationships involving a memory (as source or target)."""
        return [
            r for r in self.relationships
            if r.source_memory_id == memory_id or r.target_memory_id == memory_id
        ]

    def get_outgoing_relationships(self, memory_id: UUID) -> List[RelationshipVersion]:
        """Get relationships where memory is the source."""
        return [r for r in self.relationships if r.source_memory_id == memory_id]

    def get_incoming_relationships(self, memory_id: UUID) -> List[RelationshipVersion]:
        """Get relationships where memory is the target."""
        return [r for r in self.relationships if r.target_memory_id == memory_id]


class SituationMatch(BaseModel):
    """
    A match result from situation similarity search.

    Used for "intuition-like" retrieval: finding similar past decision points.
    """

    # The matched snapshot
    memory_version: MemoryVersion = Field(..., description="The matched memory version")

    # Similarity info
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score")

    # Context from that time
    reasoning_trace_id: Optional[UUID] = Field(default=None, description="Reasoning trace if this was a decision point")
    context_snapshot: Optional[Dict[str, Any]] = Field(default=None, description="Context at that time")

    # What happened after (for learning)
    outcome: Optional[str] = Field(default=None, description="What happened after this decision")
    outcome_success: Optional[bool] = Field(default=None, description="Was the outcome successful?")


class GraphDelta(BaseModel):
    """
    Changes between two graph states.

    Used for efficient storage (store deltas, not full snapshots) and
    for understanding what changed between two points in time.
    """

    # Time range
    from_time: datetime
    to_time: datetime

    # Memory changes
    memories_added: List[MemoryVersion] = Field(default_factory=list)
    memories_updated: List[MemoryVersion] = Field(default_factory=list)  # New versions
    memories_removed: List[UUID] = Field(default_factory=list)  # memory_ids

    # Relationship changes
    relationships_added: List[RelationshipVersion] = Field(default_factory=list)
    relationships_updated: List[RelationshipVersion] = Field(default_factory=list)
    relationships_removed: List[UUID] = Field(default_factory=list)  # relationship_ids

    @property
    def is_empty(self) -> bool:
        """Check if there were no changes."""
        return (
            not self.memories_added
            and not self.memories_updated
            and not self.memories_removed
            and not self.relationships_added
            and not self.relationships_updated
            and not self.relationships_removed
        )


class VersionHistory(BaseModel):
    """
    Complete history of versions for a memory.

    Ordered from oldest to newest.
    """

    memory_id: UUID = Field(..., description="The stable memory ID")
    versions: List[MemoryVersion] = Field(default_factory=list, description="All versions, oldest first")

    @property
    def current(self) -> Optional[MemoryVersion]:
        """Get the current (non-superseded) version."""
        for v in reversed(self.versions):
            if v.superseded_at is None:
                return v
        return None

    @property
    def first(self) -> Optional[MemoryVersion]:
        """Get the first version ever created."""
        return self.versions[0] if self.versions else None

    def at_time(self, transaction_time: datetime) -> Optional[MemoryVersion]:
        """Get the version that was current at a specific transaction time."""
        for v in reversed(self.versions):
            if v.transaction_time <= transaction_time:
                if v.superseded_at is None or v.superseded_at > transaction_time:
                    return v
        return None
