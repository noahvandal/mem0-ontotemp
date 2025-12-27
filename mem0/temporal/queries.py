"""
Temporal query interface for advanced bitemporal queries.

Provides a clean query API separate from the main TemporalMemory class
for complex temporal operations.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from mem0.temporal.stores.bitemporal import BitemporalStore
from mem0.temporal.stores.models import (
    GraphDelta,
    MemoryVersion,
    RelationshipVersion,
    SituationMatch,
    TemporalGraph,
    VersionHistory,
)


@dataclass
class TimeRange:
    """A range of time for queries."""

    start: datetime
    end: datetime

    def contains(self, time: datetime) -> bool:
        """Check if a time is within this range."""
        return self.start <= time <= self.end

    @classmethod
    def last_hours(cls, hours: int) -> "TimeRange":
        """Create a range for the last N hours."""
        end = datetime.now()
        start = end - timedelta(hours=hours)
        return cls(start=start, end=end)

    @classmethod
    def last_days(cls, days: int) -> "TimeRange":
        """Create a range for the last N days."""
        end = datetime.now()
        start = end - timedelta(days=days)
        return cls(start=start, end=end)


class TemporalQueryBuilder:
    """
    Fluent query builder for temporal queries.

    Example:
        results = (TemporalQueryBuilder(store)
            .for_user("user_123")
            .as_of(yesterday)
            .with_type("person")
            .limit(10)
            .execute())
    """

    def __init__(self, store: BitemporalStore):
        self.store = store
        self._user_id: Optional[str] = None
        self._agent_id: Optional[str] = None
        self._run_id: Optional[str] = None
        self._transaction_time: Optional[datetime] = None
        self._valid_time: Optional[datetime] = None
        self._entity_type_id: Optional[UUID] = None
        self._embedding: Optional[List[float]] = None
        self._limit: int = 100

    def for_user(self, user_id: str) -> "TemporalQueryBuilder":
        """Filter by user ID."""
        self._user_id = user_id
        return self

    def for_agent(self, agent_id: str) -> "TemporalQueryBuilder":
        """Filter by agent ID."""
        self._agent_id = agent_id
        return self

    def for_run(self, run_id: str) -> "TemporalQueryBuilder":
        """Filter by run ID."""
        self._run_id = run_id
        return self

    def as_of(self, transaction_time: datetime) -> "TemporalQueryBuilder":
        """Query as of a specific transaction time."""
        self._transaction_time = transaction_time
        return self

    def valid_at(self, valid_time: datetime) -> "TemporalQueryBuilder":
        """Filter by valid time."""
        self._valid_time = valid_time
        return self

    def with_type(self, entity_type_id: UUID) -> "TemporalQueryBuilder":
        """Filter by entity type."""
        self._entity_type_id = entity_type_id
        return self

    def similar_to(self, embedding: List[float]) -> "TemporalQueryBuilder":
        """Find similar memories by embedding."""
        self._embedding = embedding
        return self

    def limit(self, n: int) -> "TemporalQueryBuilder":
        """Limit results."""
        self._limit = n
        return self

    def execute(self) -> List[MemoryVersion]:
        """Execute the query and return results."""
        if self._embedding:
            return self.store.search_similar_memories(
                self._embedding,
                k=self._limit,
                user_id=self._user_id,
                agent_id=self._agent_id,
                run_id=self._run_id,
                as_of=self._transaction_time,
            )
        else:
            return self.store.get_all_current_memories(
                user_id=self._user_id,
                agent_id=self._agent_id,
                run_id=self._run_id,
                limit=self._limit,
            )

    def execute_graph(self) -> TemporalGraph:
        """Execute the query and return a graph."""
        transaction_time = self._transaction_time or datetime.now()
        return self.store.reconstruct_graph_as_of(
            transaction_time,
            user_id=self._user_id,
            agent_id=self._agent_id,
            run_id=self._run_id,
        )


class TemporalQueries:
    """
    Collection of temporal query operations.

    Provides high-level query methods for common temporal operations.
    """

    def __init__(self, store: BitemporalStore):
        self.store = store

    def query(self) -> TemporalQueryBuilder:
        """Start building a query."""
        return TemporalQueryBuilder(self.store)

    # =========================================================================
    # Point-in-Time Queries
    # =========================================================================

    def at_time(
        self,
        memory_id: UUID,
        transaction_time: datetime,
    ) -> Optional[MemoryVersion]:
        """
        Get a memory as it was at a specific time.

        This answers: "What did we believe about this memory at time T?"
        """
        return self.store.get_version_as_of(memory_id, transaction_time)

    def graph_at_time(
        self,
        transaction_time: datetime,
        **filters,
    ) -> TemporalGraph:
        """
        Reconstruct the full graph at a specific time.

        This answers: "What was the complete knowledge state at time T?"
        """
        return self.store.reconstruct_graph_as_of(transaction_time, **filters)

    # =========================================================================
    # Range Queries
    # =========================================================================

    def changes_in_range(
        self,
        memory_id: UUID,
        start_time: datetime,
        end_time: datetime,
    ) -> List[MemoryVersion]:
        """
        Get all versions of a memory within a time range.

        This answers: "How did this memory change between T1 and T2?"
        """
        history = self.store.get_version_history(memory_id)
        return [
            v for v in history.versions
            if start_time <= v.transaction_time <= end_time
        ]

    def graph_delta(
        self,
        start_time: datetime,
        end_time: datetime,
        **filters,
    ) -> GraphDelta:
        """
        Compute the difference between two graph states.

        This answers: "What changed in the graph between T1 and T2?"
        """
        graph_start = self.store.reconstruct_graph_as_of(start_time, **filters)
        graph_end = self.store.reconstruct_graph_as_of(end_time, **filters)

        # Compute deltas
        memories_added = []
        memories_updated = []
        memories_removed = []

        start_ids = set(graph_start.memories.keys())
        end_ids = set(graph_end.memories.keys())

        # Added
        for memory_id in end_ids - start_ids:
            memories_added.append(graph_end.memories[memory_id])

        # Removed
        memories_removed = list(start_ids - end_ids)

        # Updated (same ID, different content)
        for memory_id in start_ids & end_ids:
            old = graph_start.memories[memory_id]
            new = graph_end.memories[memory_id]
            if old.content != new.content:
                memories_updated.append(new)

        # Same for relationships
        relationships_added = []
        relationships_removed = []

        start_rels = {r.relationship_id for r in graph_start.relationships}
        end_rels = {r.relationship_id for r in graph_end.relationships}

        for rel in graph_end.relationships:
            if rel.relationship_id not in start_rels:
                relationships_added.append(rel)

        relationships_removed = list(start_rels - end_rels)

        return GraphDelta(
            from_time=start_time,
            to_time=end_time,
            memories_added=memories_added,
            memories_updated=memories_updated,
            memories_removed=memories_removed,
            relationships_added=relationships_added,
            relationships_removed=relationships_removed,
        )

    # =========================================================================
    # Similarity Queries
    # =========================================================================

    def similar_memories(
        self,
        embedding: List[float],
        k: int = 10,
        as_of: Optional[datetime] = None,
        **filters,
    ) -> List[MemoryVersion]:
        """
        Find memories similar to a query embedding.

        Optionally search historical state at a specific time.
        """
        return self.store.search_similar_memories(
            embedding,
            k=k,
            as_of=as_of,
            **filters,
        )

    def similar_situations(
        self,
        embedding: List[float],
        k: int = 5,
    ) -> List[SituationMatch]:
        """
        Find similar past decision situations.

        This is the "intuition proxy" - finding times when we faced
        similar situations and what we did.
        """
        return self.store.search_similar_situations(embedding, k)

    # =========================================================================
    # History Queries
    # =========================================================================

    def full_history(self, memory_id: UUID) -> VersionHistory:
        """
        Get complete version history for a memory.

        Returns all versions from creation to present.
        """
        return self.store.get_version_history(memory_id)

    def first_version(self, memory_id: UUID) -> Optional[MemoryVersion]:
        """Get the first (original) version of a memory."""
        history = self.store.get_version_history(memory_id)
        return history.first

    def current_version(self, memory_id: UUID) -> Optional[MemoryVersion]:
        """Get the current (latest non-superseded) version."""
        return self.store.get_current_version(memory_id)

    def version_at(
        self,
        memory_id: UUID,
        transaction_time: datetime,
    ) -> Optional[MemoryVersion]:
        """Get the version that was current at a specific time."""
        history = self.store.get_version_history(memory_id)
        return history.at_time(transaction_time)

    # =========================================================================
    # Aggregate Queries
    # =========================================================================

    def count_versions(
        self,
        memory_id: UUID,
        time_range: Optional[TimeRange] = None,
    ) -> int:
        """Count versions of a memory, optionally within a time range."""
        history = self.store.get_version_history(memory_id)

        if time_range:
            return len([
                v for v in history.versions
                if time_range.contains(v.transaction_time)
            ])
        return len(history.versions)

    def most_changed_memories(
        self,
        time_range: TimeRange,
        limit: int = 10,
        **filters,
    ) -> List[Tuple[UUID, int]]:
        """
        Find memories with the most version changes in a time range.

        Returns list of (memory_id, change_count) tuples.
        """
        # This would need a more efficient implementation with SQL aggregation
        # For now, a simple implementation
        graph = self.store.reconstruct_graph_as_of(time_range.end, **filters)
        changes = []

        for memory_id in graph.memories:
            count = self.count_versions(memory_id, time_range)
            if count > 1:  # Only include if there were changes
                changes.append((memory_id, count))

        changes.sort(key=lambda x: x[1], reverse=True)
        return changes[:limit]


def compare_graphs(
    graph1: TemporalGraph,
    graph2: TemporalGraph,
) -> Dict[str, Any]:
    """
    Compare two temporal graphs and return a summary of differences.

    Args:
        graph1: First graph (typically earlier)
        graph2: Second graph (typically later)

    Returns:
        Summary dict with differences
    """
    ids1 = set(graph1.memories.keys())
    ids2 = set(graph2.memories.keys())

    added = ids2 - ids1
    removed = ids1 - ids2
    common = ids1 & ids2

    # Check for content changes in common memories
    changed = []
    for memory_id in common:
        if graph1.memories[memory_id].content != graph2.memories[memory_id].content:
            changed.append(memory_id)

    # Relationship changes
    rels1 = {r.relationship_id for r in graph1.relationships}
    rels2 = {r.relationship_id for r in graph2.relationships}

    return {
        "memories_added": len(added),
        "memories_removed": len(removed),
        "memories_changed": len(changed),
        "memories_unchanged": len(common) - len(changed),
        "relationships_added": len(rels2 - rels1),
        "relationships_removed": len(rels1 - rels2),
        "added_memory_ids": list(added),
        "removed_memory_ids": list(removed),
        "changed_memory_ids": changed,
    }
