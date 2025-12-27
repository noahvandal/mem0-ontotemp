"""
Bitemporal store using PostgreSQL with pgvector.

Provides immutable versioned storage with:
- Point-in-time queries (as_of)
- Graph reconstruction at any transaction time
- Vector similarity search on current or historical data
"""

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional
from uuid import UUID, uuid4

from mem0.temporal.config import TemporalStoreConfig
from mem0.temporal.stores.models import (
    GraphDelta,
    MemoryVersion,
    RelationshipVersion,
    SituationMatch,
    TemporalGraph,
    VersionHistory,
)

logger = logging.getLogger(__name__)


def _to_uuid(value) -> Optional[UUID]:
    """Safely convert a value to UUID, handling already-UUID objects."""
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    return UUID(value)


def _parse_embedding(value) -> Optional[List[float]]:
    """Parse embedding from database, handling various formats."""
    if value is None:
        return None
    # If it's already a list, return it
    if isinstance(value, list):
        return value
    # If it's a string (pgvector returns "[0.1,0.2,...]"), parse it
    if isinstance(value, str):
        # Remove brackets and split by comma
        cleaned = value.strip("[]")
        if not cleaned:
            return None
        return [float(x.strip()) for x in cleaned.split(",")]
    # Try to iterate if it has __iter__ (some pgvector types)
    try:
        return list(value)
    except (TypeError, ValueError):
        return None


def _embedding_to_sql(embedding: List[float]) -> str:
    """Convert embedding list to pgvector string format for SQL queries."""
    return "[" + ",".join(str(x) for x in embedding) + "]"


# SQL for creating tables
CREATE_TABLES_SQL = """
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Memory versions (immutable)
CREATE TABLE IF NOT EXISTS memory_versions (
    version_id UUID PRIMARY KEY,
    memory_id UUID NOT NULL,
    content TEXT NOT NULL,
    embedding vector({embedding_dims}),

    valid_from TIMESTAMPTZ NOT NULL,
    valid_to TIMESTAMPTZ,
    transaction_time TIMESTAMPTZ DEFAULT now(),
    superseded_at TIMESTAMPTZ,

    entity_type_id UUID,
    previous_version_id UUID,
    reasoning_trace_id UUID,
    context_snapshot JSONB,

    user_id TEXT,
    agent_id TEXT,
    run_id TEXT,
    metadata JSONB
);

-- Relationship versions (immutable)
CREATE TABLE IF NOT EXISTS relationship_versions (
    version_id UUID PRIMARY KEY,
    relationship_id UUID NOT NULL,
    source_memory_id UUID NOT NULL,
    target_memory_id UUID NOT NULL,
    relation_type_id UUID,
    relation_name TEXT NOT NULL,
    attributes JSONB DEFAULT '{{}}',
    confidence FLOAT DEFAULT 1.0,

    valid_from TIMESTAMPTZ NOT NULL,
    valid_to TIMESTAMPTZ,
    transaction_time TIMESTAMPTZ DEFAULT now(),
    superseded_at TIMESTAMPTZ,

    previous_version_id UUID,
    reasoning_trace_id UUID
);

-- Entity type versions (for ontology)
CREATE TABLE IF NOT EXISTS entity_type_versions (
    version_id UUID PRIMARY KEY,
    type_id UUID NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    observed_attributes JSONB DEFAULT '{{}}',
    type_embedding vector({embedding_dims}),

    transaction_time TIMESTAMPTZ DEFAULT now(),
    superseded_at TIMESTAMPTZ,

    instance_count INT DEFAULT 0,
    confidence FLOAT DEFAULT 0.5,
    derivation TEXT DEFAULT 'LLM_EXTRACTED'
);

-- Relation type versions (for ontology)
CREATE TABLE IF NOT EXISTS relation_type_versions (
    version_id UUID PRIMARY KEY,
    type_id UUID NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    source_type_ids UUID[] DEFAULT '{{}}',
    target_type_ids UUID[] DEFAULT '{{}}',
    is_symmetric BOOLEAN DEFAULT FALSE,
    is_transitive BOOLEAN DEFAULT FALSE,

    transaction_time TIMESTAMPTZ DEFAULT now(),
    superseded_at TIMESTAMPTZ,

    instance_count INT DEFAULT 0,
    confidence FLOAT DEFAULT 0.5
);

-- Reasoning traces
CREATE TABLE IF NOT EXISTS reasoning_traces (
    trace_id UUID PRIMARY KEY,
    snapshot_id UUID,
    steps JSONB DEFAULT '[]',
    goal TEXT,
    conclusion TEXT,
    retrieved_context JSONB,
    confidence FLOAT,
    uncertainties JSONB,
    pattern_embedding vector({embedding_dims}),
    timestamp TIMESTAMPTZ DEFAULT now()
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_memory_id ON memory_versions (memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_current ON memory_versions (memory_id) WHERE superseded_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_memory_bitemporal ON memory_versions (memory_id, transaction_time DESC, superseded_at);
CREATE INDEX IF NOT EXISTS idx_memory_user ON memory_versions (user_id) WHERE superseded_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_memory_agent ON memory_versions (agent_id) WHERE superseded_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_relationship_id ON relationship_versions (relationship_id);
CREATE INDEX IF NOT EXISTS idx_relationship_current ON relationship_versions (relationship_id) WHERE superseded_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_relationship_source ON relationship_versions (source_memory_id) WHERE superseded_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_relationship_target ON relationship_versions (target_memory_id) WHERE superseded_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_entity_type_id ON entity_type_versions (type_id);
CREATE INDEX IF NOT EXISTS idx_entity_type_current ON entity_type_versions (type_id) WHERE superseded_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_relation_type_id ON relation_type_versions (type_id);
CREATE INDEX IF NOT EXISTS idx_relation_type_current ON relation_type_versions (type_id) WHERE superseded_at IS NULL;

-- Vector indexes (IVFFlat for approximate nearest neighbor)
CREATE INDEX IF NOT EXISTS idx_memory_embedding ON memory_versions
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100)
    WHERE superseded_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_entity_type_embedding ON entity_type_versions
    USING ivfflat (type_embedding vector_cosine_ops)
    WITH (lists = 50)
    WHERE superseded_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_reasoning_embedding ON reasoning_traces
    USING ivfflat (pattern_embedding vector_cosine_ops)
    WITH (lists = 50);
"""


class BitemporalStore:
    """
    Bitemporal storage layer using PostgreSQL with pgvector.

    All operations are immutable - updates create new versions and supersede old ones.
    """

    def __init__(self, config: TemporalStoreConfig):
        """
        Initialize the bitemporal store.

        Args:
            config: Store configuration with connection string
        """
        self.config = config
        self._pool = None
        self._initialized = False

    def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is None:
            try:
                import psycopg_pool
                self._pool = psycopg_pool.ConnectionPool(
                    self.config.connection_string,
                    min_size=1,
                    max_size=self.config.pool_size,
                    timeout=self.config.pool_timeout,
                    open=True,
                )
            except ImportError:
                raise ImportError(
                    "psycopg and psycopg_pool are required for BitemporalStore. "
                    "Install with: pip install 'mem0ai[temporal]'"
                )
        return self._pool

    @contextmanager
    def _get_connection(self) -> Generator:
        """Get a connection from the pool."""
        pool = self._get_pool()
        with pool.connection() as conn:
            yield conn

    def initialize(self) -> None:
        """Create tables and indexes if they don't exist."""
        if self._initialized:
            return

        sql = CREATE_TABLES_SQL.format(embedding_dims=self.config.embedding_dims)

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()

        self._initialized = True
        logger.info("Bitemporal store initialized")

    def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    # =========================================================================
    # Memory Version Operations
    # =========================================================================

    def create_memory_version(
        self,
        memory_id: UUID,
        content: str,
        valid_from: datetime,
        embedding: Optional[List[float]] = None,
        entity_type_id: Optional[UUID] = None,
        previous_version_id: Optional[UUID] = None,
        reasoning_trace_id: Optional[UUID] = None,
        context_snapshot: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryVersion:
        """
        Create a new immutable memory version.

        Args:
            memory_id: Stable identity for the memory
            content: The memory content
            valid_from: When this fact became true
            embedding: Vector embedding (optional)
            entity_type_id: Link to ontology entity type
            previous_version_id: Version this supersedes (if update)
            reasoning_trace_id: Link to reasoning that created this
            context_snapshot: Context available when created
            user_id, agent_id, run_id: mem0 identifiers

        Returns:
            The created MemoryVersion
        """
        self.initialize()

        version = MemoryVersion(
            version_id=uuid4(),
            memory_id=memory_id,
            content=content,
            embedding=embedding,
            valid_from=valid_from,
            valid_to=None,
            transaction_time=datetime.now(),
            superseded_at=None,
            entity_type_id=entity_type_id,
            previous_version_id=previous_version_id,
            reasoning_trace_id=reasoning_trace_id,
            context_snapshot=context_snapshot,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            metadata=metadata,
        )

        sql = """
            INSERT INTO memory_versions (
                version_id, memory_id, content, embedding,
                valid_from, valid_to, transaction_time, superseded_at,
                entity_type_id, previous_version_id, reasoning_trace_id, context_snapshot,
                user_id, agent_id, run_id, metadata
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s
            )
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    str(version.version_id),
                    str(version.memory_id),
                    version.content,
                    version.embedding,
                    version.valid_from,
                    version.valid_to,
                    version.transaction_time,
                    version.superseded_at,
                    str(version.entity_type_id) if version.entity_type_id else None,
                    str(version.previous_version_id) if version.previous_version_id else None,
                    str(version.reasoning_trace_id) if version.reasoning_trace_id else None,
                    json.dumps(version.context_snapshot) if version.context_snapshot else None,
                    version.user_id,
                    version.agent_id,
                    version.run_id,
                    json.dumps(version.metadata) if version.metadata else None,
                ))
            conn.commit()

        logger.debug(f"Created memory version {version.version_id} for memory {memory_id}")
        return version

    def supersede_memory(self, version_id: UUID) -> None:
        """
        Mark a memory version as superseded.

        Args:
            version_id: The version to supersede
        """
        self.initialize()

        sql = """
            UPDATE memory_versions
            SET superseded_at = %s
            WHERE version_id = %s AND superseded_at IS NULL
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (datetime.now(), str(version_id)))
            conn.commit()

        logger.debug(f"Superseded memory version {version_id}")

    def get_current_version(self, memory_id: UUID) -> Optional[MemoryVersion]:
        """
        Get the current (non-superseded) version of a memory.

        Args:
            memory_id: The stable memory ID

        Returns:
            Current MemoryVersion or None if not found
        """
        self.initialize()

        sql = """
            SELECT * FROM memory_versions
            WHERE memory_id = %s AND superseded_at IS NULL
            ORDER BY transaction_time DESC
            LIMIT 1
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (str(memory_id),))
                row = cur.fetchone()
                if row:
                    return self._row_to_memory_version(row, cur.description)
        return None

    def get_version_as_of(
        self,
        memory_id: UUID,
        transaction_time: datetime,
    ) -> Optional[MemoryVersion]:
        """
        Get the version of a memory as it was at a specific transaction time.

        This is the core bitemporal query: "What did we believe at time T?"

        Args:
            memory_id: The stable memory ID
            transaction_time: The point in time to query

        Returns:
            The MemoryVersion that was current at that time, or None
        """
        self.initialize()

        sql = """
            SELECT * FROM memory_versions
            WHERE memory_id = %s
              AND transaction_time <= %s
              AND (superseded_at > %s OR superseded_at IS NULL)
            ORDER BY transaction_time DESC
            LIMIT 1
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (str(memory_id), transaction_time, transaction_time))
                row = cur.fetchone()
                if row:
                    return self._row_to_memory_version(row, cur.description)
        return None

    def get_version_history(self, memory_id: UUID) -> VersionHistory:
        """
        Get the complete version history for a memory.

        Args:
            memory_id: The stable memory ID

        Returns:
            VersionHistory with all versions, oldest first
        """
        self.initialize()

        sql = """
            SELECT * FROM memory_versions
            WHERE memory_id = %s
            ORDER BY transaction_time ASC
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (str(memory_id),))
                rows = cur.fetchall()
                versions = [self._row_to_memory_version(row, cur.description) for row in rows]

        return VersionHistory(memory_id=memory_id, versions=versions)

    # =========================================================================
    # Relationship Version Operations
    # =========================================================================

    def create_relationship_version(
        self,
        relationship_id: UUID,
        source_memory_id: UUID,
        target_memory_id: UUID,
        relation_name: str,
        valid_from: datetime,
        relation_type_id: Optional[UUID] = None,
        attributes: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        previous_version_id: Optional[UUID] = None,
        reasoning_trace_id: Optional[UUID] = None,
    ) -> RelationshipVersion:
        """Create a new immutable relationship version."""
        self.initialize()

        version = RelationshipVersion(
            version_id=uuid4(),
            relationship_id=relationship_id,
            source_memory_id=source_memory_id,
            target_memory_id=target_memory_id,
            relation_type_id=relation_type_id,
            relation_name=relation_name,
            attributes=attributes or {},
            confidence=confidence,
            valid_from=valid_from,
            valid_to=None,
            transaction_time=datetime.now(),
            superseded_at=None,
            previous_version_id=previous_version_id,
            reasoning_trace_id=reasoning_trace_id,
        )

        sql = """
            INSERT INTO relationship_versions (
                version_id, relationship_id, source_memory_id, target_memory_id,
                relation_type_id, relation_name, attributes, confidence,
                valid_from, valid_to, transaction_time, superseded_at,
                previous_version_id, reasoning_trace_id
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s
            )
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    str(version.version_id),
                    str(version.relationship_id),
                    str(version.source_memory_id),
                    str(version.target_memory_id),
                    str(version.relation_type_id) if version.relation_type_id else None,
                    version.relation_name,
                    json.dumps(version.attributes),
                    version.confidence,
                    version.valid_from,
                    version.valid_to,
                    version.transaction_time,
                    version.superseded_at,
                    str(version.previous_version_id) if version.previous_version_id else None,
                    str(version.reasoning_trace_id) if version.reasoning_trace_id else None,
                ))
            conn.commit()

        return version

    def supersede_relationship(self, version_id: UUID) -> None:
        """Mark a relationship version as superseded."""
        self.initialize()

        sql = """
            UPDATE relationship_versions
            SET superseded_at = %s
            WHERE version_id = %s AND superseded_at IS NULL
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (datetime.now(), str(version_id)))
            conn.commit()

    def get_current_relationship_version(self, relationship_id: UUID) -> Optional[RelationshipVersion]:
        """Get the current (non-superseded) version of a relationship."""
        self.initialize()

        sql = """
            SELECT * FROM relationship_versions
            WHERE relationship_id = %s AND superseded_at IS NULL
            ORDER BY transaction_time DESC
            LIMIT 1
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (str(relationship_id),))
                row = cur.fetchone()
                if row:
                    return self._row_to_relationship_version(row, cur.description)
        return None

    def get_relationship_as_of(
        self,
        relationship_id: UUID,
        transaction_time: datetime,
    ) -> Optional[RelationshipVersion]:
        """Get the relationship version that was current at a specific time."""
        self.initialize()

        sql = """
            SELECT * FROM relationship_versions
            WHERE relationship_id = %s
              AND transaction_time <= %s
              AND (superseded_at IS NULL OR superseded_at > %s)
            ORDER BY transaction_time DESC
            LIMIT 1
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (str(relationship_id), transaction_time, transaction_time))
                row = cur.fetchone()
                if row:
                    return self._row_to_relationship_version(row, cur.description)
        return None

    # =========================================================================
    # Graph Reconstruction
    # =========================================================================

    def reconstruct_graph_as_of(
        self,
        transaction_time: datetime,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> TemporalGraph:
        """
        Reconstruct the entire knowledge graph as it existed at a point in time.

        Args:
            transaction_time: The point in time to reconstruct
            user_id, agent_id, run_id: Optional filters

        Returns:
            TemporalGraph with all memories and relationships at that time
        """
        self.initialize()

        # Build filter conditions
        filters = []
        params = [transaction_time, transaction_time]

        if user_id:
            filters.append("user_id = %s")
            params.append(user_id)
        if agent_id:
            filters.append("agent_id = %s")
            params.append(agent_id)
        if run_id:
            filters.append("run_id = %s")
            params.append(run_id)

        filter_sql = " AND ".join(filters) if filters else "1=1"

        # Get memories
        memory_sql = f"""
            SELECT DISTINCT ON (memory_id) *
            FROM memory_versions
            WHERE transaction_time <= %s
              AND (superseded_at > %s OR superseded_at IS NULL)
              AND {filter_sql}
            ORDER BY memory_id, transaction_time DESC
        """

        # Get relationships
        rel_sql = """
            SELECT DISTINCT ON (relationship_id) *
            FROM relationship_versions
            WHERE transaction_time <= %s
              AND (superseded_at > %s OR superseded_at IS NULL)
            ORDER BY relationship_id, transaction_time DESC
        """

        memories = {}
        relationships = []

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Fetch memories
                cur.execute(memory_sql, params)
                for row in cur.fetchall():
                    mv = self._row_to_memory_version(row, cur.description)
                    memories[mv.memory_id] = mv

                # Fetch relationships (filter to only those with valid memories)
                cur.execute(rel_sql, [transaction_time, transaction_time])
                for row in cur.fetchall():
                    rv = self._row_to_relationship_version(row, cur.description)
                    # Only include if both memories exist in our filtered set
                    if rv.source_memory_id in memories and rv.target_memory_id in memories:
                        relationships.append(rv)

        return TemporalGraph(
            as_of_transaction_time=transaction_time,
            memories=memories,
            relationships=relationships,
            memory_count=len(memories),
            relationship_count=len(relationships),
        )

    # =========================================================================
    # Similarity Search
    # =========================================================================

    def search_similar_memories(
        self,
        embedding: List[float],
        k: int = 10,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        as_of: Optional[datetime] = None,
    ) -> List[MemoryVersion]:
        """
        Search for similar memories using vector similarity.

        Args:
            embedding: Query embedding
            k: Number of results to return
            user_id, agent_id, run_id: Optional filters
            as_of: If provided, search historical state at that time

        Returns:
            List of similar MemoryVersions, ordered by similarity
        """
        self.initialize()

        # Build filter conditions
        filters = []
        params = [embedding, k]

        if as_of:
            filters.append("transaction_time <= %s")
            filters.append("(superseded_at > %s OR superseded_at IS NULL)")
            params.insert(0, as_of)
            params.insert(1, as_of)
        else:
            filters.append("superseded_at IS NULL")

        if user_id:
            filters.append("user_id = %s")
            params.append(user_id)
        if agent_id:
            filters.append("agent_id = %s")
            params.append(agent_id)
        if run_id:
            filters.append("run_id = %s")
            params.append(run_id)

        filter_sql = " AND ".join(filters)

        embedding_str = _embedding_to_sql(embedding)

        sql = f"""
            SELECT *, embedding <=> %s::vector as distance
            FROM memory_versions
            WHERE {filter_sql}
              AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        # Adjust params order for the query
        query_params = []
        if as_of:
            query_params.extend([as_of, as_of])
        query_params.append(embedding_str)  # for distance calc
        if user_id:
            query_params.append(user_id)
        if agent_id:
            query_params.append(agent_id)
        if run_id:
            query_params.append(run_id)
        query_params.append(embedding_str)  # for ORDER BY
        query_params.append(k)

        results = []
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, query_params)
                for row in cur.fetchall():
                    # Last column is distance, skip it for model creation
                    mv = self._row_to_memory_version(row[:-1], cur.description[:-1])
                    results.append(mv)

        return results

    def search_similar_situations(
        self,
        embedding: List[float],
        k: int = 5,
    ) -> List[SituationMatch]:
        """
        Search for similar past situations (for intuition-like retrieval).

        Args:
            embedding: Query embedding representing current situation
            k: Number of results

        Returns:
            List of SituationMatch with similarity scores
        """
        self.initialize()

        embedding_str = _embedding_to_sql(embedding)

        sql = """
            SELECT *, embedding <=> %s::vector as distance
            FROM memory_versions
            WHERE superseded_at IS NULL
              AND embedding IS NOT NULL
              AND (reasoning_trace_id IS NOT NULL OR context_snapshot IS NOT NULL)
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        results = []
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (embedding_str, embedding_str, k))
                for row in cur.fetchall():
                    distance = row[-1]
                    mv = self._row_to_memory_version(row[:-1], cur.description[:-1])

                    # Convert distance to similarity (cosine distance to similarity)
                    similarity = 1 - distance

                    results.append(SituationMatch(
                        memory_version=mv,
                        similarity_score=similarity,
                        reasoning_trace_id=mv.reasoning_trace_id,
                        context_snapshot=mv.context_snapshot,
                    ))

        return results

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _row_to_memory_version(self, row: tuple, description) -> MemoryVersion:
        """Convert a database row to a MemoryVersion."""
        columns = [col.name for col in description]
        data = dict(zip(columns, row))

        return MemoryVersion(
            version_id=_to_uuid(data["version_id"]),
            memory_id=_to_uuid(data["memory_id"]),
            content=data["content"],
            embedding=_parse_embedding(data.get("embedding")),
            valid_from=data["valid_from"],
            valid_to=data.get("valid_to"),
            transaction_time=data["transaction_time"],
            superseded_at=data.get("superseded_at"),
            entity_type_id=_to_uuid(data.get("entity_type_id")),
            previous_version_id=_to_uuid(data.get("previous_version_id")),
            reasoning_trace_id=_to_uuid(data.get("reasoning_trace_id")),
            context_snapshot=data.get("context_snapshot"),
            user_id=data.get("user_id"),
            agent_id=data.get("agent_id"),
            run_id=data.get("run_id"),
            metadata=data.get("metadata"),
        )

    def _row_to_relationship_version(self, row: tuple, description) -> RelationshipVersion:
        """Convert a database row to a RelationshipVersion."""
        columns = [col.name for col in description]
        data = dict(zip(columns, row))

        return RelationshipVersion(
            version_id=_to_uuid(data["version_id"]),
            relationship_id=_to_uuid(data["relationship_id"]),
            source_memory_id=_to_uuid(data["source_memory_id"]),
            target_memory_id=_to_uuid(data["target_memory_id"]),
            relation_type_id=_to_uuid(data.get("relation_type_id")),
            relation_name=data["relation_name"],
            attributes=data.get("attributes") or {},
            confidence=data.get("confidence", 1.0),
            valid_from=data["valid_from"],
            valid_to=data.get("valid_to"),
            transaction_time=data["transaction_time"],
            superseded_at=data.get("superseded_at"),
            previous_version_id=_to_uuid(data.get("previous_version_id")),
            reasoning_trace_id=_to_uuid(data.get("reasoning_trace_id")),
        )

    def get_all_current_memories(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[MemoryVersion]:
        """Get all current (non-superseded) memories."""
        self.initialize()

        filters = ["superseded_at IS NULL"]
        params = []

        if user_id:
            filters.append("user_id = %s")
            params.append(user_id)
        if agent_id:
            filters.append("agent_id = %s")
            params.append(agent_id)
        if run_id:
            filters.append("run_id = %s")
            params.append(run_id)

        params.append(limit)
        filter_sql = " AND ".join(filters)

        sql = f"""
            SELECT * FROM memory_versions
            WHERE {filter_sql}
            ORDER BY transaction_time DESC
            LIMIT %s
        """

        results = []
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                for row in cur.fetchall():
                    results.append(self._row_to_memory_version(row, cur.description))

        return results
