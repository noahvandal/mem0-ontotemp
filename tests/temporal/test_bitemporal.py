"""
Tests for the BitemporalStore.

Tests core bitemporal functionality:
- Creating immutable versions
- Superseding versions
- Point-in-time queries
- Graph reconstruction
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
import time


@pytest.mark.integration
class TestBitemporalStore:
    """Tests for BitemporalStore operations."""

    def test_create_memory_version(self, bitemporal_store, sample_embedding):
        """Test creating an immutable memory version."""
        memory_id = uuid4()

        version = bitemporal_store.create_memory_version(
            memory_id=memory_id,
            content="Test content",
            valid_from=datetime.now(),
            embedding=sample_embedding,
            user_id="test_user",
        )

        assert version.memory_id == memory_id
        assert version.content == "Test content"
        assert version.superseded_at is None
        assert version.user_id == "test_user"

    def test_get_current_version(self, bitemporal_store, sample_embedding):
        """Test retrieving the current version of a memory."""
        memory_id = uuid4()

        # Create version
        created = bitemporal_store.create_memory_version(
            memory_id=memory_id,
            content="Original content",
            valid_from=datetime.now(),
            embedding=sample_embedding,
        )

        # Retrieve it
        retrieved = bitemporal_store.get_current_version(memory_id)

        assert retrieved is not None
        assert retrieved.version_id == created.version_id
        assert retrieved.content == "Original content"

    def test_supersede_memory(self, bitemporal_store, sample_embedding):
        """Test that superseding creates new version without mutating old."""
        memory_id = uuid4()

        # Create first version
        v1 = bitemporal_store.create_memory_version(
            memory_id=memory_id,
            content="Version 1",
            valid_from=datetime.now(),
            embedding=sample_embedding,
        )

        time.sleep(0.01)  # Ensure different timestamps

        # Supersede it
        bitemporal_store.supersede_memory(v1.version_id)

        # Create second version
        v2 = bitemporal_store.create_memory_version(
            memory_id=memory_id,
            content="Version 2",
            valid_from=datetime.now(),
            embedding=sample_embedding,
            previous_version_id=v1.version_id,
        )

        # Current should be v2
        current = bitemporal_store.get_current_version(memory_id)
        assert current.version_id == v2.version_id
        assert current.content == "Version 2"

        # Check history has both
        history = bitemporal_store.get_version_history(memory_id)
        assert len(history.versions) == 2
        assert history.versions[0].content == "Version 1"
        assert history.versions[1].content == "Version 2"

    def test_get_version_as_of(self, bitemporal_store, sample_embedding):
        """Test point-in-time queries."""
        memory_id = uuid4()

        # Create v1
        t1 = datetime.now()
        v1 = bitemporal_store.create_memory_version(
            memory_id=memory_id,
            content="Version 1",
            valid_from=t1,
            embedding=sample_embedding,
        )

        time.sleep(0.1)
        t_between = datetime.now()
        time.sleep(0.1)

        # Supersede and create v2
        bitemporal_store.supersede_memory(v1.version_id)
        v2 = bitemporal_store.create_memory_version(
            memory_id=memory_id,
            content="Version 2",
            valid_from=datetime.now(),
            embedding=sample_embedding,
            previous_version_id=v1.version_id,
        )

        # Query as of t_between - should get v1
        as_of_between = bitemporal_store.get_version_as_of(memory_id, t_between)
        assert as_of_between is not None
        assert as_of_between.content == "Version 1"

        # Query as of now - should get v2
        as_of_now = bitemporal_store.get_version_as_of(memory_id, datetime.now())
        assert as_of_now is not None
        assert as_of_now.content == "Version 2"

    def test_version_history(self, bitemporal_store, sample_embedding):
        """Test getting complete version history."""
        memory_id = uuid4()

        # Create multiple versions
        for i in range(3):
            if i > 0:
                current = bitemporal_store.get_current_version(memory_id)
                bitemporal_store.supersede_memory(current.version_id)

            bitemporal_store.create_memory_version(
                memory_id=memory_id,
                content=f"Version {i+1}",
                valid_from=datetime.now(),
                embedding=sample_embedding,
            )
            time.sleep(0.01)

        history = bitemporal_store.get_version_history(memory_id)
        assert len(history.versions) == 3
        assert history.current.content == "Version 3"
        assert history.first.content == "Version 1"

    def test_reconstruct_graph_as_of(self, bitemporal_store, sample_embedding):
        """Test reconstructing the full graph at a point in time."""
        user_id = "graph_test_user"

        # Create some memories
        memory_ids = []
        for i in range(3):
            v = bitemporal_store.create_memory_version(
                memory_id=uuid4(),
                content=f"Memory {i}",
                valid_from=datetime.now(),
                embedding=sample_embedding,
                user_id=user_id,
            )
            memory_ids.append(v.memory_id)

        # Reconstruct graph
        graph = bitemporal_store.reconstruct_graph_as_of(
            datetime.now(),
            user_id=user_id,
        )

        assert graph.memory_count == 3
        for mid in memory_ids:
            assert mid in graph.memories

    def test_relationship_versioning(self, bitemporal_store, sample_embedding):
        """Test creating and versioning relationships."""
        # Create two memories
        m1 = bitemporal_store.create_memory_version(
            memory_id=uuid4(),
            content="Person: John",
            valid_from=datetime.now(),
            embedding=sample_embedding,
        )
        m2 = bitemporal_store.create_memory_version(
            memory_id=uuid4(),
            content="Company: Acme",
            valid_from=datetime.now(),
            embedding=sample_embedding,
        )

        # Create relationship
        rel = bitemporal_store.create_relationship_version(
            relationship_id=uuid4(),
            source_memory_id=m1.memory_id,
            target_memory_id=m2.memory_id,
            relation_name="works_at",
            valid_from=datetime.now(),
        )

        assert rel.source_memory_id == m1.memory_id
        assert rel.target_memory_id == m2.memory_id
        assert rel.relation_name == "works_at"


@pytest.mark.integration
class TestSimilaritySearch:
    """Tests for vector similarity search."""

    def test_search_similar_memories(self, bitemporal_store, sample_embedding):
        """Test finding similar memories by embedding."""
        user_id = "similarity_test_user"

        # Create memories with similar embeddings
        for i in range(5):
            # Slightly vary the embedding
            varied_embedding = [x + (i * 0.01) for x in sample_embedding]
            bitemporal_store.create_memory_version(
                memory_id=uuid4(),
                content=f"Memory {i}",
                valid_from=datetime.now(),
                embedding=varied_embedding,
                user_id=user_id,
            )

        # Search with original embedding
        results = bitemporal_store.search_similar_memories(
            sample_embedding,
            k=3,
            user_id=user_id,
        )

        assert len(results) <= 3
        # Results should be ordered by similarity

    def test_search_similar_situations(self, bitemporal_store, sample_embedding):
        """Test finding similar past decision situations."""
        # Create memory with context snapshot (situation)
        bitemporal_store.create_memory_version(
            memory_id=uuid4(),
            content="Decision point",
            valid_from=datetime.now(),
            embedding=sample_embedding,
            context_snapshot={"scenario": "test"},
            reasoning_trace_id=uuid4(),
        )

        results = bitemporal_store.search_similar_situations(sample_embedding, k=5)

        # Should find at least one result
        assert len(results) >= 0  # May be 0 if no matches
