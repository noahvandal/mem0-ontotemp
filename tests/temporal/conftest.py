"""
Pytest fixtures for temporal memory tests.

Uses testcontainers to spin up a real Postgres instance with pgvector.
"""

import pytest
from datetime import datetime
from uuid import uuid4


@pytest.fixture(scope="session")
def postgres_container():
    """
    Start a Postgres container with pgvector for the test session.

    Uses testcontainers to automatically manage the container lifecycle.
    """
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:
        pytest.skip("testcontainers not installed. Run: pip install testcontainers[postgres]")

    # Use pgvector-enabled Postgres image
    with PostgresContainer(
        image="pgvector/pgvector:pg16",
        username="test",
        password="test",
        dbname="test_temporal",
    ) as postgres:
        yield postgres


@pytest.fixture(scope="session")
def connection_string(postgres_container):
    """Get the connection string for the test Postgres container."""
    return postgres_container.get_connection_url()


@pytest.fixture
def temporal_store_config(connection_string):
    """Create a TemporalStoreConfig for testing."""
    from mem0.temporal.config import TemporalStoreConfig

    return TemporalStoreConfig(
        connection_string=connection_string.replace("postgresql+psycopg2://", "postgresql://"),
        embedding_dims=384,  # Smaller for testing
    )


@pytest.fixture
def bitemporal_store(temporal_store_config):
    """Create and initialize a BitemporalStore for testing."""
    from mem0.temporal.stores.bitemporal import BitemporalStore

    store = BitemporalStore(temporal_store_config)
    store.initialize()
    yield store
    store.close()


@pytest.fixture
def temporal_memory_config(temporal_store_config):
    """Create a TemporalMemoryConfig for testing."""
    from mem0.temporal.config import TemporalMemoryConfig

    return TemporalMemoryConfig(
        temporal_store=temporal_store_config,
        mem0_config=None,  # Will use defaults
    )


@pytest.fixture
def sample_memory_id():
    """Generate a sample memory ID."""
    return uuid4()


@pytest.fixture
def sample_embedding():
    """Generate a sample embedding (384 dimensions)."""
    import random
    return [random.random() for _ in range(384)]


@pytest.fixture
def sample_memory_version(sample_memory_id, sample_embedding):
    """Create a sample MemoryVersion for testing."""
    from mem0.temporal.stores.models import MemoryVersion

    return MemoryVersion(
        memory_id=sample_memory_id,
        content="Test memory content",
        embedding=sample_embedding,
        valid_from=datetime.now(),
        user_id="test_user",
    )


@pytest.fixture
def sample_reasoning_trace():
    """Create a sample ReasoningTrace for testing."""
    from mem0.temporal.reasoning.models import ReasoningTrace, ReasoningStep, StepType

    return ReasoningTrace(
        goal="Test goal",
        steps=[
            ReasoningStep(
                step_type=StepType.OBSERVATION,
                content="Observed something",
            ),
            ReasoningStep(
                step_type=StepType.INFERENCE,
                content="Inferred something",
            ),
            ReasoningStep(
                step_type=StepType.DECISION,
                content="Decided something",
            ),
        ],
        conclusion="Test conclusion",
        confidence=0.9,
    )


# Markers for slow tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
