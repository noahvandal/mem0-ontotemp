"""
Configuration for the Temporal Memory system.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class TemporalStoreConfig(BaseModel):
    """Configuration for the bitemporal Postgres store."""

    connection_string: str = Field(
        ...,
        description="PostgreSQL connection string (e.g., 'postgresql://user:pass@localhost:5432/dbname')",
    )
    embedding_dims: int = Field(
        default=1536,
        description="Dimension of vector embeddings (default 1536 for OpenAI)",
    )
    pool_size: int = Field(
        default=5,
        description="Connection pool size",
    )
    pool_timeout: int = Field(
        default=30,
        description="Connection pool timeout in seconds",
    )

    class Config:
        extra = "forbid"


class OntologyConfig(BaseModel):
    """Configuration for the ontology auto-generation system."""

    auto_generate: bool = Field(
        default=True,
        description="Whether to automatically generate entity/relation types from extracted data",
    )
    type_similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for matching new entities to existing types",
    )
    use_llm_for_type_resolution: bool = Field(
        default=True,
        description="Whether to use LLM to confirm type matches (more accurate but slower)",
    )
    min_instances_for_stable_type: int = Field(
        default=3,
        description="Minimum instances before a type is considered stable",
    )

    class Config:
        extra = "forbid"


class ReasoningConfig(BaseModel):
    """Configuration for reasoning trace tracking."""

    enabled: bool = Field(
        default=True,
        description="Whether to track reasoning traces",
    )
    embed_patterns: bool = Field(
        default=True,
        description="Whether to embed reasoning patterns for similarity search",
    )
    max_steps_per_trace: int = Field(
        default=100,
        description="Maximum reasoning steps to store per trace",
    )

    class Config:
        extra = "forbid"


class TemporalMemoryConfig(BaseModel):
    """
    Main configuration for TemporalMemory.

    Example usage:
        config = TemporalMemoryConfig(
            temporal_store=TemporalStoreConfig(
                connection_string="postgresql://user:pass@localhost:5432/mem0_temporal"
            ),
            mem0_config={
                "llm": {"provider": "openai", "config": {"model": "gpt-4"}},
                "embedder": {"provider": "openai"},
            }
        )
        memory = TemporalMemory(config)
    """

    # Temporal store configuration (required)
    temporal_store: TemporalStoreConfig = Field(
        ...,
        description="Configuration for the bitemporal Postgres store",
    )

    # mem0 configuration (optional - uses defaults if not provided)
    mem0_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration dict to pass to mem0 Memory class",
    )

    # Ontology configuration
    ontology: OntologyConfig = Field(
        default_factory=OntologyConfig,
        description="Configuration for ontology auto-generation",
    )

    # Reasoning configuration
    reasoning: ReasoningConfig = Field(
        default_factory=ReasoningConfig,
        description="Configuration for reasoning trace tracking",
    )

    # Versioning behavior
    create_version_on_search: bool = Field(
        default=False,
        description="Whether to create context snapshots on search operations (for full audit trail)",
    )

    class Config:
        extra = "forbid"

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        mem0_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "TemporalMemoryConfig":
        """
        Create config from just a connection string.

        Convenience method for simple setups.
        """
        return cls(
            temporal_store=TemporalStoreConfig(connection_string=connection_string),
            mem0_config=mem0_config,
            **kwargs,
        )
