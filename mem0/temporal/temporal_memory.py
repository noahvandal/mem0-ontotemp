"""
Temporal Memory: A wrapper around mem0 that adds bitemporal versioning.

Provides:
- Immutable versioned memories (never mutate, always supersede)
- Bitemporal queries (valid time + transaction time)
- Auto-generated ontology that evolves with data
- Reasoning trace storage
- Situation embeddings for intuition-like retrieval
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from mem0.memory.main import Memory

from mem0.temporal.config import TemporalMemoryConfig
from mem0.temporal.ontology.manager import EmergentOntologyManager
from mem0.temporal.reasoning.models import DecisionExplanation, ReasoningStep, ReasoningTrace, StepType
from mem0.temporal.reasoning.tracer import ReasoningTracer
from mem0.temporal.stores.bitemporal import BitemporalStore
from mem0.temporal.stores.models import (
    MemoryVersion,
    RelationshipVersion,
    SituationMatch,
    TemporalGraph,
    VersionHistory,
)

logger = logging.getLogger(__name__)


class TemporalMemory:
    """
    Temporal extension of mem0 Memory with bitemporal versioning.

    Wraps mem0's Memory class to intercept operations and create immutable versions.
    Adds temporal queries that aren't available in standard mem0.

    Example usage:
        from mem0.temporal import TemporalMemory, TemporalMemoryConfig

        config = TemporalMemoryConfig.from_connection_string(
            "postgresql://user:pass@localhost:5432/mem0_temporal"
        )
        memory = TemporalMemory(config)

        # Standard mem0 operations (now with versioning)
        result = memory.add("User prefers dark mode", user_id="user_123")
        memory.update(result["results"][0]["id"], "User prefers light mode")

        # Temporal queries
        old_version = memory.as_of(result["results"][0]["id"], past_time)
        full_graph = memory.graph_as_of(past_time)
    """

    def __init__(self, config: TemporalMemoryConfig):
        """
        Initialize TemporalMemory.

        Args:
            config: Configuration including temporal store and mem0 settings
        """
        self.config = config

        # Initialize mem0 Memory for current-state operations
        if config.mem0_config:
            from mem0.configs.base import MemoryConfig
            mem0_cfg = MemoryConfig(**config.mem0_config)
            self.mem0 = Memory(mem0_cfg)
        else:
            self.mem0 = Memory()

        # Initialize bitemporal store
        self.temporal_store = BitemporalStore(config.temporal_store)
        self.temporal_store.initialize()

        # Initialize ontology manager
        self.ontology = EmergentOntologyManager(
            store=self.temporal_store,
            llm=self.mem0.llm if hasattr(self.mem0, 'llm') else None,
            embedder=self.mem0.embedding_model if hasattr(self.mem0, 'embedding_model') else None,
            type_similarity_threshold=config.ontology.type_similarity_threshold,
            use_llm_for_resolution=config.ontology.use_llm_for_type_resolution,
        )

        # Initialize reasoning tracer
        self.tracer = ReasoningTracer(
            store=self.temporal_store,
            embedder=self.mem0.embedding_model if hasattr(self.mem0, 'embedding_model') else None,
            max_steps_per_trace=config.reasoning.max_steps_per_trace,
        )

        logger.info("TemporalMemory initialized")

    # =========================================================================
    # Write Operations (intercept mem0, create versions)
    # =========================================================================

    def add(
        self,
        messages: Any,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        valid_from: Optional[datetime] = None,
        reasoning_trace_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Add memories with automatic versioning.

        Wraps mem0.add() and creates immutable versions of all created memories.

        Args:
            messages: Messages to add (same as mem0)
            user_id, agent_id, run_id: Scope identifiers
            valid_from: When this information became true (defaults to now)
            reasoning_trace_id: Link to reasoning that led to this addition
            metadata: Additional metadata
            **kwargs: Additional arguments passed to mem0

        Returns:
            Same format as mem0.add(), with added version information
        """
        # Let mem0 do extraction and storage (current state)
        result = self.mem0.add(
            messages,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            metadata=metadata,
            **kwargs,
        )

        valid_time = valid_from or datetime.now()
        context_snapshot = {"messages": messages} if isinstance(messages, list) else {"message": str(messages)}

        # Create immutable versions for each memory
        for memory in result.get("results", []):
            memory_id = UUID(memory["id"])

            # Resolve entity type (if ontology enabled)
            entity_type_id = None
            if self.config.ontology.auto_generate:
                try:
                    entity_type = self.ontology.resolve_entity_type(
                        entity_name=memory.get("type", "memory"),
                        attributes=memory.get("metadata", {}),
                        context=memory.get("memory", ""),
                    )
                    entity_type_id = entity_type.type_id
                except Exception as e:
                    logger.warning(f"Failed to resolve entity type: {e}")

            # Get embedding if available
            embedding = None
            if hasattr(self.mem0, 'embedding_model'):
                try:
                    embedding = self.mem0.embedding_model.embed(memory["memory"])
                except Exception as e:
                    logger.warning(f"Failed to get embedding: {e}")

            # Create version
            self.temporal_store.create_memory_version(
                memory_id=memory_id,
                content=memory["memory"],
                valid_from=valid_time,
                embedding=embedding,
                entity_type_id=entity_type_id,
                reasoning_trace_id=reasoning_trace_id,
                context_snapshot=context_snapshot,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                metadata=metadata,
            )

        # Create relationship versions (if graph enabled)
        for relation in result.get("relations", []):
            self._create_relationship_version(relation, valid_time, reasoning_trace_id)

        logger.debug(f"Added {len(result.get('results', []))} memories with versions")
        return result

    def update(
        self,
        memory_id: str,
        data: str,
        valid_from: Optional[datetime] = None,
        reason: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Update a memory by creating a new version (never mutate).

        Args:
            memory_id: ID of the memory to update
            data: New content
            valid_from: When this change became true (defaults to now)
            reason: Reason for the update (for provenance)
            **kwargs: Additional arguments passed to mem0

        Returns:
            Same format as mem0.update()
        """
        # Get current version before update
        current = self.temporal_store.get_current_version(UUID(memory_id))

        # Let mem0 update current state
        result = self.mem0.update(memory_id, data, **kwargs)

        valid_time = valid_from or datetime.now()

        # Supersede old version and create new
        if current:
            self.temporal_store.supersede_memory(current.version_id)

            # Get new embedding
            embedding = None
            if hasattr(self.mem0, 'embedding_model'):
                try:
                    embedding = self.mem0.embedding_model.embed(data)
                except Exception as e:
                    logger.warning(f"Failed to get embedding: {e}")

            self.temporal_store.create_memory_version(
                memory_id=UUID(memory_id),
                content=data,
                valid_from=valid_time,
                embedding=embedding,
                entity_type_id=current.entity_type_id,
                previous_version_id=current.version_id,
                context_snapshot={"reason": reason} if reason else None,
                user_id=current.user_id,
                agent_id=current.agent_id,
                run_id=current.run_id,
            )

        logger.debug(f"Updated memory {memory_id} with new version")
        return result

    def delete(
        self,
        memory_id: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Delete a memory by marking its version as superseded.

        The version history is preserved - this is a soft delete.

        Args:
            memory_id: ID of the memory to delete
            **kwargs: Additional arguments passed to mem0

        Returns:
            Same format as mem0.delete()
        """
        # Get current version
        current = self.temporal_store.get_current_version(UUID(memory_id))

        # Let mem0 delete from current state
        result = self.mem0.delete(memory_id, **kwargs)

        # Supersede the version (soft delete - history preserved)
        if current:
            self.temporal_store.supersede_memory(current.version_id)

        logger.debug(f"Deleted memory {memory_id} (version superseded)")
        return result

    # =========================================================================
    # Temporal Queries (not in mem0)
    # =========================================================================

    def as_of(
        self,
        memory_id: str,
        transaction_time: datetime,
    ) -> Optional[MemoryVersion]:
        """
        Get the version of a memory as it was at a specific transaction time.

        This is the core bitemporal query: "What did we believe at time T?"

        Args:
            memory_id: The memory ID
            transaction_time: The point in time to query

        Returns:
            The MemoryVersion that was current at that time, or None
        """
        return self.temporal_store.get_version_as_of(UUID(memory_id), transaction_time)

    def graph_as_of(
        self,
        transaction_time: datetime,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> TemporalGraph:
        """
        Reconstruct the entire knowledge graph at a point in time.

        Args:
            transaction_time: The point in time to reconstruct
            user_id, agent_id, run_id: Optional filters

        Returns:
            TemporalGraph with all memories and relationships at that time
        """
        return self.temporal_store.reconstruct_graph_as_of(
            transaction_time,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
        )

    def version_history(self, memory_id: str) -> VersionHistory:
        """
        Get the complete version history for a memory.

        Args:
            memory_id: The memory ID

        Returns:
            VersionHistory with all versions, oldest first
        """
        return self.temporal_store.get_version_history(UUID(memory_id))

    def explain_decision(
        self,
        reasoning_trace_id: str,
    ) -> DecisionExplanation:
        """
        Get full explanation of a past decision.

        Combines the reasoning trace with the knowledge state at the time.

        Args:
            reasoning_trace_id: ID of the reasoning trace

        Returns:
            DecisionExplanation with full context
        """
        trace = self.tracer.get_trace(UUID(reasoning_trace_id))
        if not trace:
            raise ValueError(f"Reasoning trace {reasoning_trace_id} not found")

        # Reconstruct knowledge state at decision time
        knowledge_state = None
        if trace.completed_at:
            graph = self.graph_as_of(trace.completed_at)
            knowledge_state = {
                "memories": {str(k): v.dict() for k, v in graph.memories.items()},
                "relationships": [r.dict() for r in graph.relationships],
            }

        return self.tracer.explain_decision(
            UUID(reasoning_trace_id),
            knowledge_state=knowledge_state,
        )

    def similar_situations(
        self,
        current_context: str,
        k: int = 5,
    ) -> List[SituationMatch]:
        """
        Find similar past decision points (intuition proxy).

        Args:
            current_context: Description of current situation
            k: Number of results

        Returns:
            List of SituationMatch with similarity scores
        """
        if not hasattr(self.mem0, 'embedding_model'):
            logger.warning("No embedding model available for situation similarity")
            return []

        embedding = self.mem0.embedding_model.embed(current_context)
        return self.temporal_store.search_similar_situations(embedding, k)

    # =========================================================================
    # Reasoning Support
    # =========================================================================

    def start_reasoning(
        self,
        goal: str,
        retrieved_context: Optional[List[str]] = None,
    ) -> str:
        """
        Begin tracking a reasoning trace.

        Args:
            goal: The goal being pursued
            retrieved_context: Context that was retrieved

        Returns:
            The trace_id as a string
        """
        if not self.config.reasoning.enabled:
            raise ValueError("Reasoning tracking is disabled in config")

        trace_id = self.tracer.start_trace(goal, retrieved_context)
        return str(trace_id)

    def add_reasoning_step(
        self,
        trace_id: str,
        step_type: str,
        content: str,
        evidence: Optional[List[str]] = None,
    ) -> None:
        """
        Add a step to the current reasoning trace.

        Args:
            trace_id: The trace to add to
            step_type: One of: observation, inference, hypothesis, decision, action, reflection
            content: Description of the step
            evidence: Supporting evidence
        """
        self.tracer.add_step(
            UUID(trace_id),
            StepType(step_type),
            content,
            supporting_evidence=evidence,
        )

    def complete_reasoning(
        self,
        trace_id: str,
        conclusion: str,
        confidence: float = 1.0,
        action_taken: Optional[str] = None,
    ) -> ReasoningTrace:
        """
        Complete and store the reasoning trace.

        Args:
            trace_id: The trace to complete
            conclusion: The conclusion reached
            confidence: Confidence in the reasoning
            action_taken: Action taken based on reasoning

        Returns:
            The completed ReasoningTrace
        """
        return self.tracer.complete_trace(
            UUID(trace_id),
            conclusion,
            confidence=confidence,
            action_taken=action_taken,
        )

    def record_outcome(
        self,
        trace_id: str,
        outcome: str,
        success: bool,
    ) -> None:
        """
        Record the outcome of a decision for learning.

        Args:
            trace_id: The reasoning trace
            outcome: What happened
            success: Whether it was successful
        """
        self.tracer.record_outcome(UUID(trace_id), outcome, success)

    # =========================================================================
    # Pass-through to mem0 (current state operations)
    # =========================================================================

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Search memories (delegates to mem0)."""
        return self.mem0.search(
            query,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            **kwargs,
        )

    def get(self, memory_id: str) -> Dict[str, Any]:
        """Get a memory by ID (delegates to mem0)."""
        return self.mem0.get(memory_id)

    def get_all(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Get all memories (delegates to mem0)."""
        return self.mem0.get_all(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            **kwargs,
        )

    def history(self, memory_id: str) -> List[Dict[str, Any]]:
        """Get memory history (delegates to mem0)."""
        return self.mem0.history(memory_id)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _create_relationship_version(
        self,
        relation: Dict[str, Any],
        valid_from: datetime,
        reasoning_trace_id: Optional[UUID],
    ) -> None:
        """Create a version for a relationship."""
        try:
            # Parse relation info (format depends on mem0's graph output)
            source_id = UUID(relation.get("source_id") or relation.get("source", {}).get("id", ""))
            target_id = UUID(relation.get("target_id") or relation.get("target", {}).get("id", ""))
            relation_name = relation.get("relationship") or relation.get("type", "related_to")

            # Resolve relation type
            relation_type_id = None
            if self.config.ontology.auto_generate:
                # Would need source/target type IDs from the memories
                pass

            self.temporal_store.create_relationship_version(
                relationship_id=uuid4(),
                source_memory_id=source_id,
                target_memory_id=target_id,
                relation_name=relation_name,
                valid_from=valid_from,
                relation_type_id=relation_type_id,
                reasoning_trace_id=reasoning_trace_id,
            )
        except Exception as e:
            logger.warning(f"Failed to create relationship version: {e}")

    def close(self) -> None:
        """Close connections and clean up resources."""
        self.temporal_store.close()
        logger.info("TemporalMemory closed")
