"""
Reasoning Tracer for capturing and querying decision-making processes.

Provides:
- Trace lifecycle management (start, add steps, complete)
- Persistence to the bitemporal store
- Similarity search for finding similar past reasoning
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from mem0.temporal.reasoning.models import (
    DecisionExplanation,
    ReasoningStep,
    ReasoningTrace,
    StepType,
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
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        cleaned = value.strip("[]")
        if not cleaned:
            return None
        return [float(x.strip()) for x in cleaned.split(",")]
    try:
        return list(value)
    except (TypeError, ValueError):
        return None


class ReasoningTracer:
    """
    Manages reasoning traces for temporal memory.

    Usage:
        tracer = ReasoningTracer(store, embedder)

        # Start a trace
        trace_id = tracer.start_trace("Decide which action to take")

        # Add reasoning steps
        tracer.add_step(trace_id, StepType.OBSERVATION, "User requested X")
        tracer.add_step(trace_id, StepType.INFERENCE, "This implies Y")
        tracer.add_step(trace_id, StepType.DECISION, "Best action is Z")

        # Complete the trace
        trace = tracer.complete_trace(trace_id, "Decision: do Z", confidence=0.9)

        # Later, find similar reasoning
        similar = tracer.find_similar_reasoning(current_pattern_embedding)
    """

    def __init__(
        self,
        store,  # BitemporalStore
        embedder=None,  # For pattern embeddings
        max_steps_per_trace: int = 100,
    ):
        """
        Initialize the reasoning tracer.

        Args:
            store: BitemporalStore for persistence
            embedder: Optional embedder for pattern similarity search
            max_steps_per_trace: Maximum steps allowed per trace
        """
        self.store = store
        self.embedder = embedder
        self.max_steps_per_trace = max_steps_per_trace

        # In-progress traces (not yet persisted)
        self._active_traces: Dict[UUID, ReasoningTrace] = {}

    def start_trace(
        self,
        goal: str,
        retrieved_context: Optional[List[str]] = None,
        retrieved_memory_ids: Optional[List[UUID]] = None,
    ) -> UUID:
        """
        Start a new reasoning trace.

        Args:
            goal: The goal being pursued
            retrieved_context: Context that was retrieved for this reasoning
            retrieved_memory_ids: IDs of memories that were retrieved

        Returns:
            The trace_id for the new trace
        """
        trace = ReasoningTrace(
            goal=goal,
            retrieved_context=retrieved_context or [],
            retrieved_memory_ids=retrieved_memory_ids or [],
            started_at=datetime.now(),
        )

        self._active_traces[trace.trace_id] = trace
        logger.debug(f"Started reasoning trace {trace.trace_id} for goal: {goal}")

        return trace.trace_id

    def add_step(
        self,
        trace_id: UUID,
        step_type: StepType,
        content: str,
        supporting_evidence: Optional[List[str]] = None,
        memory_ids_used: Optional[List[UUID]] = None,
        confidence: Optional[float] = None,
    ) -> ReasoningStep:
        """
        Add a step to an active reasoning trace.

        Args:
            trace_id: The trace to add to
            step_type: Type of reasoning step
            content: Description of the step
            supporting_evidence: Evidence supporting this step
            memory_ids_used: Memories used in this step
            confidence: Confidence in this step

        Returns:
            The created ReasoningStep

        Raises:
            ValueError: If trace not found or max steps exceeded
        """
        if trace_id not in self._active_traces:
            raise ValueError(f"No active trace with ID {trace_id}")

        trace = self._active_traces[trace_id]

        if len(trace.steps) >= self.max_steps_per_trace:
            raise ValueError(f"Maximum steps ({self.max_steps_per_trace}) exceeded for trace {trace_id}")

        step = ReasoningStep(
            step_type=step_type,
            content=content,
            supporting_evidence=supporting_evidence or [],
            memory_ids_used=memory_ids_used or [],
            confidence=confidence,
            timestamp=datetime.now(),
        )

        trace.steps.append(step)
        logger.debug(f"Added {step_type} step to trace {trace_id}")

        return step

    def complete_trace(
        self,
        trace_id: UUID,
        conclusion: str,
        confidence: float = 1.0,
        uncertainties: Optional[List[str]] = None,
        action_taken: Optional[str] = None,
    ) -> ReasoningTrace:
        """
        Complete a reasoning trace and persist it.

        Args:
            trace_id: The trace to complete
            conclusion: The conclusion reached
            confidence: Overall confidence in the reasoning
            uncertainties: Any uncertainties or caveats
            action_taken: Action taken based on the reasoning

        Returns:
            The completed ReasoningTrace

        Raises:
            ValueError: If trace not found
        """
        if trace_id not in self._active_traces:
            raise ValueError(f"No active trace with ID {trace_id}")

        trace = self._active_traces[trace_id]
        trace.conclusion = conclusion
        trace.confidence = confidence
        trace.uncertainties = uncertainties or []
        trace.action_taken = action_taken
        trace.completed_at = datetime.now()

        # Generate pattern embedding for similarity search
        if self.embedder:
            pattern_text = self._generate_pattern_text(trace)
            trace.pattern_embedding = self.embedder.embed(pattern_text)

        # Persist to store
        self._persist_trace(trace)

        # Remove from active traces
        del self._active_traces[trace_id]

        logger.info(f"Completed reasoning trace {trace_id}: {conclusion[:50]}...")
        return trace

    def abort_trace(self, trace_id: UUID) -> None:
        """
        Abort an in-progress trace without completing it.

        Args:
            trace_id: The trace to abort
        """
        if trace_id in self._active_traces:
            del self._active_traces[trace_id]
            logger.debug(f"Aborted trace {trace_id}")

    def get_active_trace(self, trace_id: UUID) -> Optional[ReasoningTrace]:
        """Get an active (in-progress) trace."""
        return self._active_traces.get(trace_id)

    def get_trace(self, trace_id: UUID) -> Optional[ReasoningTrace]:
        """
        Get a completed trace from the store.

        Args:
            trace_id: The trace to retrieve

        Returns:
            The ReasoningTrace or None if not found
        """
        # Check active traces first
        if trace_id in self._active_traces:
            return self._active_traces[trace_id]

        # Query the store
        return self._load_trace(trace_id)

    def record_outcome(
        self,
        trace_id: UUID,
        outcome: str,
        success: bool,
    ) -> None:
        """
        Record the outcome of a reasoning trace.

        This is called later when we observe what happened.

        Args:
            trace_id: The trace to update
            outcome: Description of the outcome
            success: Whether the outcome was successful
        """
        trace = self.get_trace(trace_id)
        if not trace:
            raise ValueError(f"Trace {trace_id} not found")

        trace.outcome = outcome
        trace.outcome_success = success

        # Update in store
        self._update_trace_outcome(trace_id, outcome, success)

        logger.info(f"Recorded outcome for trace {trace_id}: success={success}")

    def find_similar_reasoning(
        self,
        pattern_embedding: List[float],
        k: int = 5,
    ) -> List[ReasoningTrace]:
        """
        Find reasoning traces with similar patterns.

        Args:
            pattern_embedding: Embedding of the current reasoning pattern
            k: Number of results to return

        Returns:
            List of similar ReasoningTraces, ordered by similarity
        """
        return self._search_similar_traces(pattern_embedding, k)

    def explain_decision(
        self,
        trace_id: UUID,
        knowledge_state: Optional[Dict[str, Any]] = None,
        current_knowledge: Optional[Dict[str, Any]] = None,
    ) -> DecisionExplanation:
        """
        Create a full explanation of a past decision.

        Args:
            trace_id: The reasoning trace to explain
            knowledge_state: Knowledge state at decision time (if available)
            current_knowledge: Current knowledge state for comparison

        Returns:
            DecisionExplanation with full context
        """
        trace = self.get_trace(trace_id)
        if not trace:
            raise ValueError(f"Trace {trace_id} not found")

        # Build knowledge state summary
        knowledge_summary = {}
        if knowledge_state:
            knowledge_summary = {
                "memory_count": len(knowledge_state.get("memories", {})),
                "relationship_count": len(knowledge_state.get("relationships", [])),
            }

        # Check if knowledge has changed
        knowledge_changed = False
        delta_summary = None
        if current_knowledge and knowledge_state:
            # Simple comparison - in practice would be more sophisticated
            old_count = len(knowledge_state.get("memories", {}))
            new_count = len(current_knowledge.get("memories", {}))
            if old_count != new_count:
                knowledge_changed = True
                delta_summary = f"Memory count changed from {old_count} to {new_count}"

        return DecisionExplanation(
            reasoning_trace=trace,
            knowledge_state_summary=knowledge_summary,
            memories_used=[],  # Would be populated from trace.retrieved_memory_ids
            knowledge_changed=knowledge_changed,
            knowledge_delta_summary=delta_summary,
        )

    # =========================================================================
    # Persistence Helpers
    # =========================================================================

    def _persist_trace(self, trace: ReasoningTrace) -> None:
        """Persist a completed trace to the store."""
        self.store.initialize()

        sql = """
            INSERT INTO reasoning_traces (
                trace_id, snapshot_id, steps, goal, conclusion,
                retrieved_context, confidence, uncertainties,
                pattern_embedding, timestamp
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s
            )
        """

        with self.store._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    str(trace.trace_id),
                    str(trace.snapshot_id) if trace.snapshot_id else None,
                    json.dumps([s.dict() for s in trace.steps], default=str),
                    trace.goal,
                    trace.conclusion,
                    json.dumps(trace.retrieved_context),
                    trace.confidence,
                    json.dumps(trace.uncertainties),
                    trace.pattern_embedding,
                    trace.completed_at,
                ))
            conn.commit()

    def _load_trace(self, trace_id: UUID) -> Optional[ReasoningTrace]:
        """Load a trace from the store."""
        self.store.initialize()

        sql = """
            SELECT * FROM reasoning_traces WHERE trace_id = %s
        """

        with self.store._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (str(trace_id),))
                row = cur.fetchone()
                if row:
                    return self._row_to_trace(row, cur.description)
        return None

    def _update_trace_outcome(
        self,
        trace_id: UUID,
        outcome: str,
        success: bool,
    ) -> None:
        """Update outcome in the store."""
        # In a full implementation, we'd add outcome columns to the table
        # For now, this is a placeholder
        pass

    def _search_similar_traces(
        self,
        embedding: List[float],
        k: int,
    ) -> List[ReasoningTrace]:
        """Search for similar traces by pattern embedding."""
        self.store.initialize()

        sql = """
            SELECT *, pattern_embedding <=> %s as distance
            FROM reasoning_traces
            WHERE pattern_embedding IS NOT NULL
            ORDER BY pattern_embedding <=> %s
            LIMIT %s
        """

        results = []
        with self.store._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (embedding, embedding, k))
                for row in cur.fetchall():
                    # Skip distance column
                    trace = self._row_to_trace(row[:-1], cur.description[:-1])
                    if trace:
                        results.append(trace)

        return results

    def _row_to_trace(self, row: tuple, description) -> ReasoningTrace:
        """Convert a database row to a ReasoningTrace."""
        columns = [col.name for col in description]
        data = dict(zip(columns, row))

        steps = []
        if data.get("steps"):
            steps_data = data["steps"] if isinstance(data["steps"], list) else json.loads(data["steps"])
            steps = [ReasoningStep(**s) for s in steps_data]

        return ReasoningTrace(
            trace_id=_to_uuid(data["trace_id"]),
            snapshot_id=_to_uuid(data.get("snapshot_id")),
            goal=data.get("goal", ""),
            steps=steps,
            conclusion=data.get("conclusion"),
            retrieved_context=data.get("retrieved_context") or [],
            confidence=data.get("confidence", 1.0),
            uncertainties=data.get("uncertainties") or [],
            pattern_embedding=_parse_embedding(data.get("pattern_embedding")),
            completed_at=data.get("timestamp"),
        )

    def _generate_pattern_text(self, trace: ReasoningTrace) -> str:
        """Generate text for pattern embedding."""
        parts = [
            f"Goal: {trace.goal}",
            f"Steps: {', '.join(s.step_type for s in trace.steps)}",
            f"Conclusion: {trace.conclusion or 'pending'}",
        ]
        return " | ".join(parts)
