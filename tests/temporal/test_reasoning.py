"""
Tests for the Reasoning trace system.

Tests:
- Creating and completing reasoning traces
- Adding reasoning steps
- Recording outcomes
- Finding similar reasoning patterns
"""

import pytest
from datetime import datetime
from uuid import uuid4

from mem0.temporal.reasoning.models import (
    ReasoningStep,
    ReasoningTrace,
    StepType,
    DecisionExplanation,
)


class TestReasoningStep:
    """Tests for ReasoningStep model."""

    def test_create_step(self):
        """Test creating a reasoning step."""
        step = ReasoningStep(
            step_type=StepType.OBSERVATION,
            content="User requested help with task X",
            supporting_evidence=["Message: 'help me with X'"],
        )

        assert step.step_type == StepType.OBSERVATION
        assert step.content == "User requested help with task X"
        assert len(step.supporting_evidence) == 1

    def test_step_types(self):
        """Test all step types can be created."""
        for step_type in StepType:
            step = ReasoningStep(
                step_type=step_type,
                content=f"Test {step_type.value}",
            )
            assert step.step_type == step_type


class TestReasoningTrace:
    """Tests for ReasoningTrace model."""

    def test_create_trace(self, sample_reasoning_trace):
        """Test creating a reasoning trace."""
        assert sample_reasoning_trace.goal == "Test goal"
        assert len(sample_reasoning_trace.steps) == 3
        assert sample_reasoning_trace.conclusion == "Test conclusion"
        assert sample_reasoning_trace.confidence == 0.9

    def test_trace_is_complete(self):
        """Test is_complete property."""
        # Incomplete trace
        incomplete = ReasoningTrace(goal="Test goal")
        assert not incomplete.is_complete

        # Complete trace
        complete = ReasoningTrace(
            goal="Test goal",
            conclusion="Decided X",
            completed_at=datetime.now(),
        )
        assert complete.is_complete

    def test_trace_duration(self):
        """Test duration calculation."""
        start = datetime(2024, 1, 1, 10, 0, 0)
        end = datetime(2024, 1, 1, 10, 0, 30)  # 30 seconds later

        trace = ReasoningTrace(
            goal="Test",
            started_at=start,
            completed_at=end,
            conclusion="Done",
        )

        assert trace.duration_seconds == 30.0


class TestReasoningTracer:
    """Tests for ReasoningTracer."""

    @pytest.fixture
    def tracer(self, bitemporal_store):
        """Create a ReasoningTracer for testing."""
        from mem0.temporal.reasoning.tracer import ReasoningTracer

        return ReasoningTracer(
            store=bitemporal_store,
            embedder=None,
            max_steps_per_trace=10,
        )

    def test_start_trace(self, tracer):
        """Test starting a new trace."""
        trace_id = tracer.start_trace(
            goal="Decide which action to take",
            retrieved_context=["Context 1", "Context 2"],
        )

        assert trace_id is not None

        # Should be in active traces
        active = tracer.get_active_trace(trace_id)
        assert active is not None
        assert active.goal == "Decide which action to take"

    def test_add_steps(self, tracer):
        """Test adding steps to a trace."""
        trace_id = tracer.start_trace("Test goal")

        tracer.add_step(trace_id, StepType.OBSERVATION, "Observed X")
        tracer.add_step(trace_id, StepType.INFERENCE, "Therefore Y")
        tracer.add_step(trace_id, StepType.DECISION, "Decided Z")

        active = tracer.get_active_trace(trace_id)
        assert len(active.steps) == 3

    def test_max_steps_limit(self, tracer):
        """Test that max steps limit is enforced."""
        trace_id = tracer.start_trace("Test goal")

        # Add max steps
        for i in range(10):
            tracer.add_step(trace_id, StepType.OBSERVATION, f"Step {i}")

        # Should raise when exceeding limit
        with pytest.raises(ValueError, match="Maximum steps"):
            tracer.add_step(trace_id, StepType.OBSERVATION, "One too many")

    def test_complete_trace(self, tracer):
        """Test completing a trace."""
        trace_id = tracer.start_trace("Test goal")
        tracer.add_step(trace_id, StepType.DECISION, "Decided X")

        completed = tracer.complete_trace(
            trace_id,
            conclusion="Final decision: X",
            confidence=0.95,
            action_taken="Did X",
        )

        assert completed.is_complete
        assert completed.conclusion == "Final decision: X"
        assert completed.confidence == 0.95
        assert completed.action_taken == "Did X"

        # Should no longer be in active traces
        assert tracer.get_active_trace(trace_id) is None

    def test_abort_trace(self, tracer):
        """Test aborting a trace."""
        trace_id = tracer.start_trace("Test goal")
        tracer.add_step(trace_id, StepType.OBSERVATION, "Something")

        tracer.abort_trace(trace_id)

        # Should be gone
        assert tracer.get_active_trace(trace_id) is None

    def test_add_step_to_nonexistent_trace(self, tracer):
        """Test adding step to non-existent trace raises error."""
        fake_id = uuid4()

        with pytest.raises(ValueError, match="No active trace"):
            tracer.add_step(fake_id, StepType.OBSERVATION, "Test")

    @pytest.mark.integration
    def test_persist_and_retrieve_trace(self, tracer):
        """Test that completed traces are persisted and retrievable."""
        trace_id = tracer.start_trace("Persisted goal")
        tracer.add_step(trace_id, StepType.DECISION, "Made decision")
        tracer.complete_trace(trace_id, "Conclusion", confidence=0.8)

        # Should be retrievable from store
        retrieved = tracer.get_trace(trace_id)
        assert retrieved is not None
        assert retrieved.goal == "Persisted goal"
        assert retrieved.conclusion == "Conclusion"


class TestDecisionExplanation:
    """Tests for DecisionExplanation model."""

    def test_create_explanation(self, sample_reasoning_trace):
        """Test creating a decision explanation."""
        explanation = DecisionExplanation(
            reasoning_trace=sample_reasoning_trace,
            knowledge_state_summary={"memory_count": 10},
            knowledge_changed=True,
            knowledge_delta_summary="2 new memories added",
        )

        assert explanation.reasoning_trace.goal == "Test goal"
        assert explanation.knowledge_changed
        assert "2 new memories" in explanation.knowledge_delta_summary
