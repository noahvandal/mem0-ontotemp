"""
Models for reasoning traces.

Reasoning traces capture the decision-making process:
- What goal was being pursued
- What steps were taken
- What evidence was considered
- What conclusion was reached
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class StepType(str, Enum):
    """Types of reasoning steps."""

    OBSERVATION = "observation"  # Noting a fact or input
    INFERENCE = "inference"  # Drawing a conclusion from facts
    HYPOTHESIS = "hypothesis"  # Forming a tentative explanation
    DECISION = "decision"  # Making a choice
    ACTION = "action"  # Taking an action
    REFLECTION = "reflection"  # Meta-cognition about the reasoning


class ReasoningStep(BaseModel):
    """A single step in a reasoning trace."""

    model_config = ConfigDict(use_enum_values=True)

    step_id: UUID = Field(default_factory=uuid4)
    step_type: StepType = Field(..., description="Type of reasoning step")
    content: str = Field(..., description="Description of the step")
    supporting_evidence: List[str] = Field(
        default_factory=list,
        description="Evidence or facts that support this step",
    )
    memory_ids_used: List[UUID] = Field(
        default_factory=list,
        description="Memory IDs that were retrieved/used in this step",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence in this step",
    )
    timestamp: datetime = Field(default_factory=datetime.now)


class ReasoningTrace(BaseModel):
    """
    A complete reasoning trace capturing a decision-making process.

    Traces are linked to memory versions so we can understand:
    - What the agent knew when making a decision
    - How it reasoned about that knowledge
    - What conclusion it reached
    """

    trace_id: UUID = Field(default_factory=uuid4)
    snapshot_id: Optional[UUID] = Field(
        default=None,
        description="Link to graph snapshot at time of reasoning",
    )

    # The reasoning process
    goal: str = Field(..., description="What goal was being pursued")
    steps: List[ReasoningStep] = Field(default_factory=list, description="Steps in the reasoning")
    conclusion: Optional[str] = Field(default=None, description="Final conclusion reached")

    # Context
    retrieved_context: List[str] = Field(
        default_factory=list,
        description="Memories/context that were retrieved",
    )
    retrieved_memory_ids: List[UUID] = Field(
        default_factory=list,
        description="IDs of memories that were retrieved",
    )

    # Metadata
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Overall confidence")
    uncertainties: List[str] = Field(
        default_factory=list,
        description="Uncertainties or caveats in the reasoning",
    )

    # For similarity search
    pattern_embedding: Optional[List[float]] = Field(
        default=None,
        description="Embedding of the reasoning pattern",
    )

    # Timing
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = Field(default=None)

    # Outcome tracking (can be updated later)
    action_taken: Optional[str] = Field(default=None, description="Action taken based on reasoning")
    outcome: Optional[str] = Field(default=None, description="Observed outcome")
    outcome_success: Optional[bool] = Field(default=None, description="Was the outcome successful?")

    model_config = ConfigDict(frozen=False)  # Allow updating outcome later

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get the duration of the reasoning process."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def is_complete(self) -> bool:
        """Check if the trace has been completed."""
        return self.conclusion is not None and self.completed_at is not None


class DecisionExplanation(BaseModel):
    """
    Full explanation of a past decision.

    Combines the reasoning trace with the knowledge state at the time.
    """

    # The decision point
    reasoning_trace: ReasoningTrace = Field(..., description="The reasoning trace")

    # Knowledge state at decision time
    knowledge_state_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of what was known at the time",
    )
    memories_used: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Memories that were used in the decision",
    )

    # Comparison with current knowledge
    knowledge_changed: bool = Field(
        default=False,
        description="Whether relevant knowledge has changed since",
    )
    knowledge_delta_summary: Optional[str] = Field(
        default=None,
        description="Summary of how knowledge has changed",
    )

    # Evaluation
    was_reasonable: Optional[bool] = Field(
        default=None,
        description="Was the decision reasonable given the knowledge at the time?",
    )
    retrospective_notes: Optional[str] = Field(
        default=None,
        description="Notes from retrospective analysis",
    )


class ReasoningPattern(BaseModel):
    """
    A pattern in reasoning that can be matched for similarity search.

    Used for "intuition-like" retrieval: "When have I reasoned like this before?"
    """

    pattern_id: UUID = Field(default_factory=uuid4)

    # Pattern definition
    goal_type: str = Field(..., description="Type/category of goal")
    step_sequence: List[StepType] = Field(
        default_factory=list,
        description="Sequence of step types",
    )
    key_concepts: List[str] = Field(
        default_factory=list,
        description="Key concepts involved in the reasoning",
    )

    # Statistics
    occurrence_count: int = Field(default=1)
    success_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # For similarity matching
    pattern_embedding: Optional[List[float]] = Field(default=None)
