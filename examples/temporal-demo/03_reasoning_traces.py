"""
Demo 3: Reasoning Traces with LLM

This example demonstrates how to capture reasoning traces during
actual LLM-powered decision making. The trace records:
- What context was retrieved
- What the LLM observed and inferred
- What decision was made and why

This is essential for:
- Explainability: "Why did the AI do X?"
- Debugging: "What went wrong in this decision?"
- Learning: "How can we improve similar decisions?"

Prerequisites:
    pip install 'mem0ai[temporal]' openai
    export OPENAI_API_KEY=your-api-key

    # Start Postgres with pgvector:
    docker run -d --name temporal-postgres \
        -e POSTGRES_USER=user \
        -e POSTGRES_PASSWORD=password \
        -e POSTGRES_DB=temporal_demo \
        -p 5432:5432 \
        pgvector/pgvector:pg16
"""

import os
from datetime import datetime
import json

from openai import OpenAI
from mem0.temporal.config import TemporalStoreConfig
from mem0.temporal.stores.bitemporal import BitemporalStore
from mem0.temporal.reasoning.tracer import ReasoningTracer
from mem0.temporal.reasoning.models import StepType
from uuid import uuid4


def get_embedding(client: OpenAI, text: str) -> list[float]:
    """Get embedding from OpenAI."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def analyze_user_preference(client: OpenAI, context: list[str]) -> dict:
    """Use LLM to analyze user preferences from context."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Analyze the user context and determine their communication preference.
                Return a JSON object with:
                - observations: list of key observations
                - inferences: list of inferences drawn
                - recommendation: recommended communication style
                - confidence: confidence score 0-1
                - reasoning: brief explanation"""
            },
            {
                "role": "user",
                "content": f"User context:\n" + "\n".join(f"- {c}" for c in context)
            }
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


def main():
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    config = TemporalStoreConfig(
        connection_string=os.getenv(
            "DATABASE_URL",
            "postgresql://user:password@localhost:5432/temporal_demo"
        ),
        embedding_dims=1536,
    )

    store = BitemporalStore(config)
    store.initialize()

    # Create a reasoning tracer with embedder
    class OpenAIEmbedder:
        def __init__(self, client):
            self.client = client
        def embed(self, text):
            return get_embedding(self.client, text)

    tracer = ReasoningTracer(
        store=store,
        embedder=OpenAIEmbedder(openai_client),
        max_steps_per_trace=20,
    )

    print("=" * 60)
    print("Demo 3: Reasoning Traces with LLM")
    print("=" * 60)

    # -----------------------------------------------------------------
    # Scenario: AI agent deciding how to respond to a user
    # -----------------------------------------------------------------

    # Simulated retrieved context (in real app, this comes from memory search)
    retrieved_context = [
        "User has asked 5 technical questions about Python in the last hour",
        "User previously said 'I prefer concise answers, no fluff'",
        "User profile: Senior Software Engineer, 8 years experience",
        "User timezone: PST, current time for them is 2pm",
        "Previous interaction: User appreciated code examples",
    ]

    print("\n[Scenario] AI deciding on response style for a technical question...")
    print("\n[Retrieved Context]")
    for ctx in retrieved_context:
        print(f"  - {ctx}")

    # -----------------------------------------------------------------
    # Step 1: Start reasoning trace
    # -----------------------------------------------------------------
    print("\n[Step 1] Starting reasoning trace...")

    trace_id = tracer.start_trace(
        goal="Determine the best way to respond to this user's technical question",
        retrieved_context=retrieved_context,
    )
    print(f"  Trace ID: {trace_id}")

    # -----------------------------------------------------------------
    # Step 2: Use LLM to analyze and record reasoning
    # -----------------------------------------------------------------
    print("\n[Step 2] Calling LLM for analysis...")

    analysis = analyze_user_preference(openai_client, retrieved_context)

    print(f"\n  LLM Analysis:")
    print(f"  Recommendation: {analysis['recommendation']}")
    print(f"  Confidence: {analysis['confidence']}")

    # Record each observation from the LLM
    print("\n[Step 3] Recording observations from LLM...")
    for obs in analysis.get('observations', []):
        tracer.add_step(
            trace_id,
            StepType.OBSERVATION,
            obs,
        )
        print(f"  OBSERVATION: {obs}")

    # Record inferences
    print("\n[Step 4] Recording inferences from LLM...")
    for inf in analysis.get('inferences', []):
        tracer.add_step(
            trace_id,
            StepType.INFERENCE,
            inf,
        )
        print(f"  INFERENCE: {inf}")

    # Record the decision
    print("\n[Step 5] Recording decision...")
    tracer.add_step(
        trace_id,
        StepType.DECISION,
        f"Will use: {analysis['recommendation']}",
    )
    print(f"  DECISION: {analysis['recommendation']}")

    # -----------------------------------------------------------------
    # Step 6: Complete the trace
    # -----------------------------------------------------------------
    print("\n[Step 6] Completing trace...")

    completed = tracer.complete_trace(
        trace_id,
        conclusion=analysis.get('reasoning', analysis['recommendation']),
        confidence=analysis['confidence'],
        action_taken=f"Set response style to: {analysis['recommendation']}",
    )

    print(f"  Completed at: {completed.completed_at}")
    print(f"  Confidence: {completed.confidence}")

    # -----------------------------------------------------------------
    # Create a memory linked to this reasoning
    # -----------------------------------------------------------------
    print("\n[Step 7] Creating memory linked to reasoning trace...")

    memory_content = f"User prefers {analysis['recommendation']} communication style"
    memory = store.create_memory_version(
        memory_id=uuid4(),
        content=memory_content,
        valid_from=datetime.now(),
        embedding=get_embedding(openai_client, memory_content),
        user_id="demo_user",
        reasoning_trace_id=completed.trace_id,
        context_snapshot={
            "decision_type": "communication_style",
            "llm_model": "gpt-4o-mini",
            "context_count": len(retrieved_context),
        },
    )

    print(f"  Memory: '{memory.content}'")
    print(f"  Linked to trace: {memory.reasoning_trace_id}")

    # -----------------------------------------------------------------
    # Later: Query and explain the reasoning
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Querying the Reasoning Trace (Explainability)")
    print("=" * 60)

    retrieved = tracer.get_trace(trace_id)

    print(f"\n[Goal] {retrieved.goal}")

    print(f"\n[Context Retrieved] ({len(retrieved.retrieved_context)} items)")
    for ctx in retrieved.retrieved_context[:3]:
        print(f"  - {ctx[:60]}...")

    print(f"\n[Reasoning Chain]")
    for i, step in enumerate(retrieved.steps, 1):
        print(f"  {i}. [{step.step_type.upper()}] {step.content}")

    print(f"\n[Conclusion] {retrieved.conclusion}")
    print(f"[Confidence] {retrieved.confidence:.0%}")

    # -----------------------------------------------------------------
    # Find similar past reasoning (for learning)
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Finding Similar Past Reasoning")
    print("=" * 60)

    # Search for similar decision situations
    situation_embedding = get_embedding(
        openai_client,
        "deciding communication style for technical user"
    )

    similar = store.search_similar_situations(situation_embedding, k=3)

    print(f"\n  Found {len(similar)} similar past situations")
    for match in similar:
        print(f"\n  Situation (similarity: {match.similarity_score:.2f}):")
        print(f"    Memory: {match.memory_version.content[:60]}...")
        if match.reasoning_trace_id:
            print(f"    Has reasoning trace: {match.reasoning_trace_id}")

    print("\n" + "=" * 60)
    print("Key Insight: Every LLM decision is fully traceable.")
    print("We can see exactly what context was used, what the LLM")
    print("observed and inferred, and why it made its decision.")
    print("=" * 60)

    store.close()


if __name__ == "__main__":
    main()
