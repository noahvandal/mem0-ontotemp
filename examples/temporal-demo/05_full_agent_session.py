"""
Demo 5: Full Agent Session with Temporal Memory

This comprehensive demo simulates a complete AI agent session showing
all temporal memory capabilities working together:
- Learning from conversation
- Making decisions with reasoning traces
- Updating knowledge over time
- Time-travel queries
- Explainability

Prerequisites:
    pip install 'mem0ai[temporal]' openai
    export OPENAI_API_KEY=your-api-key

    # Start Postgres with pgvector:
    docker-compose up -d
"""

import os
from datetime import datetime
import time
import json

from openai import OpenAI
from mem0.temporal.config import TemporalStoreConfig
from mem0.temporal.stores.bitemporal import BitemporalStore
from mem0.temporal.reasoning.tracer import ReasoningTracer
from mem0.temporal.reasoning.models import StepType
from mem0.temporal.queries import TemporalQueries
from uuid import uuid4


class TemporalAgent:
    """An AI agent with temporal memory capabilities."""

    def __init__(self, openai_client: OpenAI, store: BitemporalStore):
        self.openai = openai_client
        self.store = store
        self.tracer = ReasoningTracer(store=store, embedder=self)
        self.queries = TemporalQueries(store)
        self.user_id = "agent_demo_user"

    def embed(self, text: str) -> list[float]:
        """Get embedding from OpenAI."""
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    def chat(self, prompt: str) -> str:
        """Get response from LLM."""
        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def extract_facts(self, message: str) -> list[dict]:
        """Use LLM to extract facts from a message."""
        prompt = f"""Extract key facts from this message as JSON array.
Each fact should have: "content" (the fact), "type" (person/preference/event/other)

Message: {message}

Return JSON array only."""

        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("facts", result.get("items", [result] if "content" in result else []))

    def remember(self, content: str, reasoning_trace_id=None, context=None) -> uuid4:
        """Store a memory with embedding."""
        memory_id = uuid4()
        self.store.create_memory_version(
            memory_id=memory_id,
            content=content,
            valid_from=datetime.now(),
            embedding=self.embed(content),
            user_id=self.user_id,
            reasoning_trace_id=reasoning_trace_id,
            context_snapshot=context,
        )
        return memory_id

    def recall(self, query: str, k: int = 5) -> list:
        """Search for relevant memories."""
        return self.store.search_similar_memories(
            self.embed(query),
            k=k,
            user_id=self.user_id,
        )

    def update_memory(self, memory_id: uuid4, new_content: str, reason: str):
        """Update a memory (creates new version)."""
        current = self.store.get_current_version(memory_id)
        if current:
            self.store.supersede_memory(current.version_id)
            self.store.create_memory_version(
                memory_id=memory_id,
                content=new_content,
                valid_from=datetime.now(),
                embedding=self.embed(new_content),
                user_id=self.user_id,
                previous_version_id=current.version_id,
                context_snapshot={"update_reason": reason},
            )


def main():
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

    agent = TemporalAgent(openai_client, store)

    print("=" * 70)
    print("Demo 5: Full Agent Session with Temporal Memory")
    print("=" * 70)

    # =========================================================================
    # PHASE 1: Initial Conversation - Learning about the user
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Initial Conversation")
    print("=" * 70)

    user_message_1 = """
    Hi! I'm Sarah, a data scientist at DataCorp. I've been working there for 3 years.
    I mainly work with Python and love using pandas for data analysis.
    Oh, and I prefer detailed explanations when learning new things.
    """

    print(f"\n[User]: {user_message_1.strip()}")
    print("\n[Agent Processing...]")

    # Extract and store facts
    facts = agent.extract_facts(user_message_1)
    memory_ids = {}

    for fact in facts:
        if isinstance(fact, dict) and "content" in fact:
            mid = agent.remember(fact["content"])
            memory_ids[fact["content"][:30]] = mid
            print(f"  Remembered: {fact['content']}")

    time.sleep(0.1)
    time_phase1 = datetime.now()
    time.sleep(0.1)

    # =========================================================================
    # PHASE 2: Agent Makes a Decision with Reasoning
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Agent Decision with Reasoning Trace")
    print("=" * 70)

    user_question = "Can you explain how neural networks work?"
    print(f"\n[User]: {user_question}")

    # Retrieve context
    relevant_memories = agent.recall("user preferences learning style")
    context = [m.content for m in relevant_memories]

    print("\n[Agent Reasoning...]")
    print(f"  Retrieved {len(context)} relevant memories")

    # Start reasoning trace
    trace_id = agent.tracer.start_trace(
        goal="Determine how to explain neural networks to this user",
        retrieved_context=context,
    )

    # Use LLM to decide
    decision_prompt = f"""Based on this user context, how should I explain neural networks?

Context:
{chr(10).join(f'- {c}' for c in context)}

Return JSON with: observations (list), decision (string), confidence (0-1)"""

    decision_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": decision_prompt}],
        response_format={"type": "json_object"},
    )
    decision = json.loads(decision_response.choices[0].message.content)

    # Record reasoning
    for obs in decision.get("observations", []):
        agent.tracer.add_step(trace_id, StepType.OBSERVATION, obs)
        print(f"  OBSERVATION: {obs}")

    agent.tracer.add_step(
        trace_id,
        StepType.DECISION,
        decision.get("decision", "Use detailed explanation style"),
    )
    print(f"  DECISION: {decision.get('decision')}")

    completed_trace = agent.tracer.complete_trace(
        trace_id,
        conclusion=decision.get("decision", "Detailed explanation"),
        confidence=decision.get("confidence", 0.9),
    )

    # Store the decision as a memory linked to reasoning
    agent.remember(
        f"When explaining to Sarah, use detailed explanations with Python examples",
        reasoning_trace_id=completed_trace.trace_id,
        context={"question_type": "technical_explanation"},
    )

    print(f"\n  Trace completed with {completed_trace.confidence:.0%} confidence")

    time.sleep(0.1)
    time_phase2 = datetime.now()
    time.sleep(0.1)

    # =========================================================================
    # PHASE 3: Information Update
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Information Update")
    print("=" * 70)

    user_message_2 = """
    Actually, I just switched jobs! I'm now at AIStartup as a ML Engineer.
    Also, I've gotten more experienced now, so I prefer concise explanations.
    """

    print(f"\n[User]: {user_message_2.strip()}")
    print("\n[Agent Processing Update...]")

    # Find and update the job memory
    job_memories = agent.recall("Sarah works at company job")
    for mem in job_memories:
        if "DataCorp" in mem.content:
            print(f"  Updating: '{mem.content[:50]}...'")
            agent.update_memory(
                mem.memory_id,
                "Sarah is a ML Engineer at AIStartup (previously Data Scientist at DataCorp)",
                reason="User announced job change",
            )
            print(f"  New: 'Sarah is a ML Engineer at AIStartup'")

    # Update preference
    pref_memories = agent.recall("Sarah prefers detailed explanations")
    for mem in pref_memories:
        if "detailed" in mem.content.lower():
            print(f"  Updating: '{mem.content[:50]}...'")
            agent.update_memory(
                mem.memory_id,
                "Sarah now prefers concise explanations (changed from detailed)",
                reason="User stated new preference",
            )
            print(f"  New: 'Sarah now prefers concise explanations'")

    time.sleep(0.1)
    time_phase3 = datetime.now()

    # =========================================================================
    # PHASE 4: Time Travel Queries
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: Time Travel Queries")
    print("=" * 70)

    # Current state
    print("\n[Current Knowledge State]")
    current_graph = agent.queries.graph_at_time(datetime.now(), user_id=agent.user_id)
    print(f"  Total memories: {current_graph.memory_count}")

    current_memories = agent.recall("Sarah job preferences", k=5)
    for mem in current_memories:
        print(f"    - {mem.content[:70]}...")

    # State after Phase 1
    print(f"\n[Knowledge State after Phase 1 - {time_phase1.strftime('%H:%M:%S')}]")
    phase1_graph = agent.queries.graph_at_time(time_phase1, user_id=agent.user_id)
    print(f"  Total memories: {phase1_graph.memory_count}")

    for mid, mem in list(phase1_graph.memories.items())[:3]:
        print(f"    - {mem.content[:70]}...")

    # State after Phase 2
    print(f"\n[Knowledge State after Phase 2 - {time_phase2.strftime('%H:%M:%S')}]")
    phase2_graph = agent.queries.graph_at_time(time_phase2, user_id=agent.user_id)
    print(f"  Total memories: {phase2_graph.memory_count}")

    # =========================================================================
    # PHASE 5: Explainability Query
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 5: Explainability - Why did the agent decide X?")
    print("=" * 70)

    print("\n[Query: Why did the agent decide to use detailed explanations?]")

    # Retrieve the reasoning trace
    retrieved_trace = agent.tracer.get_trace(trace_id)

    print(f"\n  Goal: {retrieved_trace.goal}")
    print(f"\n  Context Retrieved:")
    for ctx in retrieved_trace.retrieved_context[:3]:
        print(f"    - {ctx[:60]}...")

    print(f"\n  Reasoning Chain:")
    for i, step in enumerate(retrieved_trace.steps, 1):
        print(f"    {i}. [{step.step_type.upper()}] {step.content}")

    print(f"\n  Conclusion: {retrieved_trace.conclusion}")
    print(f"  Confidence: {retrieved_trace.confidence:.0%}")

    # =========================================================================
    # PHASE 6: Similar Situations
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 6: Finding Similar Past Decisions")
    print("=" * 70)

    query = "how to explain technical concepts to users"
    similar = store.search_similar_situations(agent.embed(query), k=3)

    print(f"\n[Query: '{query}']")
    print(f"\n  Found {len(similar)} similar past situations:")

    for match in similar:
        print(f"\n  Similarity: {match.similarity_score:.2f}")
        print(f"    Memory: {match.memory_version.content[:60]}...")
        if match.reasoning_trace_id:
            trace = agent.tracer.get_trace(match.reasoning_trace_id)
            if trace:
                print(f"    Decision: {trace.conclusion[:60]}...")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Temporal Memory Capabilities Demonstrated")
    print("=" * 70)

    print("""
    1. IMMUTABLE VERSIONING
       - Memories are never deleted or modified in place
       - Updates create new versions linked to previous ones
       - Complete history is always preserved

    2. POINT-IN-TIME QUERIES
       - Reconstruct knowledge state at any past moment
       - Answer "what did the agent believe at time T?"
       - Semantic search works on historical states

    3. REASONING TRACES
       - Every decision is fully traceable
       - Records: goal, context, observations, inferences, decision
       - Enables "why did the agent do X?" queries

    4. SITUATION SIMILARITY
       - Find similar past decision contexts
       - Learn from previous decisions
       - Enables intuition-like retrieval

    5. BITEMPORAL MODELING
       - Valid time: when facts are true in the world
       - Transaction time: when we learned them
       - Enables complete auditability
    """)

    store.close()


if __name__ == "__main__":
    main()
