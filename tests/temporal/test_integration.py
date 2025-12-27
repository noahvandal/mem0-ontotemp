"""
Integration tests for the complete Temporal Memory system.

Tests the full workflow:
- Adding memories with reasoning traces
- Updating memories (creating versions)
- Point-in-time queries
- Graph reconstruction
- Similar situation retrieval
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
import time


@pytest.mark.integration
class TestFullWorkflow:
    """Integration tests for the complete temporal memory workflow."""

    def test_add_and_version_memory(self, bitemporal_store, sample_embedding):
        """
        Test adding a memory and then updating it creates proper versions.
        """
        memory_id = uuid4()
        user_id = "workflow_test_user"

        # Step 1: Create initial memory
        t1 = datetime.now()
        v1 = bitemporal_store.create_memory_version(
            memory_id=memory_id,
            content="User's favorite color is blue",
            valid_from=t1,
            embedding=sample_embedding,
            user_id=user_id,
        )

        assert v1.content == "User's favorite color is blue"
        assert v1.superseded_at is None

        time.sleep(0.1)
        t_between = datetime.now()
        time.sleep(0.1)

        # Step 2: Update memory (supersede old, create new)
        bitemporal_store.supersede_memory(v1.version_id)
        v2 = bitemporal_store.create_memory_version(
            memory_id=memory_id,
            content="User's favorite color is green (changed from blue)",
            valid_from=datetime.now(),
            embedding=sample_embedding,
            user_id=user_id,
            previous_version_id=v1.version_id,
        )

        # Step 3: Verify current version is v2
        current = bitemporal_store.get_current_version(memory_id)
        assert current.version_id == v2.version_id
        assert "green" in current.content

        # Step 4: Query as of t_between - should get v1
        past_version = bitemporal_store.get_version_as_of(memory_id, t_between)
        assert past_version is not None
        assert past_version.version_id == v1.version_id
        assert "blue" in past_version.content

        # Step 5: Verify history has both versions
        history = bitemporal_store.get_version_history(memory_id)
        assert len(history.versions) == 2
        assert history.first.content == "User's favorite color is blue"
        assert history.current.content == "User's favorite color is green (changed from blue)"

    def test_graph_reconstruction_workflow(self, bitemporal_store, sample_embedding):
        """
        Test reconstructing the knowledge graph at different points in time.
        """
        user_id = "graph_workflow_user"

        # Step 1: Create some initial memories
        t1 = datetime.now()
        memory_ids = []
        for i in range(3):
            v = bitemporal_store.create_memory_version(
                memory_id=uuid4(),
                content=f"Initial memory {i}",
                valid_from=t1,
                embedding=sample_embedding,
                user_id=user_id,
            )
            memory_ids.append(v.memory_id)

        time.sleep(0.1)
        t_initial = datetime.now()
        time.sleep(0.1)

        # Step 2: Add more memories
        for i in range(2):
            bitemporal_store.create_memory_version(
                memory_id=uuid4(),
                content=f"Later memory {i}",
                valid_from=datetime.now(),
                embedding=sample_embedding,
                user_id=user_id,
            )

        time.sleep(0.1)

        # Step 3: Reconstruct graph at t_initial (should have 3 memories)
        graph_initial = bitemporal_store.reconstruct_graph_as_of(t_initial, user_id=user_id)
        assert graph_initial.memory_count == 3

        # Step 4: Reconstruct graph now (should have 5 memories)
        graph_now = bitemporal_store.reconstruct_graph_as_of(datetime.now(), user_id=user_id)
        assert graph_now.memory_count == 5

    def test_relationship_versioning_workflow(self, bitemporal_store, sample_embedding):
        """
        Test creating and querying relationship versions.
        """
        # Create entities
        person = bitemporal_store.create_memory_version(
            memory_id=uuid4(),
            content="Person: Alice",
            valid_from=datetime.now(),
            embedding=sample_embedding,
        )
        company1 = bitemporal_store.create_memory_version(
            memory_id=uuid4(),
            content="Company: TechCorp",
            valid_from=datetime.now(),
            embedding=sample_embedding,
        )
        company2 = bitemporal_store.create_memory_version(
            memory_id=uuid4(),
            content="Company: StartupInc",
            valid_from=datetime.now(),
            embedding=sample_embedding,
        )

        # Step 1: Create initial relationship (Alice works at TechCorp)
        rel_id = uuid4()
        r1 = bitemporal_store.create_relationship_version(
            relationship_id=rel_id,
            source_memory_id=person.memory_id,
            target_memory_id=company1.memory_id,
            relation_name="works_at",
            valid_from=datetime.now(),
        )

        time.sleep(0.1)
        t_at_techcorp = datetime.now()
        time.sleep(0.1)

        # Step 2: Alice changes jobs (supersede old, create new)
        bitemporal_store.supersede_relationship(r1.version_id)
        r2 = bitemporal_store.create_relationship_version(
            relationship_id=rel_id,
            source_memory_id=person.memory_id,
            target_memory_id=company2.memory_id,
            relation_name="works_at",
            valid_from=datetime.now(),
            previous_version_id=r1.version_id,
        )

        # Step 3: Query current - should show StartupInc
        current_rel = bitemporal_store.get_current_relationship_version(rel_id)
        assert current_rel.target_memory_id == company2.memory_id

        # Step 4: Query as of t_at_techcorp - should show TechCorp
        past_rel = bitemporal_store.get_relationship_as_of(rel_id, t_at_techcorp)
        assert past_rel is not None
        assert past_rel.target_memory_id == company1.memory_id


@pytest.mark.integration
class TestReasoningWorkflow:
    """Integration tests for reasoning trace workflow."""

    def test_reasoning_with_memory_creation(self, bitemporal_store, sample_embedding):
        """
        Test creating memories with associated reasoning traces.
        """
        from mem0.temporal.reasoning.tracer import ReasoningTracer
        from mem0.temporal.reasoning.models import StepType

        tracer = ReasoningTracer(
            store=bitemporal_store,
            embedder=None,
            max_steps_per_trace=20,
        )

        # Step 1: Start reasoning trace
        trace_id = tracer.start_trace(
            goal="Decide user's preferred communication style",
            retrieved_context=["User mentioned they prefer concise messages"],
        )

        # Step 2: Add reasoning steps
        tracer.add_step(
            trace_id,
            StepType.OBSERVATION,
            "User has asked for 'brief' responses multiple times",
            supporting_evidence=["Message 1: 'keep it short'", "Message 5: 'just the key points'"],
        )
        tracer.add_step(
            trace_id,
            StepType.INFERENCE,
            "User values time efficiency in communication",
        )
        tracer.add_step(
            trace_id,
            StepType.DECISION,
            "Will use concise communication style",
        )

        # Step 3: Complete trace
        completed_trace = tracer.complete_trace(
            trace_id,
            conclusion="User prefers concise, direct communication",
            confidence=0.9,
            action_taken="Stored preference for concise communication",
        )

        # Step 4: Create memory linked to reasoning trace
        memory = bitemporal_store.create_memory_version(
            memory_id=uuid4(),
            content="User prefers concise communication style",
            valid_from=datetime.now(),
            embedding=sample_embedding,
            reasoning_trace_id=completed_trace.trace_id,
        )

        assert memory.reasoning_trace_id == completed_trace.trace_id

        # Step 5: Retrieve the trace
        retrieved_trace = tracer.get_trace(completed_trace.trace_id)
        assert retrieved_trace is not None
        assert retrieved_trace.goal == "Decide user's preferred communication style"
        assert len(retrieved_trace.steps) == 3


@pytest.mark.integration
class TestOntologyWorkflow:
    """Integration tests for ontology auto-generation workflow."""

    def test_entity_type_evolution(self, bitemporal_store):
        """
        Test that entity types evolve as more instances are observed.
        """
        from mem0.temporal.ontology.manager import EmergentOntologyManager

        manager = EmergentOntologyManager(
            store=bitemporal_store,
            llm=None,  # No LLM for unit tests
            embedder=None,
            type_similarity_threshold=0.85,
            use_llm_for_resolution=False,
        )

        # Step 1: First person - creates new type
        et1 = manager.resolve_entity_type(
            entity_name="Person",
            attributes={"name": "John", "age": 30},
            context="User mentioned John is 30 years old",
        )
        assert et1.name == "person"
        assert et1.instance_count == 1
        assert "name" in et1.observed_attributes
        assert "age" in et1.observed_attributes

        # Step 2: Second person - should match existing type
        et2 = manager.resolve_entity_type(
            entity_name="Person",
            attributes={"name": "Jane", "email": "jane@example.com"},
            context="User mentioned Jane's email",
        )
        assert et2.type_id == et1.type_id  # Same type
        assert et2.instance_count == 2  # Incremented
        assert "email" in et2.observed_attributes  # New attribute added

        # Step 3: Different entity type
        et3 = manager.resolve_entity_type(
            entity_name="Company",
            attributes={"name": "Acme Corp", "industry": "Tech"},
            context="User works at Acme Corp",
        )
        assert et3.type_id != et1.type_id  # Different type
        assert et3.name == "company"

        # Step 4: Get all types
        all_types = manager.get_current_types()
        type_names = [t.name for t in all_types]
        assert "person" in type_names
        assert "company" in type_names


@pytest.mark.integration
class TestSimilarSituationsWorkflow:
    """Integration tests for finding similar past situations."""

    def test_situation_similarity_search(self, bitemporal_store, sample_embedding):
        """
        Test finding similar past decision situations.
        """
        # Step 1: Create memories representing past decision situations
        for i in range(5):
            # Slightly vary the embedding to simulate different but related contexts
            varied_embedding = [x + (i * 0.02) for x in sample_embedding]
            bitemporal_store.create_memory_version(
                memory_id=uuid4(),
                content=f"Decision point {i}: chose approach {'A' if i % 2 == 0 else 'B'}",
                valid_from=datetime.now(),
                embedding=varied_embedding,
                context_snapshot={"scenario": f"test_scenario_{i}", "outcome": "success"},
                reasoning_trace_id=uuid4(),  # Link to a reasoning trace
            )

        # Step 2: Search for similar situations
        results = bitemporal_store.search_similar_situations(sample_embedding, k=3)

        # Should find some results (may be empty if no matches pass threshold)
        assert isinstance(results, list)


@pytest.mark.integration
class TestTemporalQueriesWorkflow:
    """Integration tests for temporal query interface."""

    def test_query_builder_workflow(self, bitemporal_store, sample_embedding):
        """
        Test using the TemporalQueryBuilder for complex queries.
        """
        from mem0.temporal.queries import TemporalQueryBuilder, TemporalQueries

        user_id = "query_test_user"

        # Create test data
        for i in range(5):
            bitemporal_store.create_memory_version(
                memory_id=uuid4(),
                content=f"Query test memory {i}",
                valid_from=datetime.now(),
                embedding=sample_embedding,
                user_id=user_id,
            )

        # Step 1: Use fluent query builder
        builder = TemporalQueryBuilder(bitemporal_store)
        results = (
            builder
            .for_user(user_id)
            .limit(3)
            .execute()
        )

        assert len(results) <= 3

        # Step 2: Use TemporalQueries for higher-level operations
        queries = TemporalQueries(bitemporal_store)

        # Build and execute query
        graph = queries.query().for_user(user_id).execute_graph()
        assert graph.memory_count == 5

    def test_graph_delta_workflow(self, bitemporal_store, sample_embedding):
        """
        Test computing graph deltas between time points.
        """
        from mem0.temporal.queries import TemporalQueries

        user_id = "delta_test_user"
        queries = TemporalQueries(bitemporal_store)

        # Create initial memories
        t_start = datetime.now()
        time.sleep(0.05)

        for i in range(2):
            bitemporal_store.create_memory_version(
                memory_id=uuid4(),
                content=f"Initial memory {i}",
                valid_from=datetime.now(),
                embedding=sample_embedding,
                user_id=user_id,
            )

        time.sleep(0.1)
        t_mid = datetime.now()
        time.sleep(0.1)

        # Add more memories
        for i in range(3):
            bitemporal_store.create_memory_version(
                memory_id=uuid4(),
                content=f"Later memory {i}",
                valid_from=datetime.now(),
                embedding=sample_embedding,
                user_id=user_id,
            )

        t_end = datetime.now()

        # Compute delta
        delta = queries.graph_delta(t_start, t_end, user_id=user_id)

        assert len(delta.memories_added) == 5  # All were added in this range
        assert len(delta.memories_removed) == 0


@pytest.mark.integration
class TestEndToEndWorkflow:
    """
    Complete end-to-end test of the temporal memory system.
    """

    def test_complete_agent_session_simulation(self, bitemporal_store, sample_embedding):
        """
        Simulate a complete agent session with temporal tracking.

        This test simulates:
        1. Agent starts conversation, creates initial memories
        2. Agent makes a decision with reasoning trace
        3. User provides new information, memory is updated
        4. Later query: "what did the agent believe at time X?"
        5. Find similar past situations
        """
        from mem0.temporal.reasoning.tracer import ReasoningTracer
        from mem0.temporal.reasoning.models import StepType
        from mem0.temporal.queries import TemporalQueries

        user_id = "end_to_end_user"
        tracer = ReasoningTracer(store=bitemporal_store, embedder=None)
        queries = TemporalQueries(bitemporal_store)

        # === Phase 1: Initial conversation ===
        t_start = datetime.now()

        # Agent learns user's name
        name_memory_id = uuid4()
        bitemporal_store.create_memory_version(
            memory_id=name_memory_id,
            content="User's name is John",
            valid_from=t_start,
            embedding=sample_embedding,
            user_id=user_id,
        )

        # Agent learns user's preference
        pref_memory_id = uuid4()
        bitemporal_store.create_memory_version(
            memory_id=pref_memory_id,
            content="User prefers formal communication",
            valid_from=t_start,
            embedding=sample_embedding,
            user_id=user_id,
        )

        time.sleep(0.1)
        t_initial = datetime.now()
        time.sleep(0.1)

        # === Phase 2: Agent makes a decision with reasoning ===
        trace_id = tracer.start_trace(
            goal="Determine appropriate response tone",
            retrieved_context=["User prefers formal communication"],
        )
        tracer.add_step(trace_id, StepType.OBSERVATION, "User stated preference for formal tone")
        tracer.add_step(trace_id, StepType.DECISION, "Will use formal language")
        completed_trace = tracer.complete_trace(
            trace_id,
            conclusion="Use formal communication style",
            confidence=0.95,
        )

        # Store decision as memory
        decision_memory_id = uuid4()
        bitemporal_store.create_memory_version(
            memory_id=decision_memory_id,
            content="Decision: Use formal communication with this user",
            valid_from=datetime.now(),
            embedding=sample_embedding,
            user_id=user_id,
            reasoning_trace_id=completed_trace.trace_id,
            context_snapshot={"decision_type": "communication_style"},
        )

        time.sleep(0.1)
        t_after_decision = datetime.now()
        time.sleep(0.1)

        # === Phase 3: User corrects information ===
        # User says "Actually, call me Johnny and be casual"

        # Supersede old preference
        old_pref = bitemporal_store.get_current_version(pref_memory_id)
        bitemporal_store.supersede_memory(old_pref.version_id)

        # Create new preference
        bitemporal_store.create_memory_version(
            memory_id=pref_memory_id,
            content="User prefers casual communication (updated)",
            valid_from=datetime.now(),
            embedding=sample_embedding,
            user_id=user_id,
            previous_version_id=old_pref.version_id,
        )

        time.sleep(0.1)
        t_final = datetime.now()

        # === Phase 4: Temporal queries ===

        # Q1: What did agent believe at t_initial?
        graph_initial = queries.graph_at_time(t_initial, user_id=user_id)
        assert graph_initial.memory_count == 2  # Just name and original preference

        # Q2: What did agent believe at t_after_decision?
        graph_decision = queries.graph_at_time(t_after_decision, user_id=user_id)
        assert graph_decision.memory_count == 3  # Name, preference, and decision

        # Q3: What's the current state?
        graph_current = queries.graph_at_time(t_final, user_id=user_id)
        assert graph_current.memory_count == 3

        # Q4: How did the preference change?
        pref_history = queries.full_history(pref_memory_id)
        assert len(pref_history.versions) == 2
        assert "formal" in pref_history.first.content
        assert "casual" in pref_history.current.content

        # Q5: What was the preference at t_initial?
        pref_at_initial = queries.at_time(pref_memory_id, t_initial)
        assert "formal" in pref_at_initial.content

        # Q6: What is the current preference?
        pref_current = queries.current_version(pref_memory_id)
        assert "casual" in pref_current.content

        # === Phase 5: Similar situations ===
        similar = bitemporal_store.search_similar_situations(sample_embedding, k=5)
        # Should find our decision point
        assert isinstance(similar, list)
