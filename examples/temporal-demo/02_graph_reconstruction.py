"""
Demo 2: Point-in-Time Graph Reconstruction

This example demonstrates reconstructing the complete knowledge graph
at any past point in time. This is crucial for:
- Understanding what the AI "knew" when it made a decision
- Debugging unexpected behavior
- Auditing past interactions

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
import time

from openai import OpenAI
from mem0.temporal.config import TemporalStoreConfig
from mem0.temporal.stores.bitemporal import BitemporalStore
from uuid import uuid4


def get_embedding(client: OpenAI, text: str) -> list[float]:
    """Get embedding from OpenAI."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


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

    print("=" * 60)
    print("Demo 2: Point-in-Time Graph Reconstruction")
    print("=" * 60)

    user_id = "demo_user_graph"

    # -----------------------------------------------------------------
    # Phase 1: Build initial knowledge graph
    # -----------------------------------------------------------------
    print("\n[Phase 1] Building initial knowledge graph...")

    # Create some entities with real embeddings
    person_content = "Person: Alice - Software Engineer at TechCorp"
    person_id = uuid4()
    store.create_memory_version(
        memory_id=person_id,
        content=person_content,
        valid_from=datetime.now(),
        embedding=get_embedding(openai_client, person_content),
        user_id=user_id,
    )

    company_content = "Company: TechCorp - AI startup founded in 2020"
    company_id = uuid4()
    store.create_memory_version(
        memory_id=company_id,
        content=company_content,
        valid_from=datetime.now(),
        embedding=get_embedding(openai_client, company_content),
        user_id=user_id,
    )

    # Create relationship
    rel_id = uuid4()
    store.create_relationship_version(
        relationship_id=rel_id,
        source_memory_id=person_id,
        target_memory_id=company_id,
        relation_name="works_at",
        valid_from=datetime.now(),
    )

    print("  Created: Alice (Person)")
    print("  Created: TechCorp (Company)")
    print("  Created: Alice --works_at--> TechCorp")

    # Capture state at this point
    time.sleep(0.1)
    time_phase1 = datetime.now()
    time.sleep(0.1)

    # -----------------------------------------------------------------
    # Phase 2: Knowledge evolves
    # -----------------------------------------------------------------
    print("\n[Phase 2] Knowledge evolves over time...")

    # Add more information
    project_content = "Project: AI Assistant - Natural language interface, Alice is the lead developer"
    project_id = uuid4()
    store.create_memory_version(
        memory_id=project_id,
        content=project_content,
        valid_from=datetime.now(),
        embedding=get_embedding(openai_client, project_content),
        user_id=user_id,
    )

    store.create_relationship_version(
        relationship_id=uuid4(),
        source_memory_id=person_id,
        target_memory_id=project_id,
        relation_name="leads",
        valid_from=datetime.now(),
    )

    print("  Created: AI Assistant (Project)")
    print("  Created: Alice --leads--> AI Assistant")

    time.sleep(0.1)
    time_phase2 = datetime.now()
    time.sleep(0.1)

    # -----------------------------------------------------------------
    # Phase 3: Alice changes jobs!
    # -----------------------------------------------------------------
    print("\n[Phase 3] Alice changes jobs...")

    # Supersede old relationship
    old_rel = store.get_current_relationship_version(rel_id)
    store.supersede_relationship(old_rel.version_id)

    # New company
    new_company_content = "Company: StartupInc - Series A AI company, competitor to TechCorp"
    new_company_id = uuid4()
    store.create_memory_version(
        memory_id=new_company_id,
        content=new_company_content,
        valid_from=datetime.now(),
        embedding=get_embedding(openai_client, new_company_content),
        user_id=user_id,
    )

    # New employment relationship
    store.create_relationship_version(
        relationship_id=rel_id,  # Same relationship ID, new version
        source_memory_id=person_id,
        target_memory_id=new_company_id,
        relation_name="works_at",
        valid_from=datetime.now(),
        previous_version_id=old_rel.version_id,
    )

    print("  Created: StartupInc (Company)")
    print("  Updated: Alice --works_at--> StartupInc (was TechCorp)")

    time.sleep(0.1)
    time_phase3 = datetime.now()

    # -----------------------------------------------------------------
    # Now: Reconstruct graphs at different points in time
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Graph Reconstruction at Different Points in Time")
    print("=" * 60)

    # Graph at Phase 1
    print(f"\n[Graph at Phase 1] ({time_phase1.strftime('%H:%M:%S')})")
    graph1 = store.reconstruct_graph_as_of(time_phase1, user_id=user_id)
    print(f"  Memories: {graph1.memory_count}")
    print(f"  Relationships: {len(graph1.relationships)}")
    for mid, mv in graph1.memories.items():
        print(f"    - {mv.content[:60]}...")

    # Graph at Phase 2
    print(f"\n[Graph at Phase 2] ({time_phase2.strftime('%H:%M:%S')})")
    graph2 = store.reconstruct_graph_as_of(time_phase2, user_id=user_id)
    print(f"  Memories: {graph2.memory_count}")
    print(f"  Relationships: {len(graph2.relationships)}")
    for mid, mv in graph2.memories.items():
        print(f"    - {mv.content[:60]}...")

    # Graph at Phase 3 (current)
    print(f"\n[Graph at Phase 3 - Current] ({time_phase3.strftime('%H:%M:%S')})")
    graph3 = store.reconstruct_graph_as_of(time_phase3, user_id=user_id)
    print(f"  Memories: {graph3.memory_count}")
    print(f"  Relationships: {len(graph3.relationships)}")
    for mid, mv in graph3.memories.items():
        print(f"    - {mv.content[:60]}...")

    # -----------------------------------------------------------------
    # Semantic search at different points in time
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Semantic Search: 'Where does Alice work?'")
    print("=" * 60)

    query = "Where does Alice work?"
    query_embedding = get_embedding(openai_client, query)

    # Search current state
    print("\n[Current state]")
    current_results = store.search_similar_memories(
        query_embedding, k=2, user_id=user_id
    )
    for mem in current_results:
        print(f"  - {mem.content}")

    # Search as of Phase 1
    print(f"\n[As of Phase 1 - {time_phase1.strftime('%H:%M:%S')}]")
    past_results = store.search_similar_memories(
        query_embedding, k=2, user_id=user_id, as_of=time_phase1
    )
    for mem in past_results:
        print(f"  - {mem.content}")

    print("\n" + "=" * 60)
    print("Key Insight: We can reconstruct the EXACT knowledge state")
    print("at any past moment, including semantic search results.")
    print("This enables full auditability of AI decisions in context.")
    print("=" * 60)

    store.close()


if __name__ == "__main__":
    main()
