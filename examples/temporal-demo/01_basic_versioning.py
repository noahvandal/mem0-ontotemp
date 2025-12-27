"""
Demo 1: Basic Memory Versioning

This example demonstrates the core concept of immutable memory versioning:
- Memories are never mutated, only superseded
- You can query the current state or any past state
- Full version history is preserved

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

    # Connect to Postgres
    config = TemporalStoreConfig(
        connection_string=os.getenv(
            "DATABASE_URL",
            "postgresql://user:password@localhost:5432/temporal_demo"
        ),
        embedding_dims=1536,  # text-embedding-3-small dimension
    )

    store = BitemporalStore(config)
    store.initialize()

    print("=" * 60)
    print("Demo 1: Basic Memory Versioning")
    print("=" * 60)

    # Create a stable memory ID (this never changes)
    memory_id = uuid4()
    user_id = "demo_user"

    # -----------------------------------------------------------------
    # Step 1: Create initial memory
    # -----------------------------------------------------------------
    print("\n[Step 1] Creating initial memory...")

    content_v1 = "User's favorite color is blue"
    embedding_v1 = get_embedding(openai_client, content_v1)

    v1 = store.create_memory_version(
        memory_id=memory_id,
        content=content_v1,
        valid_from=datetime.now(),
        embedding=embedding_v1,
        user_id=user_id,
    )

    print(f"  Created version: {v1.version_id}")
    print(f"  Content: '{v1.content}'")
    print(f"  Transaction time: {v1.transaction_time}")

    # Remember this time for later queries
    time_after_v1 = datetime.now()
    time.sleep(0.1)  # Small delay to ensure distinct timestamps

    # -----------------------------------------------------------------
    # Step 2: Update the memory (creates new version, doesn't mutate!)
    # -----------------------------------------------------------------
    print("\n[Step 2] User changes preference - creating new version...")

    # First, supersede the old version
    store.supersede_memory(v1.version_id)

    # Create new version with new embedding
    content_v2 = "User's favorite color is green (changed from blue)"
    embedding_v2 = get_embedding(openai_client, content_v2)

    v2 = store.create_memory_version(
        memory_id=memory_id,  # Same memory ID!
        content=content_v2,
        valid_from=datetime.now(),
        embedding=embedding_v2,
        user_id=user_id,
        previous_version_id=v1.version_id,  # Track lineage
    )

    print(f"  Created version: {v2.version_id}")
    print(f"  Content: '{v2.content}'")
    print(f"  Previous version: {v2.previous_version_id}")

    time.sleep(0.1)

    # -----------------------------------------------------------------
    # Step 3: Query current state
    # -----------------------------------------------------------------
    print("\n[Step 3] Querying current state...")

    current = store.get_current_version(memory_id)
    print(f"  Current content: '{current.content}'")

    # -----------------------------------------------------------------
    # Step 4: Query past state (time travel!)
    # -----------------------------------------------------------------
    print("\n[Step 4] Time travel - querying state from before the update...")

    past = store.get_version_as_of(memory_id, time_after_v1)
    print(f"  Content at {time_after_v1}: '{past.content}'")

    # -----------------------------------------------------------------
    # Step 5: View complete history
    # -----------------------------------------------------------------
    print("\n[Step 5] Viewing complete version history...")

    history = store.get_version_history(memory_id)
    print(f"  Total versions: {len(history.versions)}")

    for i, version in enumerate(history.versions):
        status = "(superseded)" if version.superseded_at else "(current)"
        print(f"  Version {i+1}: '{version.content}' {status}")

    # -----------------------------------------------------------------
    # Step 6: Similarity search finds the right version
    # -----------------------------------------------------------------
    print("\n[Step 6] Similarity search with 'What color does the user like?'...")

    query = "What color does the user like?"
    query_embedding = get_embedding(openai_client, query)

    similar = store.search_similar_memories(
        query_embedding,
        k=3,
        user_id=user_id,
    )

    print(f"  Found {len(similar)} similar memories:")
    for mem in similar:
        print(f"    - {mem.content}")

    print("\n" + "=" * 60)
    print("Key Insight: The original memory was NEVER modified.")
    print("Instead, we created a new version and marked the old one superseded.")
    print("This means we can always reconstruct past states accurately!")
    print("=" * 60)

    store.close()


if __name__ == "__main__":
    main()
