"""
Demo 4: Auto-Generated Ontology with LLM

This example demonstrates how the ontology (entity types and relation types)
emerges automatically from data using LLM-powered type resolution:
- LLM analyzes entities and determines their types
- Similar entities are matched to existing types
- Type definitions evolve as more instances are seen

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
from mem0.temporal.ontology.manager import EmergentOntologyManager


def get_embedding(client: OpenAI, text: str) -> list[float]:
    """Get embedding from OpenAI."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


class OpenAIEmbedder:
    """Embedder using OpenAI."""
    def __init__(self, client):
        self.client = client

    def embed(self, text):
        return get_embedding(self.client, text)


class OpenAILLM:
    """Simple LLM wrapper for ontology operations."""
    def __init__(self, client):
        self.client = client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


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

    # Create ontology manager with real LLM and embedder
    ontology = EmergentOntologyManager(
        store=store,
        llm=OpenAILLM(openai_client),
        embedder=OpenAIEmbedder(openai_client),
        type_similarity_threshold=0.85,
        use_llm_for_resolution=True,  # Enable LLM-powered type resolution
    )

    print("=" * 60)
    print("Demo 4: Auto-Generated Ontology with LLM")
    print("=" * 60)

    # -----------------------------------------------------------------
    # Entity 1: First Person
    # -----------------------------------------------------------------
    print("\n[1] Observing first entity: Alice (Person)...")

    person1_type = ontology.resolve_entity_type(
        entity_name="Person",
        attributes={
            "name": "Alice",
            "age": 28,
            "role": "Software Engineer",
            "company": "TechCorp",
        },
        context="User mentioned that Alice is a 28-year-old software engineer at TechCorp",
    )

    print(f"  Type: '{person1_type.name}'")
    print(f"  Type ID: {person1_type.type_id}")
    print(f"  Instances: {person1_type.instance_count}")
    print(f"  Attributes: {list(person1_type.observed_attributes.keys())}")

    # -----------------------------------------------------------------
    # Entity 2: Second Person (should match existing type)
    # -----------------------------------------------------------------
    print("\n[2] Observing second entity: Bob (Person)...")

    person2_type = ontology.resolve_entity_type(
        entity_name="Person",
        attributes={
            "name": "Bob",
            "email": "bob@example.com",
            "department": "Engineering",
        },
        context="Bob works in the Engineering department, his email is bob@example.com",
    )

    print(f"  Type: '{person2_type.name}'")
    print(f"  Same type as Alice: {person2_type.type_id == person1_type.type_id}")
    print(f"  Instances now: {person2_type.instance_count}")
    print(f"  Attributes: {list(person2_type.observed_attributes.keys())}")
    print("  Note: 'email' and 'department' added from this instance!")

    # -----------------------------------------------------------------
    # Entity 3: Company (new type)
    # -----------------------------------------------------------------
    print("\n[3] Observing entity: TechCorp (Company)...")

    company_type = ontology.resolve_entity_type(
        entity_name="Company",
        attributes={
            "name": "TechCorp",
            "industry": "Technology",
            "founded": 2020,
            "employees": 500,
        },
        context="TechCorp is a technology company founded in 2020 with 500 employees",
    )

    print(f"  Type: '{company_type.name}'")
    print(f"  New type (not Person): {company_type.type_id != person1_type.type_id}")
    print(f"  Attributes: {list(company_type.observed_attributes.keys())}")

    # -----------------------------------------------------------------
    # Entity 4: Project (another new type)
    # -----------------------------------------------------------------
    print("\n[4] Observing entity: AI Assistant (Project)...")

    project_type = ontology.resolve_entity_type(
        entity_name="Project",
        attributes={
            "name": "AI Assistant",
            "status": "active",
            "team_size": 5,
            "tech_stack": ["Python", "PyTorch", "FastAPI"],
        },
        context="The AI Assistant project is active with a team of 5 using Python and PyTorch",
    )

    print(f"  Type: '{project_type.name}'")
    print(f"  Attributes: {list(project_type.observed_attributes.keys())}")

    # -----------------------------------------------------------------
    # Entity 5: Another company (should match Company type)
    # -----------------------------------------------------------------
    print("\n[5] Observing entity: StartupInc (Company)...")

    company2_type = ontology.resolve_entity_type(
        entity_name="Company",
        attributes={
            "name": "StartupInc",
            "industry": "AI",
            "funding_stage": "Series A",
            "headquarters": "San Francisco",
        },
        context="StartupInc is an AI company at Series A, headquartered in San Francisco",
    )

    print(f"  Type: '{company2_type.name}'")
    print(f"  Same type as TechCorp: {company2_type.type_id == company_type.type_id}")
    print(f"  Company instances now: {company2_type.instance_count}")
    print(f"  Attributes: {list(company2_type.observed_attributes.keys())}")

    # -----------------------------------------------------------------
    # View Complete Ontology
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Complete Auto-Generated Ontology")
    print("=" * 60)

    all_types = ontology.get_current_types()

    print(f"\nEntity Types: {len(all_types)}")
    for et in all_types:
        stability = "STABLE" if et.is_stable else "evolving"
        print(f"\n  [{et.name.upper()}] ({stability})")
        print(f"    Instances: {et.instance_count}")
        print(f"    Confidence: {et.confidence:.0%}")
        print(f"    Attributes:")
        for attr_name, stats in et.observed_attributes.items():
            freq_pct = stats.frequency * 100
            print(f"      - {attr_name}: {freq_pct:.0f}% of instances, types: {stats.types_observed}")

    # -----------------------------------------------------------------
    # Ontology Graph
    # -----------------------------------------------------------------
    graph = ontology.get_ontology_graph()
    print(f"\n  Total entity types: {len(graph.entity_types)}")
    print(f"  Total relation types: {len(graph.relation_types)}")

    # -----------------------------------------------------------------
    # Demonstrate type lookup
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Type Lookup by Name")
    print("=" * 60)

    person_lookup = graph.get_entity_type_by_name("person")
    if person_lookup:
        print(f"\n  Found 'person' type:")
        print(f"    ID: {person_lookup.type_id}")
        print(f"    Instances: {person_lookup.instance_count}")
        print(f"    Attributes: {list(person_lookup.observed_attributes.keys())}")

    print("\n" + "=" * 60)
    print("Key Insight: The ontology emerged from observations!")
    print("The LLM helped determine types, and attribute statistics")
    print("were automatically tracked as more instances were seen.")
    print("No upfront schema design was required.")
    print("=" * 60)

    store.close()


if __name__ == "__main__":
    main()
