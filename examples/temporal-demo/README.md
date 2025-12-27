# Temporal Memory Demos

These demos showcase the XTDB-like immutable temporal graph retrieval system for mem0.

## Quick Start

### 1. Start Postgres with pgvector

```bash
docker-compose up -d
```

Or manually:
```bash
docker run -d --name temporal-postgres \
    -e POSTGRES_USER=user \
    -e POSTGRES_PASSWORD=password \
    -e POSTGRES_DB=temporal_demo \
    -p 5432:5432 \
    pgvector/pgvector:pg16
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY=your-api-key
```

### 3. Install dependencies

```bash
pip install 'mem0ai[temporal]' openai
```

### 4. Run the demos

```bash
python 01_basic_versioning.py
python 02_graph_reconstruction.py
python 03_reasoning_traces.py
python 04_auto_ontology.py
python 05_full_agent_session.py
```

## Demo Descriptions

### 01_basic_versioning.py
**Core concept: Immutable memory versioning**

Shows how memories are never mutated, only superseded:
- Create a memory about user preferences
- Update the preference (creates new version)
- Query current state vs past state
- View complete version history
- Semantic search across versions

### 02_graph_reconstruction.py
**Core concept: Point-in-time graph reconstruction**

Demonstrates reconstructing the complete knowledge graph at any past moment:
- Build a knowledge graph with entities and relationships
- Watch it evolve over time
- Reconstruct the exact graph state at different points
- Semantic search at different points in time

### 03_reasoning_traces.py
**Core concept: LLM decision explainability**

Shows how to capture and query reasoning traces:
- Start a reasoning trace with a goal
- Use LLM to analyze context and make decisions
- Record observations, inferences, and decisions
- Link memories to their reasoning traces
- Query "why did the AI decide X?"

### 04_auto_ontology.py
**Core concept: Emergent ontology**

Demonstrates auto-generated entity types:
- Observe entities and let types emerge
- LLM helps determine type similarity
- Attribute statistics tracked automatically
- No upfront schema design needed

### 05_full_agent_session.py
**Complete end-to-end example**

Simulates a full AI agent session:
- Agent learns about user through conversation
- Makes decisions with full reasoning traces
- Updates knowledge as new info arrives
- Demonstrates time-travel queries
- Shows explainability for past decisions

## Key Concepts

### Bitemporal Modeling
Every memory has two time dimensions:
- **Valid time**: When the fact is/was true in the real world
- **Transaction time**: When we recorded this in the system

This enables answering:
- "What do we currently believe?" (current state)
- "What did we believe at time T?" (point-in-time query)
- "When did we learn X?" (transaction history)

### Immutable Versioning
Memories are never updated in place. Instead:
1. Old version is marked as "superseded"
2. New version is created with link to previous
3. Both versions remain queryable forever

### Reasoning Traces
Every AI decision can be traced:
- What context was retrieved?
- What observations were made?
- What inferences were drawn?
- What was the final decision and confidence?

### Emergent Ontology
Entity types are not defined upfront:
- First instance of a type creates the type definition
- Subsequent instances refine the type
- Attributes are tracked with frequency statistics
- Types stabilize as more instances are observed

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings and LLM | Required |
| `DATABASE_URL` | Postgres connection string | `postgresql://user:password@localhost:5432/temporal_demo` |

## Cleanup

```bash
docker-compose down -v
# or
docker rm -f temporal-postgres
```
