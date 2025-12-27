"""
Demo 6: Interactive Chat Session with Temporal Memory

An interactive chat interface with full access to temporal memory features:
- Chat naturally and memories are extracted/updated automatically
- Access menu with '/menu' command to explore:
  - Ontological graph (entity types)
  - All memories with version history
  - Relationships between memories
  - Time-travel queries (state at any point in time)
  - Current vs historical state comparison

Prerequisites:
    pip install 'mem0ai[temporal]' openai
    export OPENAI_API_KEY=your-api-key
    docker-compose up -d
"""

import os
import sys
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4
import json

from openai import OpenAI
from mem0.temporal.config import TemporalStoreConfig
from mem0.temporal.stores.bitemporal import BitemporalStore
from mem0.temporal.ontology.manager import EmergentOntologyManager
from mem0.temporal.reasoning.tracer import ReasoningTracer
from mem0.temporal.reasoning.models import StepType
from mem0.temporal.queries import TemporalQueries


class InteractiveTemporalAgent:
    """An interactive AI agent with temporal memory capabilities."""

    def __init__(
        self,
        openai_client: OpenAI,
        store: BitemporalStore,
        user_id: str = "demo_user_001",
        organization_id: str = "demo_org_001",
        session_id: str = None,
    ):
        self.openai = openai_client
        self.store = store
        self.tracer = ReasoningTracer(store=store, embedder=self)
        self.queries = TemporalQueries(store)
        self.ontology = EmergentOntologyManager(
            store=store,
            llm=self,
            embedder=self,
            type_similarity_threshold=0.85,
            use_llm_for_resolution=True,
        )

        # Identity scopes
        self.user_id = user_id
        self.organization_id = organization_id
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Default scope for new memories (can be changed via /scope command)
        self.default_scope = "user"  # Options: "user", "organization", "session", "global"

        self.memory_timestamps = []  # Track when memories were created
        self.attribute_memory_map = {}  # Maps attribute_key -> memory_id for updates

    def embed(self, text: str) -> list[float]:
        """Get embedding from OpenAI."""
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    def generate(self, prompt: str) -> str:
        """Generate text from LLM."""
        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def chat(self, system_prompt: str, user_message: str) -> str:
        """Chat with context."""
        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content

    def extract_facts(self, message: str) -> list[dict]:
        """Use LLM to extract meaningful, structured facts from a message."""
        # Get existing attribute keys for context
        existing_keys = list(self.attribute_memory_map.keys())
        existing_keys_str = ", ".join(existing_keys[:20]) if existing_keys else "none yet"

        prompt = f"""Analyze this message and extract ONLY meaningful, factual information about the user.

IGNORE completely:
- Greetings (hi, hello, hey, how are you)
- Filler words (okay, sure, yes, no, huh, well)
- Questions the user asks YOU
- Generic conversational responses

EXTRACT facts about:
- User attributes (name, job, location, age)
- User preferences (likes, dislikes, preferences)
- User state (mood, feelings, current activity)
- User events/experiences (specific things that happened - each distinct event needs its OWN key)
- User skills/knowledge
- User relationships (knows X, works with Y)
- User traditions/habits (recurring activities)

CRITICAL RULES FOR attribute_key:
1. Each DISTINCT piece of information needs a UNIQUE, SPECIFIC key
2. For events: "christmas_2024_family_celebration", "rice_pudding_game_tradition", "gift_exchange_game_tradition"
3. For states: "current_mood", "current_activity"
4. For attributes: "job_title", "employer_name", "home_city"
5. For preferences: "preferred_explanation_style", "favorite_food"
6. For relationships: "family_relationship_status"
7. NEVER use generic keys like just "event" - ALWAYS be specific to the actual content
8. Multiple distinct events/traditions = multiple distinct keys
9. Only reuse a key if the new info truly UPDATES/REPLACES that specific thing

EXISTING KEYS: [{existing_keys_str}]
- Reuse an existing key ONLY if updating that exact concept
- Create a NEW key for NEW distinct information

For each fact provide:
- "attribute_key": Specific, descriptive snake_case key (NOT generic like "event")
- "content": The fact as a clear statement about the user
- "category": personal_info, preference, mood, skill, relationship, event, tradition, other

Message: "{message}"

Return JSON: {{"facts": [...]}} or {{"facts": []}} if nothing meaningful."""

        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("facts", [])

    def remember(self, content: str, attribute_key: str = None, category: str = None, scope: str = None) -> UUID:
        """Store a memory with embedding and attribute tracking."""
        memory_id = uuid4()
        now = datetime.now()
        scope = scope or self.default_scope

        # Resolve entity type if category provided
        entity_type_id = None
        if category:
            et = self.ontology.resolve_entity_type(
                entity_name=category,
                attributes={"content": content, "attribute_key": attribute_key},
                context=content,
            )
            entity_type_id = et.type_id

        self.store.create_memory_version(
            memory_id=memory_id,
            content=content,
            valid_from=now,
            embedding=self.embed(content),
            user_id=self.user_id,
            organization_id=self.organization_id,
            session_id=self.session_id,
            entity_type_id=entity_type_id,
            scope_level=scope,
            metadata={"attribute_key": attribute_key, "category": category},
        )
        self.memory_timestamps.append(now)

        # Track attribute -> memory mapping for future updates
        if attribute_key:
            self.attribute_memory_map[attribute_key] = memory_id

        return memory_id

    def recall(self, query: str, k: int = 5) -> list:
        """Search for relevant memories."""
        return self.store.search_similar_memories(
            self.embed(query),
            k=k,
            user_id=self.user_id,
        )

    def update_memory(self, memory_id: UUID, new_content: str, reason: str):
        """Update a memory (creates new version)."""
        current = self.store.get_current_version(memory_id)
        if current:
            self.store.supersede_memory(current.version_id)
            now = datetime.now()
            self.store.create_memory_version(
                memory_id=memory_id,
                content=new_content,
                valid_from=now,
                embedding=self.embed(new_content),
                user_id=self.user_id,
                previous_version_id=current.version_id,
                context_snapshot={"update_reason": reason},
            )
            self.memory_timestamps.append(now)

    def find_similar_memory(self, content: str, threshold: float = 0.85) -> tuple[UUID, str] | None:
        """Find a very similar existing memory by embedding similarity."""
        similar = self.recall(content, k=3)
        if not similar:
            return None

        # Check the top match
        for mem in similar:
            if mem.metadata and mem.metadata.get("attribute_key"):
                # Compute similarity (we don't have the score from recall, so use LLM check)
                check_prompt = f"""Are these two statements about the SAME specific thing/topic?
Statement 1: {mem.content}
Statement 2: {content}

Reply ONLY with:
- "SAME" if they are about the exact same specific thing (e.g., same event, same preference)
- "DIFFERENT" if they are about different things"""
                result = self.generate(check_prompt).strip().upper()
                if "SAME" in result:
                    return mem.memory_id, mem.metadata.get("attribute_key")
        return None

    def process_message(self, message: str) -> tuple[str, list[str]]:
        """Process user message: extract facts, update memories, generate response."""
        extracted = []

        # Extract structured facts from the message
        facts = self.extract_facts(message)

        for fact in facts:
            if not isinstance(fact, dict) or "content" not in fact:
                continue

            fact_content = fact["content"]
            attribute_key = fact.get("attribute_key", "").lower().replace(" ", "_")
            category = fact.get("category", "other")

            if not attribute_key:
                continue

            # Check if we already have a memory for this attribute key
            if attribute_key in self.attribute_memory_map:
                # Update existing memory by key match
                existing_id = self.attribute_memory_map[attribute_key]
                current = self.store.get_current_version(existing_id)

                if current and current.content != fact_content:
                    self.update_memory(existing_id, fact_content, f"User updated {attribute_key}")
                    extracted.append(f"Updated [{attribute_key}]: '{current.content}' → '{fact_content}'")
            else:
                # Check if there's a semantically similar memory we should update instead
                similar = self.find_similar_memory(fact_content)
                if similar:
                    existing_id, existing_key = similar
                    current = self.store.get_current_version(existing_id)
                    if current and current.content != fact_content:
                        self.update_memory(existing_id, fact_content, f"User updated {existing_key}")
                        # Also update the attribute map to use the new key if different
                        if attribute_key != existing_key:
                            self.attribute_memory_map[attribute_key] = existing_id
                        extracted.append(f"Updated [{existing_key}]: '{current.content}' → '{fact_content}'")
                else:
                    # Truly new - create memory
                    self.remember(fact_content, attribute_key, category)
                    extracted.append(f"Learned [{attribute_key}]: {fact_content}")

        # Generate response with context from memories
        relevant = self.recall(message, k=5)
        context_parts = []
        for m in relevant:
            attr_key = m.metadata.get("attribute_key", "unknown") if m.metadata else "unknown"
            context_parts.append(f"- [{attr_key}] {m.content}")

        context = "\n".join(context_parts) if context_parts else "No prior context about the user."

        system_prompt = f"""You are a helpful AI assistant with memory. You remember things about the user.

What you know about the user:
{context}

You are chatting in a friendly, conversational tone with the user. In addition to having a natural conversation, you are also subtly trying to learn more details about the user when appropriate, by asking gentle follow-up questions or steering the conversation to improve your understanding of them. Use what you know, and try to encourage the user to share more facts about themselves in a natural way. Be warm and conversational."""

        response = self.chat(system_prompt, message)
        return response, extracted


class InteractiveMenu:
    """Interactive menu for exploring temporal memory."""

    def __init__(self, agent: InteractiveTemporalAgent):
        self.agent = agent

    def show_main_menu(self):
        """Display main menu."""
        print("\n" + "=" * 60)
        print("  TEMPORAL MEMORY EXPLORER")
        print("=" * 60)
        print("  1. View Ontology (Entity Types)")
        print("  2. View All Memories")
        print("  3. View Memory Details & History")
        print("  4. View State at Time T")
        print("  5. View Current State Graph")
        print("  6. Search Memories")
        print("  7. View Reasoning Traces")
        print("  0. Exit Menu (back to chat)")
        print("=" * 60)

    def run(self):
        """Run the interactive menu."""
        while True:
            self.show_main_menu()
            choice = input("\nEnter choice: ").strip()

            if choice == "0":
                print("\nReturning to chat...\n")
                break
            elif choice == "1":
                self.view_ontology()
            elif choice == "2":
                self.view_all_memories()
            elif choice == "3":
                self.view_memory_details()
            elif choice == "4":
                self.view_state_at_time()
            elif choice == "5":
                self.view_current_state()
            elif choice == "6":
                self.search_memories()
            elif choice == "7":
                self.view_reasoning_traces()
            else:
                print("Invalid choice. Please try again.")

    def view_ontology(self):
        """View the auto-generated ontology."""
        print("\n" + "-" * 60)
        print("  ONTOLOGY (Auto-Generated Entity Types)")
        print("-" * 60)

        types = self.agent.ontology.get_current_types()

        if not types:
            print("  No entity types have been created yet.")
            print("  Chat with the AI to generate some memories first!")
        else:
            for et in types:
                stability = "STABLE" if et.is_stable else "evolving"
                print(f"\n  [{et.name.upper()}] ({stability})")
                print(f"    Instances: {et.instance_count}")
                print(f"    Confidence: {et.confidence:.0%}")
                if et.observed_attributes:
                    print(f"    Attributes:")
                    for attr_name, stats in et.observed_attributes.items():
                        print(f"      - {attr_name}: {stats.frequency*100:.0f}% frequency")

        input("\nPress Enter to continue...")

    def view_all_memories(self):
        """View all current memories."""
        print("\n" + "-" * 60)
        print("  ALL CURRENT MEMORIES")
        print("-" * 60)

        graph = self.agent.queries.graph_at_time(datetime.now(), user_id=self.agent.user_id)

        if not graph.memories:
            print("  No memories stored yet. Chat with the AI first!")
        else:
            memories_list = list(graph.memories.items())
            for i, (mid, mem) in enumerate(memories_list, 1):
                has_history = mem.previous_version_id is not None
                history_marker = " [UPDATED]" if has_history else ""
                attr_key = mem.metadata.get("attribute_key", "?") if mem.metadata else "?"
                category = mem.metadata.get("category", "") if mem.metadata else ""
                print(f"\n  {i}. [{attr_key}] {mem.content}{history_marker}")
                print(f"     Category: {category} | Created: {mem.valid_from.strftime('%H:%M:%S')}")

        input("\nPress Enter to continue...")

    def view_memory_details(self):
        """View details and history of a specific memory."""
        print("\n" + "-" * 60)
        print("  MEMORY DETAILS & VERSION HISTORY")
        print("-" * 60)

        graph = self.agent.queries.graph_at_time(datetime.now(), user_id=self.agent.user_id)

        if not graph.memories:
            print("  No memories to view.")
            input("\nPress Enter to continue...")
            return

        memories_list = list(graph.memories.items())
        print("\n  Available memories:")
        for i, (mid, mem) in enumerate(memories_list, 1):
            attr_key = mem.metadata.get("attribute_key", "?") if mem.metadata else "?"
            has_history = mem.previous_version_id is not None
            marker = " *" if has_history else ""
            print(f"  {i}. [{attr_key}] {mem.content[:40]}...{marker}")

        print("\n  (* = has version history)")

        try:
            choice = int(input("\n  Select memory number: ")) - 1
            if 0 <= choice < len(memories_list):
                memory_id, current_mem = memories_list[choice]
                attr_key = current_mem.metadata.get("attribute_key", "unknown") if current_mem.metadata else "unknown"
                category = current_mem.metadata.get("category", "unknown") if current_mem.metadata else "unknown"

                print(f"\n  {'='*50}")
                print(f"  MEMORY: {current_mem.content}")
                print(f"  {'='*50}")
                print(f"  Attribute Key: {attr_key}")
                print(f"  Category: {category}")
                print(f"  Memory ID: {memory_id}")
                print(f"  Current Version ID: {current_mem.version_id}")
                print(f"  Valid From: {current_mem.valid_from}")
                print(f"  Entity Type ID: {current_mem.entity_type_id or 'None'}")

                # Get version history
                history = self.agent.store.get_version_history(memory_id)

                if history and len(history.versions) > 1:
                    print(f"\n  VERSION HISTORY ({len(history.versions)} versions):")
                    print(f"  {'-'*50}")
                    for i, ver in enumerate(history.versions, 1):
                        status = "CURRENT" if ver.superseded_at is None else f"SUPERSEDED at {ver.superseded_at.strftime('%H:%M:%S')}"
                        print(f"\n  Version {i}: [{status}]")
                        print(f"    Content: {ver.content}")
                        print(f"    Valid From: {ver.valid_from.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"    Transaction Time: {ver.transaction_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        if ver.context_snapshot:
                            print(f"    Context: {ver.context_snapshot}")
                else:
                    print("\n  No version history (this is the original version)")

                # Show relationships
                relationships = graph.get_relationships_for(memory_id)
                if relationships:
                    print(f"\n  RELATIONSHIPS:")
                    for rel in relationships:
                        direction = "outgoing" if rel.source_memory_id == memory_id else "incoming"
                        other_id = rel.target_memory_id if direction == "outgoing" else rel.source_memory_id
                        other_mem = graph.memories.get(other_id)
                        other_content = other_mem.content[:30] if other_mem else "Unknown"
                        print(f"    [{direction}] --{rel.relation_name}--> {other_content}...")

        except (ValueError, IndexError):
            print("  Invalid selection.")

        input("\nPress Enter to continue...")

    def view_state_at_time(self):
        """View the memory state at a specific point in time."""
        print("\n" + "-" * 60)
        print("  TIME TRAVEL: View State at Time T")
        print("-" * 60)

        if not self.agent.memory_timestamps:
            print("  No timestamps recorded. Chat with the AI first!")
            input("\nPress Enter to continue...")
            return

        print("\n  Recorded timestamps:")
        unique_times = sorted(set(self.agent.memory_timestamps))
        for i, ts in enumerate(unique_times, 1):
            print(f"  {i}. {ts.strftime('%Y-%m-%d %H:%M:%S.%f')}")

        print(f"\n  {len(unique_times)+1}. Enter custom time")
        print(f"  {len(unique_times)+2}. Compare two times")

        try:
            choice = int(input("\n  Select option: "))

            if choice == len(unique_times) + 2:
                # Compare two times
                print("\n  Select first time:")
                t1_choice = int(input("  Time 1 number: ")) - 1
                print("  Select second time:")
                t2_choice = int(input("  Time 2 number: ")) - 1

                if 0 <= t1_choice < len(unique_times) and 0 <= t2_choice < len(unique_times):
                    time1 = unique_times[t1_choice]
                    time2 = unique_times[t2_choice]

                    graph1 = self.agent.queries.graph_at_time(time1, user_id=self.agent.user_id)
                    graph2 = self.agent.queries.graph_at_time(time2, user_id=self.agent.user_id)

                    print(f"\n  COMPARISON:")
                    print(f"  Time 1 ({time1.strftime('%H:%M:%S')}): {graph1.memory_count} memories")
                    print(f"  Time 2 ({time2.strftime('%H:%M:%S')}): {graph2.memory_count} memories")

                    # Find differences
                    ids1 = set(graph1.memories.keys())
                    ids2 = set(graph2.memories.keys())

                    added = ids2 - ids1
                    removed = ids1 - ids2

                    if added:
                        print(f"\n  Added in Time 2:")
                        for mid in added:
                            mem = graph2.memories[mid]
                            print(f"    + {mem.content[:50]}...")

                    if removed:
                        print(f"\n  Removed/Superseded by Time 2:")
                        for mid in removed:
                            mem = graph1.memories[mid]
                            print(f"    - {mem.content[:50]}...")
            else:
                # Single time view
                if choice == len(unique_times) + 1:
                    time_str = input("  Enter time (HH:MM:SS): ")
                    today = datetime.now().date()
                    target_time = datetime.strptime(f"{today} {time_str}", "%Y-%m-%d %H:%M:%S")
                elif 1 <= choice <= len(unique_times):
                    target_time = unique_times[choice - 1]
                else:
                    print("  Invalid selection.")
                    return

                graph = self.agent.queries.graph_at_time(target_time, user_id=self.agent.user_id)

                print(f"\n  STATE AT {target_time.strftime('%Y-%m-%d %H:%M:%S')}:")
                print(f"  Total memories: {graph.memory_count}")
                print(f"  Total relationships: {graph.relationship_count}")

                if graph.memories:
                    print(f"\n  Memories at this time:")
                    for mid, mem in graph.memories.items():
                        print(f"    - {mem.content[:60]}...")

        except (ValueError, IndexError) as e:
            print(f"  Error: {e}")

        input("\nPress Enter to continue...")

    def view_current_state(self):
        """View the current complete state graph."""
        print("\n" + "-" * 60)
        print("  CURRENT STATE GRAPH")
        print("-" * 60)

        graph = self.agent.queries.graph_at_time(datetime.now(), user_id=self.agent.user_id)

        print(f"\n  Total Memories: {graph.memory_count}")
        print(f"  Total Relationships: {graph.relationship_count}")

        if graph.memories:
            print(f"\n  MEMORIES:")
            for i, (mid, mem) in enumerate(graph.memories.items(), 1):
                attr_key = mem.metadata.get("attribute_key", "?") if mem.metadata else "?"
                print(f"    {i}. [{attr_key}] {mem.content[:50]}...")

        if graph.relationships:
            print(f"\n  RELATIONSHIPS:")
            for rel in graph.relationships:
                source = graph.memories.get(rel.source_memory_id)
                target = graph.memories.get(rel.target_memory_id)
                src_text = source.content[:20] if source else "?"
                tgt_text = target.content[:20] if target else "?"
                print(f"    {src_text}... --[{rel.relation_name}]--> {tgt_text}...")

        # Show ontology summary
        types = self.agent.ontology.get_current_types()
        if types:
            print(f"\n  ENTITY TYPES: {len(types)}")
            for et in types:
                print(f"    - {et.name}: {et.instance_count} instances")

        input("\nPress Enter to continue...")

    def search_memories(self):
        """Search memories by semantic similarity."""
        print("\n" + "-" * 60)
        print("  SEMANTIC MEMORY SEARCH")
        print("-" * 60)

        query = input("\n  Enter search query: ").strip()
        if not query:
            print("  No query provided.")
            input("\nPress Enter to continue...")
            return

        results = self.agent.recall(query, k=10)

        if not results:
            print("  No matching memories found.")
        else:
            print(f"\n  Found {len(results)} matching memories:")
            for i, mem in enumerate(results, 1):
                print(f"\n  {i}. {mem.content}")
                print(f"     Valid From: {mem.valid_from.strftime('%Y-%m-%d %H:%M:%S')}")

        input("\nPress Enter to continue...")

    def view_reasoning_traces(self):
        """View reasoning traces."""
        print("\n" + "-" * 60)
        print("  REASONING TRACES")
        print("-" * 60)
        print("  (Reasoning traces are created during complex decisions)")
        print("  Chat with the AI about topics that require reasoning to see traces.")
        input("\nPress Enter to continue...")


def main():
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Run: export OPENAI_API_KEY=your-api-key")
        sys.exit(1)

    openai_client = OpenAI(api_key=api_key)

    config = TemporalStoreConfig(
        connection_string=os.getenv(
            "DATABASE_URL",
            "postgresql://user:password@localhost:5432/temporal_demo"
        ),
        embedding_dims=1536,
    )

    store = BitemporalStore(config)
    store.initialize()

    agent = InteractiveTemporalAgent(openai_client, store)
    menu = InteractiveMenu(agent)

    print("=" * 60)
    print("  INTERACTIVE TEMPORAL MEMORY CHAT")
    print("=" * 60)
    print(f"\n  User ID: {agent.user_id}")
    print(f"  Org ID:  {agent.organization_id}")
    print(f"  Session: {agent.session_id}")
    print(f"  Scope:   {agent.default_scope}")
    print("\nCommands:")
    print("  /menu   - Open the temporal memory explorer")
    print("  /info   - Show current user/org/session info")
    print("  /scope  - Change memory scope (user/org/session/global)")
    print("  /reset  - Clear all memories and start fresh")
    print("  /quit   - Exit the application")
    print("\nStart chatting! Your conversations will be remembered.\n")
    print("-" * 60)

    try:
        while True:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "/quit":
                print("\nGoodbye!")
                break

            if user_input.lower() == "/menu":
                menu.run()
                continue

            if user_input.lower() == "/info":
                print(f"\n  User ID:      {agent.user_id}")
                print(f"  Org ID:       {agent.organization_id}")
                print(f"  Session ID:   {agent.session_id}")
                print(f"  Default Scope: {agent.default_scope}")
                print(f"  Memories:     {len(agent.attribute_memory_map)} tracked attributes\n")
                continue

            if user_input.lower() == "/scope":
                print(f"\n  Current scope: {agent.default_scope}")
                print("  Available scopes:")
                print("    1. user         - Memories visible only to this user")
                print("    2. organization - Memories visible to all users in org")
                print("    3. session      - Memories visible only in this session")
                print("    4. global       - Memories visible to everyone")
                choice = input("\n  Select scope (1-4) or Enter to cancel: ").strip()
                scope_map = {"1": "user", "2": "organization", "3": "session", "4": "global"}
                if choice in scope_map:
                    agent.default_scope = scope_map[choice]
                    print(f"\n  Scope changed to: {agent.default_scope}\n")
                else:
                    print("\n  Scope unchanged.\n")
                continue

            if user_input.lower() == "/reset":
                confirm = input("Are you sure you want to clear ALL memories? (yes/no): ").strip().lower()
                if confirm == "yes":
                    # Clear local state
                    agent.attribute_memory_map.clear()
                    agent.memory_timestamps.clear()
                    agent.ontology._invalidate_cache()
                    # Drop and recreate all tables
                    store.reset()
                    print("\n✓ All memories cleared! Starting fresh.\n")
                else:
                    print("\nReset cancelled.\n")
                continue

            # Process the message
            print("\n[Processing...]")
            response, extracted = agent.process_message(user_input)

            # Show what was extracted/updated
            if extracted:
                print("\n[Memory Updates]")
                for item in extracted:
                    print(f"  {item}")

            # Show the response
            print(f"\nAssistant: {response}")

    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
    finally:
        store.close()


if __name__ == "__main__":
    main()
