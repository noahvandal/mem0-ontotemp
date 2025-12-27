"""
Tests for the Ontology auto-generation system.

Tests:
- Entity type creation and resolution
- Type matching and merging
- Relation type handling
- Type versioning
"""

import pytest
from datetime import datetime
from uuid import uuid4

from mem0.temporal.ontology.types import (
    AttributeStats,
    EntityTypeVersion,
    RelationTypeVersion,
    TypeDerivation,
    OntologyGraph,
)
from mem0.temporal.ontology.extraction import (
    normalize_type_name,
    extract_attributes_from_content,
)


class TestEntityTypeVersion:
    """Tests for EntityTypeVersion model."""

    def test_create_entity_type(self):
        """Test creating an entity type."""
        et = EntityTypeVersion(
            type_id=uuid4(),
            name="Person",
            description="A human individual",
            observed_attributes={
                "name": AttributeStats(
                    name="name",
                    frequency=1.0,
                    types_observed=["str"],
                ),
            },
            instance_count=5,
            confidence=0.8,
        )

        assert et.name == "Person"
        assert et.instance_count == 5
        assert et.is_stable  # 5 instances, 0.8 confidence

    def test_type_stability(self):
        """Test is_stable property."""
        # Not stable - too few instances
        et1 = EntityTypeVersion(
            type_id=uuid4(),
            name="NewType",
            instance_count=2,
            confidence=0.8,
        )
        assert not et1.is_stable

        # Not stable - low confidence
        et2 = EntityTypeVersion(
            type_id=uuid4(),
            name="UncertainType",
            instance_count=10,
            confidence=0.5,
        )
        assert not et2.is_stable

        # Stable
        et3 = EntityTypeVersion(
            type_id=uuid4(),
            name="StableType",
            instance_count=10,
            confidence=0.8,
        )
        assert et3.is_stable

    def test_attribute_stats_update(self):
        """Test updating attribute stats with new values."""
        stats = AttributeStats(
            name="age",
            frequency=1.0,
            types_observed=["int"],
            sample_values=[25],
        )

        updated = stats.update_with_value(30, total_instances=2)

        assert "int" in updated.types_observed
        assert 25 in updated.sample_values
        assert 30 in updated.sample_values


class TestRelationTypeVersion:
    """Tests for RelationTypeVersion model."""

    def test_create_relation_type(self):
        """Test creating a relation type."""
        person_type_id = uuid4()
        company_type_id = uuid4()

        rt = RelationTypeVersion(
            type_id=uuid4(),
            name="works_at",
            description="Employment relationship",
            source_type_ids={person_type_id},
            target_type_ids={company_type_id},
            is_symmetric=False,
            is_transitive=False,
        )

        assert rt.name == "works_at"
        assert person_type_id in rt.source_type_ids
        assert company_type_id in rt.target_type_ids


class TestOntologyGraph:
    """Tests for OntologyGraph."""

    def test_get_entity_type_by_name(self):
        """Test looking up entity type by name."""
        type_id = uuid4()
        et = EntityTypeVersion(
            type_id=type_id,
            name="Person",
        )

        graph = OntologyGraph(
            entity_types={type_id: et},
        )

        found = graph.get_entity_type_by_name("Person")
        assert found is not None
        assert found.type_id == type_id

        # Case insensitive
        found_lower = graph.get_entity_type_by_name("person")
        assert found_lower is not None

        # Not found
        not_found = graph.get_entity_type_by_name("Company")
        assert not_found is None

    def test_get_valid_relations_for(self):
        """Test finding valid relations for an entity type."""
        person_type_id = uuid4()
        company_type_id = uuid4()

        rt1 = RelationTypeVersion(
            type_id=uuid4(),
            name="works_at",
            source_type_ids={person_type_id},
            target_type_ids={company_type_id},
        )
        rt2 = RelationTypeVersion(
            type_id=uuid4(),
            name="knows",
            source_type_ids={person_type_id},
            target_type_ids={person_type_id},
        )

        graph = OntologyGraph(
            relation_types={rt1.type_id: rt1, rt2.type_id: rt2},
        )

        # Person can participate in both relations
        person_relations = graph.get_valid_relations_for(person_type_id)
        assert len(person_relations) == 2

        # Company can only be target of works_at
        company_relations = graph.get_valid_relations_for(company_type_id)
        assert len(company_relations) == 1
        assert company_relations[0].name == "works_at"


class TestOntologyExtraction:
    """Tests for extraction helpers."""

    def test_normalize_type_name(self):
        """Test type name normalization."""
        assert normalize_type_name("Person") == "person"
        assert normalize_type_name("Software Engineer") == "software_engineer"
        assert normalize_type_name("  My Type  ") == "my_type"
        assert normalize_type_name("Type-With-Dashes") == "typewithdashes"

    def test_extract_attributes_from_content(self):
        """Test simple attribute extraction."""
        content = "name: John Doe, age: 30, role is developer"
        attrs = extract_attributes_from_content(content)

        assert "name" in attrs
        assert "age" in attrs
        assert "role" in attrs


class TestEmergentOntologyManager:
    """Tests for the EmergentOntologyManager."""

    @pytest.fixture
    def ontology_manager(self, bitemporal_store):
        """Create an ontology manager for testing."""
        from mem0.temporal.ontology.manager import EmergentOntologyManager

        return EmergentOntologyManager(
            store=bitemporal_store,
            llm=None,  # No LLM for unit tests
            embedder=None,
            type_similarity_threshold=0.85,
            use_llm_for_resolution=False,
        )

    def test_resolve_new_entity_type(self, ontology_manager):
        """Test resolving a completely new entity type."""
        et = ontology_manager.resolve_entity_type(
            entity_name="Person",
            attributes={"name": "John", "age": 30},
            context="User mentioned their name is John, age 30",
        )

        assert et.name == "person"
        assert "name" in et.observed_attributes
        assert "age" in et.observed_attributes
        assert et.instance_count == 1

    def test_resolve_existing_entity_type(self, ontology_manager):
        """Test resolving to an existing type."""
        # Create first instance
        et1 = ontology_manager.resolve_entity_type(
            entity_name="Person",
            attributes={"name": "John"},
            context="",
        )

        # Create second instance - should match existing
        et2 = ontology_manager.resolve_entity_type(
            entity_name="Person",
            attributes={"name": "Jane", "email": "jane@example.com"},
            context="",
        )

        assert et1.type_id == et2.type_id
        assert et2.instance_count == 2
        assert "email" in et2.observed_attributes  # New attribute added

    def test_get_current_types(self, ontology_manager):
        """Test getting all current types."""
        # Create some types
        ontology_manager.resolve_entity_type("Person", {"name": "John"}, "")
        ontology_manager.resolve_entity_type("Company", {"name": "Acme"}, "")

        types = ontology_manager.get_current_types()
        names = [t.name for t in types]

        assert "person" in names
        assert "company" in names

    def test_get_ontology_graph(self, ontology_manager):
        """Test getting the complete ontology graph."""
        ontology_manager.resolve_entity_type("Person", {}, "")
        ontology_manager.resolve_entity_type("Company", {}, "")

        graph = ontology_manager.get_ontology_graph()

        assert len(graph.entity_types) == 2
        assert graph.get_entity_type_by_name("person") is not None
