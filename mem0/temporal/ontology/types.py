"""
Ontology type models for auto-generated schema.

Entity and relation types are first-class versioned objects,
allowing the ontology to evolve with the data.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TypeDerivation(str, Enum):
    """How an entity/relation type was derived."""

    LLM_EXTRACTED = "LLM_EXTRACTED"  # Extracted by LLM from data
    CLUSTERED = "CLUSTERED"  # Merged from similar types
    MANUAL = "MANUAL"  # Manually defined
    INHERITED = "INHERITED"  # Derived from a parent type


class AttributeStats(BaseModel):
    """Statistics about an observed attribute on an entity type."""

    name: str = Field(..., description="Attribute name")
    frequency: float = Field(default=1.0, ge=0.0, le=1.0, description="How often this attribute appears")
    types_observed: List[str] = Field(default_factory=list, description="Data types observed (string, int, etc.)")
    sample_values: List[Any] = Field(default_factory=list, description="Sample values (up to 5)")
    is_required: bool = Field(default=False, description="Whether this attribute appears on all instances")

    def update_with_value(self, value: Any, total_instances: int) -> "AttributeStats":
        """Update stats with a new observed value."""
        new_samples = self.sample_values.copy()
        if len(new_samples) < 5 and value not in new_samples:
            new_samples.append(value)

        value_type = type(value).__name__
        new_types = self.types_observed.copy()
        if value_type not in new_types:
            new_types.append(value_type)

        return AttributeStats(
            name=self.name,
            frequency=self.frequency,  # Would need instance count to update properly
            types_observed=new_types,
            sample_values=new_samples,
            is_required=self.is_required,
        )


class EntityTypeVersion(BaseModel):
    """
    A version of an entity type in the ontology.

    Entity types are auto-generated from extracted entities and evolve as
    more instances are observed.
    """

    # Identity
    version_id: UUID = Field(default_factory=uuid4)
    type_id: UUID = Field(..., description="Stable identity across versions")

    # Definition
    name: str = Field(..., description="Type name (e.g., 'Person', 'Company')")
    description: str = Field(default="", description="Auto-generated or manual description")

    # Observed schema
    observed_attributes: Dict[str, AttributeStats] = Field(
        default_factory=dict,
        description="Attributes observed on instances of this type",
    )

    # Embedding for similarity matching
    type_embedding: Optional[List[float]] = Field(
        default=None,
        description="Embedding of the type definition for similarity matching",
    )

    # Bitemporal (simplified - just transaction time)
    transaction_time: datetime = Field(default_factory=datetime.now)
    superseded_at: Optional[datetime] = Field(default=None)

    # Metadata
    instance_count: int = Field(default=0, description="Number of instances of this type")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in this type definition")
    derivation: TypeDerivation = Field(default=TypeDerivation.LLM_EXTRACTED)

    # Hierarchy
    parent_type_id: Optional[UUID] = Field(default=None, description="Parent type if this is a subtype")

    class Config:
        frozen = True

    @property
    def is_stable(self) -> bool:
        """A type is stable if it has enough instances and high confidence."""
        return self.instance_count >= 3 and self.confidence >= 0.7


class RelationTypeVersion(BaseModel):
    """
    A version of a relation type in the ontology.

    Relation types define valid relationships between entity types.
    """

    # Identity
    version_id: UUID = Field(default_factory=uuid4)
    type_id: UUID = Field(..., description="Stable identity across versions")

    # Definition
    name: str = Field(..., description="Relation name (e.g., 'works_at', 'knows')")
    description: str = Field(default="")

    # Type constraints
    source_type_ids: Set[UUID] = Field(
        default_factory=set,
        description="Entity types that can be the source",
    )
    target_type_ids: Set[UUID] = Field(
        default_factory=set,
        description="Entity types that can be the target",
    )

    # Properties
    is_symmetric: bool = Field(default=False, description="If (a, r, b) implies (b, r, a)")
    is_transitive: bool = Field(default=False, description="If (a, r, b) and (b, r, c) implies (a, r, c)")
    inverse_name: Optional[str] = Field(default=None, description="Name of inverse relation")

    # Bitemporal
    transaction_time: datetime = Field(default_factory=datetime.now)
    superseded_at: Optional[datetime] = Field(default=None)

    # Metadata
    instance_count: int = Field(default=0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    derivation: TypeDerivation = Field(default=TypeDerivation.LLM_EXTRACTED)

    class Config:
        frozen = True


class OntologyGraph(BaseModel):
    """
    The complete ontology as a graph of types and their relationships.
    """

    # Types
    entity_types: Dict[UUID, EntityTypeVersion] = Field(
        default_factory=dict,
        description="Entity types keyed by type_id",
    )
    relation_types: Dict[UUID, RelationTypeVersion] = Field(
        default_factory=dict,
        description="Relation types keyed by type_id",
    )

    # Transaction time
    as_of: datetime = Field(default_factory=datetime.now)

    def get_entity_type(self, type_id: UUID) -> Optional[EntityTypeVersion]:
        """Get an entity type by ID."""
        return self.entity_types.get(type_id)

    def get_entity_type_by_name(self, name: str) -> Optional[EntityTypeVersion]:
        """Get an entity type by name."""
        for et in self.entity_types.values():
            if et.name.lower() == name.lower():
                return et
        return None

    def get_relation_type(self, type_id: UUID) -> Optional[RelationTypeVersion]:
        """Get a relation type by ID."""
        return self.relation_types.get(type_id)

    def get_valid_relations_for(self, entity_type_id: UUID) -> List[RelationTypeVersion]:
        """Get all relation types where this entity type can be source or target."""
        return [
            rt for rt in self.relation_types.values()
            if entity_type_id in rt.source_type_ids or entity_type_id in rt.target_type_ids
        ]

    def get_subtypes_of(self, parent_type_id: UUID) -> List[EntityTypeVersion]:
        """Get all entity types that are subtypes of the given type."""
        return [
            et for et in self.entity_types.values()
            if et.parent_type_id == parent_type_id
        ]
