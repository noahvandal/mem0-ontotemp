"""
Emergent Ontology Manager.

Automatically generates and evolves entity/relation types from extracted data.
Types are versioned just like memories - they're never mutated, only superseded.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from mem0.temporal.ontology.types import (
    AttributeStats,
    EntityTypeVersion,
    OntologyGraph,
    RelationTypeVersion,
    TypeDerivation,
)
from mem0.temporal.ontology.extraction import (
    build_type_resolution_prompt,
    build_type_description_prompt,
    normalize_type_name,
    parse_type_resolution_response,
)

logger = logging.getLogger(__name__)


class EmergentOntologyManager:
    """
    Manages auto-generated ontology that evolves with data.

    Key features:
    - Resolves extracted entities to existing types or creates new ones
    - Tracks attribute statistics to build schema
    - Merges similar types when appropriate
    - Versions all type changes
    """

    def __init__(
        self,
        store,  # BitemporalStore
        llm=None,  # LLM for type resolution
        embedder=None,  # Embedder for type similarity
        type_similarity_threshold: float = 0.85,
        use_llm_for_resolution: bool = True,
    ):
        """
        Initialize the ontology manager.

        Args:
            store: BitemporalStore for persisting types
            llm: LLM instance for type resolution (optional)
            embedder: Embedder for computing type embeddings
            type_similarity_threshold: Cosine similarity threshold for type matching
            use_llm_for_resolution: Whether to use LLM to confirm type matches
        """
        self.store = store
        self.llm = llm
        self.embedder = embedder
        self.type_similarity_threshold = type_similarity_threshold
        self.use_llm_for_resolution = use_llm_for_resolution

        # Cache current types in memory for fast lookup
        self._entity_types_cache: Dict[UUID, EntityTypeVersion] = {}
        self._relation_types_cache: Dict[UUID, RelationTypeVersion] = {}
        self._cache_loaded = False

    def _ensure_cache_loaded(self) -> None:
        """Load current types into cache if not already loaded."""
        if self._cache_loaded:
            return

        self._load_current_types()
        self._cache_loaded = True

    def _load_current_types(self) -> None:
        """Load all current (non-superseded) types from the store."""
        # This would query the entity_type_versions and relation_type_versions tables
        # For now, initialize empty - types will be loaded on demand
        self._entity_types_cache = {}
        self._relation_types_cache = {}

    def _invalidate_cache(self) -> None:
        """Invalidate the type cache after modifications."""
        self._cache_loaded = False
        self._entity_types_cache = {}
        self._relation_types_cache = {}

    # =========================================================================
    # Entity Type Resolution
    # =========================================================================

    def resolve_entity_type(
        self,
        entity_name: str,
        attributes: Dict[str, Any],
        context: str = "",
    ) -> EntityTypeVersion:
        """
        Resolve an extracted entity to an existing type or create a new one.

        This is the main entry point for ontology auto-generation.

        Args:
            entity_name: Name/type of the extracted entity
            attributes: Attributes observed on the entity
            context: Context from which the entity was extracted

        Returns:
            The resolved or newly created EntityTypeVersion
        """
        self._ensure_cache_loaded()

        normalized_name = normalize_type_name(entity_name)

        # Step 1: Check for exact name match
        existing = self._find_type_by_name(normalized_name)
        if existing:
            return self._update_type_with_instance(existing, attributes)

        # Step 2: Find similar types by embedding
        similar_types = self._find_similar_types(entity_name, attributes)

        if not similar_types:
            # No similar types - create new
            return self._create_entity_type(normalized_name, attributes, context)

        # Step 3: Use LLM to decide (if enabled)
        if self.use_llm_for_resolution and self.llm:
            decision = self._resolve_with_llm(entity_name, attributes, context, similar_types)

            if decision["decision"] == "MATCH":
                matched_type = self._entity_types_cache.get(UUID(decision["matched_type_id"]))
                if matched_type:
                    return self._update_type_with_instance(matched_type, attributes)

            elif decision["decision"] == "SUBTYPE":
                parent_id = UUID(decision["parent_type_id"]) if decision.get("parent_type_id") else None
                return self._create_entity_type(
                    decision.get("suggested_name") or normalized_name,
                    attributes,
                    context,
                    parent_type_id=parent_id,
                )

        # Step 4: Fall back to similarity threshold
        best_match = similar_types[0]
        if best_match["similarity"] >= self.type_similarity_threshold:
            matched_type = self._entity_types_cache.get(best_match["type_id"])
            if matched_type:
                return self._update_type_with_instance(matched_type, attributes)

        # Step 5: Create new type
        return self._create_entity_type(normalized_name, attributes, context)

    def _find_type_by_name(self, name: str) -> Optional[EntityTypeVersion]:
        """Find an entity type by exact name match."""
        for et in self._entity_types_cache.values():
            if et.name.lower() == name.lower() and et.superseded_at is None:
                return et
        return None

    def _find_similar_types(
        self,
        entity_name: str,
        attributes: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Find entity types similar to the given entity."""
        if not self.embedder or not self._entity_types_cache:
            return []

        # Create a description for embedding
        description = f"{entity_name}: {', '.join(attributes.keys())}"
        query_embedding = self.embedder.embed(description)

        similar = []
        for type_id, et in self._entity_types_cache.items():
            if et.superseded_at is not None:
                continue
            if et.type_embedding is None:
                continue

            # Compute cosine similarity
            similarity = self._cosine_similarity(query_embedding, et.type_embedding)
            if similarity >= 0.5:  # Lower threshold for candidates
                similar.append({
                    "type_id": type_id,
                    "name": et.name,
                    "description": et.description,
                    "attributes": {k: v.model_dump() for k, v in et.observed_attributes.items()},
                    "instance_count": et.instance_count,
                    "confidence": et.confidence,
                    "similarity": similarity,
                })

        # Sort by similarity descending
        similar.sort(key=lambda x: x["similarity"], reverse=True)
        return similar[:5]  # Return top 5 candidates

    def _resolve_with_llm(
        self,
        entity_name: str,
        attributes: Dict[str, Any],
        context: str,
        similar_types: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Use LLM to determine type resolution."""
        prompt = build_type_resolution_prompt(entity_name, attributes, context, similar_types)

        try:
            response = self.llm.generate(prompt)
            return parse_type_resolution_response(response)
        except Exception as e:
            logger.warning(f"LLM type resolution failed: {e}")
            return {"decision": "NEW", "matched_type_id": None, "parent_type_id": None}

    def _create_entity_type(
        self,
        name: str,
        attributes: Dict[str, Any],
        context: str,
        parent_type_id: Optional[UUID] = None,
    ) -> EntityTypeVersion:
        """Create a new entity type."""
        type_id = uuid4()

        # Build attribute stats
        observed_attrs = {}
        for attr_name, attr_value in attributes.items():
            observed_attrs[attr_name] = AttributeStats(
                name=attr_name,
                frequency=1.0,
                types_observed=[type(attr_value).__name__],
                sample_values=[attr_value] if attr_value is not None else [],
                is_required=False,
            )

        # Generate embedding
        type_embedding = None
        if self.embedder:
            description = f"{name}: {', '.join(attributes.keys())}"
            type_embedding = self.embedder.embed(description)

        # Generate description
        description = f"Auto-generated type for '{name}'"
        if self.llm:
            try:
                desc_prompt = build_type_description_prompt(
                    name, attributes, {k: [v] for k, v in attributes.items()}
                )
                description = self.llm.generate(desc_prompt)
            except Exception as e:
                logger.warning(f"Failed to generate type description: {e}")

        entity_type = EntityTypeVersion(
            version_id=uuid4(),
            type_id=type_id,
            name=name,
            description=description,
            observed_attributes=observed_attrs,
            type_embedding=type_embedding,
            transaction_time=datetime.now(),
            superseded_at=None,
            instance_count=1,
            confidence=0.5,
            derivation=TypeDerivation.LLM_EXTRACTED,
            parent_type_id=parent_type_id,
        )

        # Save to store
        self._save_entity_type(entity_type)

        # Update cache
        self._entity_types_cache[type_id] = entity_type

        logger.info(f"Created new entity type: {name} (ID: {type_id})")
        return entity_type

    def _update_type_with_instance(
        self,
        existing: EntityTypeVersion,
        new_attributes: Dict[str, Any],
    ) -> EntityTypeVersion:
        """Update a type's statistics with a new instance."""
        # Merge attribute stats
        merged_attrs = dict(existing.observed_attributes)
        for attr_name, attr_value in new_attributes.items():
            if attr_name in merged_attrs:
                # Update existing attribute
                merged_attrs[attr_name] = merged_attrs[attr_name].update_with_value(
                    attr_value, existing.instance_count + 1
                )
            else:
                # New attribute
                merged_attrs[attr_name] = AttributeStats(
                    name=attr_name,
                    frequency=1.0 / (existing.instance_count + 1),
                    types_observed=[type(attr_value).__name__],
                    sample_values=[attr_value] if attr_value is not None else [],
                    is_required=False,
                )

        # Create new version
        new_version = EntityTypeVersion(
            version_id=uuid4(),
            type_id=existing.type_id,
            name=existing.name,
            description=existing.description,
            observed_attributes=merged_attrs,
            type_embedding=existing.type_embedding,
            transaction_time=datetime.now(),
            superseded_at=None,
            instance_count=existing.instance_count + 1,
            confidence=min(0.95, existing.confidence + 0.05),  # Confidence grows
            derivation=existing.derivation,
            parent_type_id=existing.parent_type_id,
        )

        # Supersede old version
        self._supersede_entity_type(existing.version_id)

        # Save new version
        self._save_entity_type(new_version)

        # Update cache
        self._entity_types_cache[existing.type_id] = new_version

        return new_version

    # =========================================================================
    # Relation Type Resolution
    # =========================================================================

    def resolve_relation_type(
        self,
        relation_name: str,
        source_type_id: UUID,
        target_type_id: UUID,
    ) -> RelationTypeVersion:
        """
        Resolve a relation to an existing type or create a new one.

        Args:
            relation_name: Name of the relation
            source_type_id: Type of the source entity
            target_type_id: Type of the target entity

        Returns:
            The resolved or newly created RelationTypeVersion
        """
        self._ensure_cache_loaded()

        normalized_name = normalize_type_name(relation_name)

        # Check for existing relation type with same name
        for rt in self._relation_types_cache.values():
            if rt.name.lower() == normalized_name and rt.superseded_at is None:
                # Update with new source/target types if needed
                if source_type_id not in rt.source_type_ids or target_type_id not in rt.target_type_ids:
                    return self._update_relation_type(rt, source_type_id, target_type_id)
                return rt

        # Create new relation type
        return self._create_relation_type(normalized_name, source_type_id, target_type_id)

    def _create_relation_type(
        self,
        name: str,
        source_type_id: UUID,
        target_type_id: UUID,
    ) -> RelationTypeVersion:
        """Create a new relation type."""
        type_id = uuid4()

        relation_type = RelationTypeVersion(
            version_id=uuid4(),
            type_id=type_id,
            name=name,
            description=f"Auto-generated relation type: {name}",
            source_type_ids={source_type_id},
            target_type_ids={target_type_id},
            is_symmetric=False,
            is_transitive=False,
            inverse_name=None,
            transaction_time=datetime.now(),
            superseded_at=None,
            instance_count=1,
            confidence=0.5,
            derivation=TypeDerivation.LLM_EXTRACTED,
        )

        # Save and cache
        self._save_relation_type(relation_type)
        self._relation_types_cache[type_id] = relation_type

        logger.info(f"Created new relation type: {name} (ID: {type_id})")
        return relation_type

    def _update_relation_type(
        self,
        existing: RelationTypeVersion,
        source_type_id: UUID,
        target_type_id: UUID,
    ) -> RelationTypeVersion:
        """Update a relation type with new source/target types."""
        new_sources = existing.source_type_ids | {source_type_id}
        new_targets = existing.target_type_ids | {target_type_id}

        new_version = RelationTypeVersion(
            version_id=uuid4(),
            type_id=existing.type_id,
            name=existing.name,
            description=existing.description,
            source_type_ids=new_sources,
            target_type_ids=new_targets,
            is_symmetric=existing.is_symmetric,
            is_transitive=existing.is_transitive,
            inverse_name=existing.inverse_name,
            transaction_time=datetime.now(),
            superseded_at=None,
            instance_count=existing.instance_count + 1,
            confidence=min(0.95, existing.confidence + 0.05),
            derivation=existing.derivation,
        )

        # Supersede old and save new
        self._supersede_relation_type(existing.version_id)
        self._save_relation_type(new_version)
        self._relation_types_cache[existing.type_id] = new_version

        return new_version

    # =========================================================================
    # Type Merging
    # =========================================================================

    def merge_types(
        self,
        type_id_1: UUID,
        type_id_2: UUID,
        canonical_name: Optional[str] = None,
    ) -> EntityTypeVersion:
        """
        Merge two entity types into one.

        Args:
            type_id_1: First type to merge
            type_id_2: Second type to merge
            canonical_name: Name for the merged type (defaults to type_1's name)

        Returns:
            The merged EntityTypeVersion
        """
        self._ensure_cache_loaded()

        type1 = self._entity_types_cache.get(type_id_1)
        type2 = self._entity_types_cache.get(type_id_2)

        if not type1 or not type2:
            raise ValueError(f"One or both types not found: {type_id_1}, {type_id_2}")

        # Merge attributes
        merged_attrs = dict(type1.observed_attributes)
        for attr_name, attr_stats in type2.observed_attributes.items():
            if attr_name in merged_attrs:
                # Merge stats (simplified - take union of values)
                existing = merged_attrs[attr_name]
                merged_attrs[attr_name] = AttributeStats(
                    name=attr_name,
                    frequency=max(existing.frequency, attr_stats.frequency),
                    types_observed=list(set(existing.types_observed + attr_stats.types_observed)),
                    sample_values=list(set(existing.sample_values + attr_stats.sample_values))[:5],
                    is_required=existing.is_required and attr_stats.is_required,
                )
            else:
                merged_attrs[attr_name] = attr_stats

        merged_type = EntityTypeVersion(
            version_id=uuid4(),
            type_id=type_id_1,  # Keep first type's ID as canonical
            name=canonical_name or type1.name,
            description=f"Merged from: {type1.name}, {type2.name}",
            observed_attributes=merged_attrs,
            type_embedding=type1.type_embedding,  # Could re-compute
            transaction_time=datetime.now(),
            superseded_at=None,
            instance_count=type1.instance_count + type2.instance_count,
            confidence=max(type1.confidence, type2.confidence),
            derivation=TypeDerivation.CLUSTERED,
        )

        # Supersede both old types
        self._supersede_entity_type(type1.version_id)
        self._supersede_entity_type(type2.version_id)

        # Save merged type
        self._save_entity_type(merged_type)
        self._entity_types_cache[type_id_1] = merged_type

        # Remove type2 from cache (it's now merged into type1)
        if type_id_2 in self._entity_types_cache:
            del self._entity_types_cache[type_id_2]

        logger.info(f"Merged types {type1.name} and {type2.name} into {merged_type.name}")
        return merged_type

    # =========================================================================
    # Queries
    # =========================================================================

    def get_current_types(self) -> List[EntityTypeVersion]:
        """Get all current (non-superseded) entity types."""
        self._ensure_cache_loaded()
        return [et for et in self._entity_types_cache.values() if et.superseded_at is None]

    def get_current_relation_types(self) -> List[RelationTypeVersion]:
        """Get all current (non-superseded) relation types."""
        self._ensure_cache_loaded()
        return [rt for rt in self._relation_types_cache.values() if rt.superseded_at is None]

    def get_ontology_graph(self) -> OntologyGraph:
        """Get the complete current ontology as a graph."""
        self._ensure_cache_loaded()
        return OntologyGraph(
            entity_types={k: v for k, v in self._entity_types_cache.items() if v.superseded_at is None},
            relation_types={k: v for k, v in self._relation_types_cache.items() if v.superseded_at is None},
            as_of=datetime.now(),
        )

    def find_similar_types(
        self,
        embedding: List[float],
        threshold: float = 0.7,
    ) -> List[EntityTypeVersion]:
        """Find entity types similar to the given embedding."""
        self._ensure_cache_loaded()

        similar = []
        for et in self._entity_types_cache.values():
            if et.superseded_at is not None or et.type_embedding is None:
                continue

            similarity = self._cosine_similarity(embedding, et.type_embedding)
            if similarity >= threshold:
                similar.append((similarity, et))

        # Sort by similarity descending
        similar.sort(key=lambda x: x[0], reverse=True)
        return [et for _, et in similar]

    # =========================================================================
    # Persistence Helpers
    # =========================================================================

    def _save_entity_type(self, et: EntityTypeVersion) -> None:
        """Save an entity type to the store."""
        # This would insert into entity_type_versions table
        # For now, the cache acts as our store
        pass

    def _supersede_entity_type(self, version_id: UUID) -> None:
        """Mark an entity type version as superseded."""
        # This would update superseded_at in the database
        pass

    def _save_relation_type(self, rt: RelationTypeVersion) -> None:
        """Save a relation type to the store."""
        pass

    def _supersede_relation_type(self, version_id: UUID) -> None:
        """Mark a relation type version as superseded."""
        pass

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)
