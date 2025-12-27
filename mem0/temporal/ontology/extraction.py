"""
LLM-based type extraction and resolution helpers.

Uses LLM to:
- Confirm if an entity matches an existing type
- Determine if a type should be created or merged
- Generate type descriptions
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Prompt for type resolution
TYPE_RESOLUTION_PROMPT = """You are analyzing entity types for a knowledge graph ontology.

Given a newly extracted entity and a list of existing similar types, determine if the entity matches an existing type or should create a new type.

New Entity:
- Name: {entity_name}
- Attributes: {attributes}
- Context: {context}

Existing Similar Types:
{existing_types}

Respond with a JSON object:
{{
    "decision": "MATCH" | "NEW" | "SUBTYPE",
    "matched_type_id": "<UUID if MATCH, null otherwise>",
    "parent_type_id": "<UUID if SUBTYPE, null otherwise>",
    "reasoning": "<brief explanation>",
    "suggested_name": "<name for new type if NEW or SUBTYPE>"
}}

Rules:
- Use MATCH if the entity clearly belongs to an existing type
- Use NEW if the entity represents a genuinely different concept
- Use SUBTYPE if the entity is a more specific version of an existing type
"""

# Prompt for generating type descriptions
TYPE_DESCRIPTION_PROMPT = """Generate a brief description for this entity type in a knowledge graph.

Type Name: {name}
Observed Attributes: {attributes}
Sample Values: {samples}

Respond with a single sentence description that would help classify future entities.
"""

# Prompt for relation property inference
RELATION_PROPERTIES_PROMPT = """Analyze this relationship type to determine its properties.

Relation Name: {relation_name}
Example instances:
{examples}

Respond with a JSON object:
{{
    "is_symmetric": true/false,
    "is_transitive": true/false,
    "inverse_name": "<name of inverse relation or null>",
    "reasoning": "<brief explanation>"
}}

Examples:
- "is_friend_of" is symmetric (if A is friend of B, B is friend of A)
- "is_parent_of" is not symmetric but has inverse "is_child_of"
- "is_ancestor_of" is transitive (if A ancestor of B, B ancestor of C, then A ancestor of C)
"""


def format_existing_types_for_prompt(types: List[Dict[str, Any]]) -> str:
    """Format existing types for the resolution prompt."""
    if not types:
        return "No similar types found."

    lines = []
    for t in types:
        attrs = ", ".join(t.get("attributes", {}).keys()) or "none"
        lines.append(f"- {t['name']} (ID: {t['type_id']}): {t.get('description', 'No description')}")
        lines.append(f"  Attributes: {attrs}")
        lines.append(f"  Instances: {t.get('instance_count', 0)}, Confidence: {t.get('confidence', 0):.2f}")
        lines.append("")

    return "\n".join(lines)


def parse_type_resolution_response(response: str) -> Dict[str, Any]:
    """Parse the LLM response for type resolution."""
    try:
        # Try to extract JSON from the response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            return json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse type resolution response: {response}")

    # Default to creating a new type if parsing fails
    return {
        "decision": "NEW",
        "matched_type_id": None,
        "parent_type_id": None,
        "reasoning": "Failed to parse LLM response, defaulting to new type",
        "suggested_name": None,
    }


def parse_relation_properties_response(response: str) -> Dict[str, Any]:
    """Parse the LLM response for relation properties."""
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            return json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse relation properties response: {response}")

    return {
        "is_symmetric": False,
        "is_transitive": False,
        "inverse_name": None,
        "reasoning": "Failed to parse LLM response, using defaults",
    }


def build_type_resolution_prompt(
    entity_name: str,
    attributes: Dict[str, Any],
    context: str,
    similar_types: List[Dict[str, Any]],
) -> str:
    """Build the prompt for type resolution."""
    return TYPE_RESOLUTION_PROMPT.format(
        entity_name=entity_name,
        attributes=json.dumps(attributes, indent=2),
        context=context[:500] if context else "No context available",
        existing_types=format_existing_types_for_prompt(similar_types),
    )


def build_type_description_prompt(
    name: str,
    attributes: Dict[str, Any],
    sample_values: Dict[str, List[Any]],
) -> str:
    """Build the prompt for generating a type description."""
    return TYPE_DESCRIPTION_PROMPT.format(
        name=name,
        attributes=json.dumps(list(attributes.keys())),
        samples=json.dumps(sample_values, indent=2, default=str),
    )


def build_relation_properties_prompt(
    relation_name: str,
    examples: List[Tuple[str, str, str]],
) -> str:
    """Build the prompt for inferring relation properties."""
    example_lines = [f"  ({src}, {relation_name}, {tgt})" for src, _, tgt in examples[:5]]
    return RELATION_PROPERTIES_PROMPT.format(
        relation_name=relation_name,
        examples="\n".join(example_lines),
    )


def normalize_type_name(name: str) -> str:
    """Normalize a type name to a consistent format."""
    # Convert to lowercase, replace spaces with underscores
    normalized = name.lower().strip()
    normalized = "_".join(normalized.split())
    # Remove special characters
    normalized = "".join(c if c.isalnum() or c == "_" else "" for c in normalized)
    return normalized


def extract_attributes_from_content(content: str, llm=None) -> Dict[str, Any]:
    """
    Extract potential attributes from content.

    If LLM is provided, uses it for extraction. Otherwise, uses simple heuristics.
    """
    # Simple heuristic extraction (LLM extraction would be more sophisticated)
    attributes = {}

    # Look for key-value patterns like "name: John" or "age is 30"
    import re

    patterns = [
        r"(\w+):\s*([^,\n]+)",  # key: value
        r"(\w+)\s+is\s+([^,\n]+)",  # key is value
        r"(\w+)\s*=\s*([^,\n]+)",  # key = value
    ]

    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for key, value in matches:
            key = key.lower().strip()
            value = value.strip()
            if key and value and len(key) < 30:
                attributes[key] = value

    return attributes
