import json

from app import AnalysisResponse
from provider import (
    GEMINI_CONFIG_VARIANTS,
    POTENTIALLY_PROBLEMATIC_SCHEMA_KEYS,
    GeminiProvider,
)


def schema_metrics(node: object) -> tuple[int, int, set[str]]:
    property_count = 0
    max_depth = 0
    found_keys: set[str] = set()
    definitions = node.get("$defs", {}) if isinstance(node, dict) else {}

    def visit(
        schema: object,
        depth: int,
        reference_stack: frozenset[str] = frozenset(),
    ) -> None:
        nonlocal property_count, max_depth
        if not isinstance(schema, dict):
            return
        max_depth = max(max_depth, depth)
        found_keys.update(schema.keys())
        reference = schema.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            definition_name = reference.removeprefix("#/$defs/")
            if (
                isinstance(definitions, dict)
                and definition_name in definitions
                and definition_name not in reference_stack
            ):
                visit(
                    definitions[definition_name],
                    depth,
                    reference_stack | {definition_name},
                )
        properties = schema.get("properties")
        if isinstance(properties, dict):
            property_count += len(properties)
            for child in properties.values():
                visit(child, depth + 1, reference_stack)
        for key in ("items", "additionalProperties"):
            child = schema.get(key)
            if isinstance(child, dict):
                visit(child, depth + 1, reference_stack)
        for key in ("anyOf", "oneOf", "prefixItems"):
            children = schema.get(key)
            if isinstance(children, list):
                for child in children:
                    visit(child, depth + 1, reference_stack)

    visit(node, 1)
    return property_count, max_depth, found_keys


def main() -> None:
    diagnostics = []
    for variant in GEMINI_CONFIG_VARIANTS:
        config = GeminiProvider.build_generation_config(
            AnalysisResponse,
            variant=variant,
        )
        property_count, max_depth, found_keys = schema_metrics(
            config.response_json_schema
        )
        diagnostics.append({
            "variant": variant,
            "config_fields": sorted(config.model_fields_set),
            "schema_property_count": property_count,
            "max_nesting_depth": max_depth,
            "potentially_problematic_keys": sorted(
                found_keys & POTENTIALLY_PROBLEMATIC_SCHEMA_KEYS
            ),
        })
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
