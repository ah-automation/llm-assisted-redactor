import argparse
import json
from pathlib import Path
import sys

from redactor import associate_fields


TOP_LEVEL_KEYS = {
    "extends",
    "id",
    "label",
    "description",
    "routing",
    "document_hints",
    "review",
    "fields",
    "field_overrides",
    "field_defaults",
}

FIELD_KEYS = {
    "id",
    "enabled",
    "label",
    "description",
    "anchors",
    "anchors_add",
    "match_hints",
    "match_hints_add",
    "max_value_fragments",
    "repeat_detection",
    "redaction",
}


class DocumentDefinitionValidationError(ValueError):
    def __init__(self, errors):
        super().__init__("Document definition failed validation.")
        self.errors = errors


def is_non_empty_string(value):
    return isinstance(value, str) and bool(value.strip())


def is_string_list(value):
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def is_non_empty_string_list(value):
    return is_string_list(value) and bool(value)


def add_unknown_key_errors(errors, path, value, allowed_keys):
    if not isinstance(value, dict):
        return

    for key in sorted(value):
        if key not in allowed_keys:
            errors.append(f"{path} contains unsupported key: {key}.")


def validate_string_list(errors, path, value, allow_empty=True):
    if value is None:
        return
    if not is_string_list(value):
        errors.append(f"{path} must be a list of strings.")
        return
    if not allow_empty and not value:
        errors.append(f"{path} must not be empty.")


def validate_routing(errors, routing):
    if routing is None:
        return
    if not isinstance(routing, dict):
        errors.append("routing must be an object.")
        return

    add_unknown_key_errors(errors, "routing", routing, {"markers"})
    markers = routing.get("markers")
    if markers is None:
        return
    if not isinstance(markers, dict):
        errors.append("routing.markers must be an object.")
        return

    add_unknown_key_errors(errors, "routing.markers", markers, {"strong", "weak", "strong_add", "weak_add"})
    for key in ("strong", "weak", "strong_add", "weak_add"):
        validate_string_list(errors, f"routing.markers.{key}", markers.get(key))


def validate_redaction(errors, field_path, redaction):
    if redaction is None:
        return
    if not isinstance(redaction, dict):
        errors.append(f"{field_path}.redaction must be an object.")
        return

    add_unknown_key_errors(errors, f"{field_path}.redaction", redaction, {"trim_leading_token_lengths"})
    token_lengths = redaction.get("trim_leading_token_lengths")
    if token_lengths is None:
        return
    if not isinstance(token_lengths, list) or not all(isinstance(item, int) and item > 0 for item in token_lengths):
        errors.append(f"{field_path}.redaction.trim_leading_token_lengths must be a list of positive integers.")


def validate_fields(errors, fields):
    if not isinstance(fields, dict) or not fields:
        errors.append("fields must be a non-empty object.")
        return set()

    field_ids = set(fields)
    for field_id, field in fields.items():
        field_path = f"fields.{field_id}"
        if not isinstance(field, dict):
            errors.append(f"{field_path} must be an object.")
            continue

        add_unknown_key_errors(errors, field_path, field, FIELD_KEYS)

        if field.get("enabled", True) is False:
            continue

        if not is_non_empty_string(field.get("label")):
            errors.append(f"{field_path}.label must be a non-empty string.")
        if not is_non_empty_string(field.get("description")):
            errors.append(f"{field_path}.description must be a non-empty string.")
        if not is_non_empty_string_list(field.get("anchors")):
            errors.append(f"{field_path}.anchors must be a non-empty list of strings.")

        validate_string_list(errors, f"{field_path}.match_hints", field.get("match_hints"))
        validate_string_list(errors, f"{field_path}.anchors_add", field.get("anchors_add"))
        validate_string_list(errors, f"{field_path}.match_hints_add", field.get("match_hints_add"))

        max_value_fragments = field.get("max_value_fragments")
        if max_value_fragments is not None and (
            not isinstance(max_value_fragments, int) or max_value_fragments < 1
        ):
            errors.append(f"{field_path}.max_value_fragments must be a positive integer.")

        repeat_detection = field.get("repeat_detection")
        if repeat_detection is not None and not isinstance(repeat_detection, bool):
            errors.append(f"{field_path}.repeat_detection must be true or false.")

        validate_redaction(errors, field_path, field.get("redaction"))

    return field_ids


def validate_review(errors, review, field_ids):
    if review is None:
        return
    if not isinstance(review, dict):
        errors.append("review must be an object.")
        return

    add_unknown_key_errors(errors, "review", review, {"required_fields", "required_groups"})
    required_fields = review.get("required_fields", [])
    validate_string_list(errors, "review.required_fields", required_fields)
    if isinstance(required_fields, list):
        for field_id in required_fields:
            if field_id not in field_ids:
                errors.append(f"review.required_fields contains unknown field id: {field_id}.")

    required_groups = review.get("required_groups", [])
    if not isinstance(required_groups, list):
        errors.append("review.required_groups must be a list.")
        return

    for index, group in enumerate(required_groups):
        group_path = f"review.required_groups[{index}]"
        if not isinstance(group, dict):
            errors.append(f"{group_path} must be an object.")
            continue

        add_unknown_key_errors(errors, group_path, group, {"id", "label", "any_of", "all_of"})
        if not is_non_empty_string(group.get("id")):
            errors.append(f"{group_path}.id must be a non-empty string.")
        if not is_non_empty_string(group.get("label")):
            errors.append(f"{group_path}.label must be a non-empty string.")

        any_of = group.get("any_of", [])
        all_of = group.get("all_of", [])
        validate_string_list(errors, f"{group_path}.any_of", any_of)
        validate_string_list(errors, f"{group_path}.all_of", all_of)
        if not any_of and not all_of:
            errors.append(f"{group_path} must define any_of or all_of.")

        for field_id in list(any_of or []) + list(all_of or []):
            if field_id not in field_ids:
                errors.append(f"{group_path} references unknown field id: {field_id}.")


def validate_field_overrides(errors, raw_definition, definition_path):
    overrides = raw_definition.get("field_overrides")
    if overrides is None:
        return
    if not isinstance(overrides, dict):
        errors.append("field_overrides must be an object.")
        return

    parent_path = raw_definition.get("extends")
    parent_field_ids = set()
    if parent_path:
        try:
            parent_definition = associate_fields.load_document_definition(definition_path.parent / parent_path)
            parent_field_ids = set(associate_fields.fields_to_map(parent_definition.get("fields", {})))
        except Exception as error:
            errors.append(f"extends could not be loaded for field override validation: {error}.")

    for field_id, override in overrides.items():
        if parent_field_ids and field_id not in parent_field_ids:
            errors.append(f"field_overrides contains unknown parent field id: {field_id}.")
        if not isinstance(override, dict):
            errors.append(f"field_overrides.{field_id} must be an object.")
            continue
        add_unknown_key_errors(errors, f"field_overrides.{field_id}", override, FIELD_KEYS)


def validate_document_definition(definition_path):
    definition_path = Path(definition_path)
    errors = []

    try:
        raw_definition = associate_fields.load_yaml(definition_path)
    except Exception as error:
        return {
            "status": "definition_error",
            "valid": False,
            "definition": str(definition_path),
            "document_type": None,
            "error_count": 1,
            "errors": [f"Could not read YAML: {error}."],
        }

    if not isinstance(raw_definition, dict):
        errors.append("Document definition must be a YAML object.")
        raw_definition = {}

    add_unknown_key_errors(errors, "document", raw_definition, TOP_LEVEL_KEYS)
    validate_field_overrides(errors, raw_definition, definition_path)

    try:
        document_definition = associate_fields.load_document_definition(definition_path)
    except Exception as error:
        return {
            "status": "definition_error",
            "valid": False,
            "definition": str(definition_path),
            "document_type": raw_definition.get("id"),
            "error_count": len(errors) + 1,
            "errors": errors + [f"Could not load resolved definition: {error}."],
        }

    for key in ("id", "label", "description"):
        if not is_non_empty_string(document_definition.get(key)):
            errors.append(f"{key} must be a non-empty string.")

    document_hints = document_definition.get("document_hints")
    validate_string_list(errors, "document_hints", document_hints)
    validate_routing(errors, document_definition.get("routing"))

    fields = associate_fields.fields_to_map(document_definition.get("fields", {}))
    field_ids = validate_fields(errors, fields)
    validate_review(errors, document_definition.get("review"), field_ids)

    valid = not errors
    return {
        "status": "completed" if valid else "definition_error",
        "valid": valid,
        "definition": str(definition_path),
        "document_type": document_definition.get("id"),
        "error_count": len(errors),
        "errors": errors,
    }


def require_valid_document_definition(definition_path):
    result = validate_document_definition(definition_path)
    if not result["valid"]:
        raise DocumentDefinitionValidationError(result["errors"])
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate a document definition YAML file.")
    parser.add_argument("--definition", required=True, help="Path to a document definition YAML file.")
    args = parser.parse_args()

    result = validate_document_definition(args.definition)
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
