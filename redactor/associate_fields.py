import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import yaml
from openai import OpenAI
from PIL import Image, ImageDraw


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def fields_to_map(fields):
    if isinstance(fields, dict):
        return fields

    mapped_fields = {}
    for field in fields or []:
        if isinstance(field, dict) and field.get("id"):
            field_copy = dict(field)
            field_id = field_copy.pop("id")
            mapped_fields[field_id] = field_copy
    return mapped_fields


def merge_dicts(base, override):
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_field_overrides(document_definition):
    fields = fields_to_map(document_definition.get("fields", {}))
    field_defaults = merge_dicts(
        {"max_value_fragments": 1},
        document_definition.get("field_defaults", {}),
    )

    for field_id, field in fields.items():
        fields[field_id] = merge_dicts(field_defaults, field)

    for field_id, override in (document_definition.get("field_overrides") or {}).items():
        fields[field_id] = merge_dicts(fields.get(field_id, {}), override)

    document_definition["fields"] = fields
    document_definition.pop("field_overrides", None)
    return apply_additions(document_definition)


def extend_list(target, key, additions_key):
    additions = target.pop(additions_key, [])
    if additions:
        target[key] = list(target.get(key, [])) + list(additions)


def apply_additions(document_definition):
    routing_markers = ((document_definition.get("routing") or {}).get("markers") or {})
    extend_list(routing_markers, "strong", "strong_add")
    extend_list(routing_markers, "weak", "weak_add")

    for field in fields_to_map(document_definition.get("fields", {})).values():
        extend_list(field, "anchors", "anchors_add")
        extend_list(field, "match_hints", "match_hints_add")

    return document_definition


def load_document_definition(path):
    path = Path(path)
    document_definition = load_yaml(path)

    parent_path = document_definition.get("extends")
    if parent_path:
        parent_definition = load_document_definition(path.parent / parent_path)
        child_definition = {
            key: value
            for key, value in document_definition.items()
            if key not in {"extends", "field_overrides"}
        }
        merged_definition = merge_dicts(parent_definition, child_definition)
        merged_definition["field_overrides"] = document_definition.get("field_overrides", {})
        return apply_field_overrides(merged_definition)

    return apply_field_overrides(document_definition)


def iter_fields(document_definition):
    fields = fields_to_map(document_definition.get("fields", {}))
    for field_id, field in fields.items():
        if field.get("enabled", True):
            field_with_id = dict(field)
            field_with_id["id"] = field_id
            yield field_with_id


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def is_debug_enabled(config):
    debug = config.get("debug") or {}
    return bool(debug.get("enabled", False)) if isinstance(debug, dict) else False


def add_raw_response_if_debug(target, config, raw_response):
    if is_debug_enabled(config):
        target["raw_response"] = raw_response


def get_llm_config(config):
    return config["llm"]


def make_run_paths(config, image_path):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_suffix = uuid4().hex[:8]
    image_stem = image_path.stem

    output_dir = Path(config.get("output_dir", "output"))
    logs_dir = Path(config.get("logs_dir", "logs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    overlay_path = output_dir / f"{timestamp}-{unique_suffix}-{image_stem}-field-matches.png"
    log_path = logs_dir / f"{timestamp}-{unique_suffix}-{image_stem}-field-matches.json"
    return overlay_path, log_path


def compact_box(box):
    if not isinstance(box, dict):
        return None
    return [box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2")]


def compact_fragments(document_definition, ocr_manifest):
    fragments = []
    for fragment in ocr_manifest.get("fragments", []):
        item = {
            "id": fragment.get("id"),
            "box": compact_box(fragment.get("box")),
        }
        if "text" in fragment:
            item["text"] = fragment.get("text", "")
        elif "text_length" in fragment:
            item["text_length"] = fragment.get("text_length")
        fragments.append(item)
    return fragments


def compact_fragment(fragment):
    item = {
        "id": fragment.get("id"),
        "box": compact_box(fragment.get("box")),
    }
    if "text" in fragment:
        item["text"] = fragment.get("text", "")
    elif "text_length" in fragment:
        item["text_length"] = fragment.get("text_length")
    return item


def build_field_association_prompt(document_definition, field, ocr_manifest):
    request = {
        "document": {
            "id": document_definition.get("id"),
            "label": document_definition.get("label"),
            "description": document_definition.get("description"),
            "hints": document_definition.get("document_hints", []),
        },
        "field": {
            "id": field.get("id"),
            "label": field.get("label"),
            "description": field.get("description"),
            "anchors": field.get("anchors", []),
            "match_hints": field.get("match_hints", []),
            "max_value_fragments": field.get("max_value_fragments"),
        },
        "ocr_fragments": compact_fragments(document_definition, ocr_manifest),
    }

    return (
        "Match one document field to OCR fragments. Return valid JSON only. Use ids only; never transcribe PII.\n"
        "OCR text may have typos, missing spaces, extra numbers, or multiple languages.\n"
        "Anchors are conceptual labels, not exact text. First choose label ids for anchor_fragment_ids.\n"
        "Then choose the nearest sensible value ids for value_fragment_ids, usually right of or below the label.\n"
        "If a label and value are combined in the same OCR fragment, use that id as both anchor and value.\n"
        "Do not select a label-only/header-only fragment as a value.\n"
        "Respect max_value_fragments. If not visible or unsure, use empty arrays.\n"
        f'Return exactly: {{"matches":[{{"field_id":"{field.get("id")}","value_fragment_ids":["ocr_0001"],"anchor_fragment_ids":["ocr_0002"],"confidence":0.0,"notes":"short non-PII reason"}}]}}\n'
        "Input JSON:\n"
        f"{json.dumps(request, separators=(',', ':'))}"
    )


def build_repeat_detection_prompt(document_definition, known_fields, remaining_fragments):
    request = {
        "document": {
            "id": document_definition.get("id"),
            "label": document_definition.get("label"),
            "description": document_definition.get("description"),
            "hints": document_definition.get("document_hints", []),
        },
        "known_fields": known_fields,
        "remaining_ocr_fragments": [compact_fragment(fragment) for fragment in remaining_fragments],
    }

    return (
        "Find repeats of already matched field values. Return valid JSON only. Use ids only; never transcribe PII.\n"
        "Do not discover new PII categories. Select only remaining fragments that are repeats or near-repeats of known_fields.\n"
        "Repeats may be unlabeled, smaller, vertical, low confidence, or slightly misread by OCR.\n"
        "Return one match per repeated occurrence. Each value_fragment_ids array should contain one repeated id.\n"
        "Use anchor_fragment_ids only for a nearby label; otherwise []. If none, return {\"matches\":[]}.\n"
        'Return shape: {"matches":[{"field_id":"license_number","value_fragment_ids":["ocr_0001"],"anchor_fragment_ids":[],"confidence":0.0,"notes":"short non-PII reason"}]}\n'
        "Input JSON:\n"
        f"{json.dumps(request, separators=(',', ':'))}"
    )


def call_llm(config, prompt):
    llm_config = get_llm_config(config)
    client = OpenAI(
        base_url=llm_config["base_url"],
        api_key=llm_config.get("api_key", "lm-studio"),
    )

    response = client.chat.completions.create(
        model=llm_config["model"],
        temperature=llm_config.get("temperature", 0),
        max_tokens=llm_config.get("max_tokens", 1000),
        response_format=get_match_response_format(),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a direct JSON field-association engine. "
                    "Return only the requested JSON object. "
                    "Never include actual PII values."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    choice = response.choices[0]
    content = choice.message.content or ""
    diagnostic = {
        "finish_reason": choice.finish_reason,
        "message_role": choice.message.role,
        "content_is_empty": not bool(content),
    }

    if response.usage:
        diagnostic["completion_tokens"] = response.usage.completion_tokens
        token_details = getattr(response.usage, "completion_tokens_details", None)
        if token_details:
            diagnostic["reasoning_tokens"] = getattr(token_details, "reasoning_tokens", None)

    return content, diagnostic


def get_match_response_format():
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "field_matches",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "matches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "field_id": {"type": "string"},
                                "value_fragment_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "anchor_fragment_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "confidence": {"type": "number"},
                                "notes": {"type": "string"},
                            },
                            "required": [
                                "field_id",
                                "value_fragment_ids",
                                "anchor_fragment_ids",
                                "confidence",
                                "notes",
                            ],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["matches"],
                "additionalProperties": False,
            },
        },
    }


def parse_matches(raw_response):
    cleaned_response = clean_json_response(raw_response)
    if not cleaned_response:
        raise ValueError("LLM returned an empty response.")

    parsed = json.loads(cleaned_response)
    if not isinstance(parsed, dict):
        raise ValueError("JSON response must be an object.")

    matches = parsed.get("matches")
    if not isinstance(matches, list):
        raise ValueError("JSON response must contain a 'matches' list.")

    return matches


def get_repeat_detection_field_ids(fields_by_id):
    return {
        field_id
        for field_id, field in fields_by_id.items()
        if field.get("repeat_detection", False)
    }


def build_known_repeat_fields(matches, fields_by_id, fragments_by_id, repeat_field_ids):
    known_fields = []
    for match in matches:
        field_id = match.get("field_id")
        value_ids = match.get("value_fragment_ids", [])
        if field_id not in repeat_field_ids:
            continue
        if not value_ids:
            continue

        field = fields_by_id.get(field_id, {})
        source_fragments = [
            fragments_by_id[fragment_id]
            for fragment_id in value_ids
            if fragment_id in fragments_by_id
        ]
        if not source_fragments:
            continue

        known_fields.append(
            {
                "field_id": field_id,
                "label": field.get("label"),
                "description": field.get("description"),
                "source_fragment_ids": [fragment["id"] for fragment in source_fragments],
                "source_fragments": [compact_fragment(fragment) for fragment in source_fragments],
            }
        )

    return known_fields


def find_repeat_matches(config, document_definition, ocr_manifest, valid_matches, fields_by_id, fragments_by_id):
    repeat_field_ids = get_repeat_detection_field_ids(fields_by_id)
    if not repeat_field_ids:
        return [], {
            "enabled": False,
            "status": "skipped",
            "reason": "No fields have repeat_detection enabled.",
        }

    known_fields = build_known_repeat_fields(valid_matches, fields_by_id, fragments_by_id, repeat_field_ids)
    matched_value_ids = {
        fragment_id
        for match in valid_matches
        for fragment_id in match.get("value_fragment_ids", [])
    }
    matched_anchor_ids = {
        fragment_id
        for match in valid_matches
        for fragment_id in match.get("anchor_fragment_ids", [])
    }
    remaining_fragments = [
        fragment
        for fragment in ocr_manifest.get("fragments", [])
        if fragment.get("id") not in matched_value_ids
        and fragment.get("id") not in matched_anchor_ids
    ]

    result = {
        "enabled": True,
        "status": "started",
        "known_field_count": len(known_fields),
        "configured_field_ids": sorted(repeat_field_ids),
        "remaining_fragment_count": len(remaining_fragments),
        "excluded_anchor_fragment_count": len(matched_anchor_ids),
    }

    if not known_fields or not remaining_fragments:
        result.update(
            {
                "status": "skipped",
                "reason": "No known fields or remaining OCR fragments were available for repeat detection.",
            }
        )
        return [], result

    prompt = build_repeat_detection_prompt(document_definition, known_fields, remaining_fragments)
    raw_response, diagnostic = call_llm(config, prompt)
    result["llm_diagnostic"] = diagnostic

    try:
        matches = parse_matches(raw_response)
        result.update(
            {
                "status": "completed",
                "match_count": len(matches),
            }
        )
        return matches, result
    except (json.JSONDecodeError, ValueError) as error:
        result.update(
            {
                "status": "llm_response_error",
                "error": "LLM response was not valid JSON in the expected shape.",
                "error_details": str(error),
            }
        )
        add_raw_response_if_debug(result, config, raw_response)
        return [], result


def associate_fields(config, document_definition, ocr_manifest):
    matches = []
    field_results = []

    for field in iter_fields(document_definition):
        field_id = field.get("id")
        prompt = build_field_association_prompt(document_definition, field, ocr_manifest)
        raw_response, diagnostic = call_llm(config, prompt)
        field_result = {
            "field_id": field_id,
            "llm_diagnostic": diagnostic,
            "association_context": {
                "strategy": "full_ocr_context",
                "fragment_count": len(ocr_manifest.get("fragments", [])),
            },
            "status": "started",
        }

        try:
            field_matches = parse_matches(raw_response)
            matches.extend(field_matches)
            field_result.update(
                {
                    "status": "completed",
                    "match_count": len(field_matches),
                }
            )
        except (json.JSONDecodeError, ValueError) as error:
            field_result.update(
                {
                    "status": "llm_response_error",
                    "error": "LLM response was not valid JSON in the expected shape.",
                    "error_details": str(error),
                }
            )
            add_raw_response_if_debug(field_result, config, raw_response)

        field_results.append(field_result)

    return matches, field_results


def clean_json_response(raw_response):
    text = raw_response.strip()
    if text.startswith("```json"):
        text = text[7:].strip()
    elif text.startswith("```"):
        text = text[3:].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def box_center(box):
    return ((box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2)


def box_distance(first_box, second_box):
    first_x, first_y = box_center(first_box)
    second_x, second_y = box_center(second_box)
    return math.hypot(first_x - second_x, first_y - second_y)


def limit_value_fragments(value_ids, anchor_ids, field, fragments_by_id):
    max_value_fragments = field.get("max_value_fragments")
    if not isinstance(max_value_fragments, int) or max_value_fragments < 1:
        return value_ids

    if len(value_ids) <= max_value_fragments:
        return value_ids

    anchor_boxes = [
        fragments_by_id[fragment_id]["box"]
        for fragment_id in anchor_ids
        if fragment_id in fragments_by_id and isinstance(fragments_by_id[fragment_id].get("box"), dict)
    ]

    def sort_key(fragment_id):
        fragment_box = fragments_by_id[fragment_id]["box"]
        if not anchor_boxes:
            return (0, value_ids.index(fragment_id))

        nearest_anchor_distance = min(box_distance(fragment_box, anchor_box) for anchor_box in anchor_boxes)
        fragment_center_y = box_center(fragment_box)[1]
        nearest_anchor_center_y = min(
            box_center(anchor_box)[1]
            for anchor_box in anchor_boxes
        )
        is_above_anchor = fragment_center_y < nearest_anchor_center_y
        return (is_above_anchor, nearest_anchor_distance, value_ids.index(fragment_id))

    return sorted(value_ids, key=sort_key)[:max_value_fragments]


def validate_matches(
    matches,
    fields_by_id,
    fragments_by_id,
    disallowed_value_ids=None,
    include_notes=False,
):
    valid_matches = []
    rejected_matches = []
    fragment_ids = set(fragments_by_id)
    disallowed_value_ids = set(disallowed_value_ids or [])

    for index, match in enumerate(matches):
        if not isinstance(match, dict):
            rejected_matches.append({"index": index, "error": "Match must be an object."})
            continue

        field_id = match.get("field_id")
        if field_id not in fields_by_id:
            rejected_matches.append({"index": index, "field_id": field_id, "error": "Unknown field_id."})
            continue

        value_ids = match.get("value_fragment_ids", [])
        anchor_ids = match.get("anchor_fragment_ids", [])
        if not isinstance(value_ids, list) or not isinstance(anchor_ids, list):
            rejected_matches.append({"index": index, "field_id": field_id, "error": "Fragment ids must be lists."})
            continue

        unknown_ids = sorted(
            fragment_id
            for fragment_id in value_ids + anchor_ids
            if fragment_id not in fragment_ids
        )
        if unknown_ids:
            rejected_matches.append(
                {
                    "index": index,
                    "field_id": field_id,
                    "error": "Unknown fragment ids.",
                    "unknown_fragment_ids": unknown_ids,
                }
            )
            continue

        disallowed_ids = sorted(fragment_id for fragment_id in value_ids if fragment_id in disallowed_value_ids)
        if disallowed_ids:
            rejected_matches.append(
                {
                    "index": index,
                    "field_id": field_id,
                    "error": "Disallowed fragment ids were selected as values.",
                    "disallowed_value_fragment_ids": disallowed_ids,
                }
            )
            continue

        value_ids = limit_value_fragments(value_ids, anchor_ids, fields_by_id[field_id], fragments_by_id)

        confidence = match.get("confidence")
        if not is_number(confidence):
            confidence = None

        valid_matches.append(
            {
                "field_id": field_id,
                "value_fragment_ids": value_ids,
                "anchor_fragment_ids": anchor_ids,
                "confidence": confidence,
                "notes": str(match.get("notes", ""))[:160] if include_notes else "",
            }
        )

    return valid_matches, rejected_matches


def merge_boxes(boxes):
    if not boxes:
        return None

    return {
        "x1": min(box["x1"] for box in boxes),
        "y1": min(box["y1"] for box in boxes),
        "x2": max(box["x2"] for box in boxes),
        "y2": max(box["y2"] for box in boxes),
    }


def trim_leading_token_box(fragment, box, token_lengths):
    text = str(fragment.get("text", ""))
    if " " not in text:
        return box

    leading_token, remaining_text = text.split(" ", 1)
    if len(leading_token) not in token_lengths or not leading_token.isupper() or not remaining_text.strip():
        return box

    text_length = len(text)
    trim_length = len(leading_token) + 1
    box_width = box["x2"] - box["x1"]
    trim_width = round(box_width * (trim_length / text_length))

    adjusted_box = dict(box)
    adjusted_box["x1"] = min(box["x2"] - 1, box["x1"] + trim_width)
    return adjusted_box


def get_fragment_redaction_box(fragment, field):
    box = dict(fragment["box"])
    redaction = field.get("redaction", {})
    token_lengths = redaction.get("trim_leading_token_lengths", [])
    if token_lengths:
        box = trim_leading_token_box(fragment, box, set(token_lengths))
    return box


def build_redaction_boxes(matches, fragments_by_id, fields_by_id):
    redaction_boxes = []
    for match in matches:
        field = fields_by_id.get(match["field_id"], {})
        boxes = [
            get_fragment_redaction_box(fragments_by_id[fragment_id], field)
            for fragment_id in match["value_fragment_ids"]
            if fragment_id in fragments_by_id
        ]
        merged_box = merge_boxes(boxes)
        if merged_box:
            redaction_boxes.append(
                {
                    "field_id": match["field_id"],
                    "box": merged_box,
                    "source_fragment_ids": match["value_fragment_ids"],
                    "confidence": match["confidence"],
                }
            )
    return redaction_boxes


def draw_label(draw, label, box):
    text_top = max(0, box["y1"] - 14)
    draw.rectangle([box["x1"], text_top, box["x1"] + 8 * len(label) + 6, text_top + 13], fill="black")
    draw.text((box["x1"] + 3, text_top), label, fill="white")


def save_match_overlay(image_path, overlay_path, redaction_boxes):
    with Image.open(image_path) as image:
        overlay = image.convert("RGB")
        draw = ImageDraw.Draw(overlay)

        for item in redaction_boxes:
            box = item["box"]
            draw.rectangle([box["x1"], box["y1"], box["x2"], box["y2"]], outline="blue", width=3)
            draw_label(draw, item["field_id"], box)

        overlay.save(overlay_path)


def main():
    parser = argparse.ArgumentParser(description="Associate OCR fragments with document fields using a local LLM.")
    parser.add_argument("--image", required=True, help="Path to the source image.")
    parser.add_argument("--ocr-log", required=True, help="Path to an OCR JSON log created while debug.enabled is true.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--document-definition", required=True, help="Path to a document definition YAML file.")
    args = parser.parse_args()

    image_path = Path(args.image)
    config_path = Path(args.config)
    ocr_log_path = Path(args.ocr_log)
    document_definition_path = Path(args.document_definition)

    config = load_yaml(config_path)
    debug_enabled = is_debug_enabled(config)
    ocr_manifest = load_json(ocr_log_path)
    document_definition = load_document_definition(document_definition_path)
    overlay_path, log_path = make_run_paths(config, image_path)

    manifest = {
        "image": str(image_path),
        "ocr_log": str(ocr_log_path),
        "ocr_source_image": ocr_manifest.get("image"),
        "config": str(config_path),
        "document_definition": str(document_definition_path),
        "document_type": document_definition.get("id"),
        "model": get_llm_config(config).get("model"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "debug": {"enabled": debug_enabled},
        "match_overlay": str(overlay_path) if debug_enabled else None,
        "status": "started",
    }

    try:
        matches, field_results = associate_fields(config, document_definition, ocr_manifest)
        manifest["field_results"] = field_results

        fields_by_id = {field.get("id"): field for field in iter_fields(document_definition)}
        fragments = ocr_manifest.get("fragments", [])
        fragments_by_id = {fragment["id"]: fragment for fragment in fragments}
        valid_matches, rejected_matches = validate_matches(
            matches,
            fields_by_id,
            fragments_by_id,
            include_notes=debug_enabled,
        )
        repeat_matches, repeat_result = find_repeat_matches(
            config,
            document_definition,
            ocr_manifest,
            valid_matches,
            fields_by_id,
            fragments_by_id,
        )
        valid_repeat_matches, rejected_repeat_matches = validate_matches(
            repeat_matches,
            fields_by_id,
            fragments_by_id,
            include_notes=debug_enabled,
            disallowed_value_ids={
                fragment_id
                for match in valid_matches
                for fragment_id in match.get("anchor_fragment_ids", [])
            },
        )
        redaction_boxes = build_redaction_boxes(valid_matches, fragments_by_id, fields_by_id)
        repeat_redaction_boxes = build_redaction_boxes(valid_repeat_matches, fragments_by_id, fields_by_id)
        redaction_boxes.extend(repeat_redaction_boxes)
        if debug_enabled:
            save_match_overlay(image_path, overlay_path, redaction_boxes)

        manifest.update(
            {
                "status": "completed",
                "valid_matches": valid_matches,
                "rejected_matches": rejected_matches,
                "repeat_detection": {
                    **repeat_result,
                    "valid_matches": valid_repeat_matches,
                    "rejected_matches": rejected_repeat_matches,
                    "redaction_boxes": repeat_redaction_boxes,
                    "valid_match_count": len(valid_repeat_matches),
                    "rejected_match_count": len(rejected_repeat_matches),
                    "redaction_box_count": len(repeat_redaction_boxes),
                },
                "redaction_boxes": redaction_boxes,
                "valid_match_count": len(valid_matches),
                "rejected_match_count": len(rejected_matches),
                "redaction_box_count": len(redaction_boxes),
            }
        )
        if debug_enabled:
            print(f"Saved field match overlay: {overlay_path}")
        print(f"Saved field match log: {log_path}")
    except Exception as error:
        manifest.update(
            {
                "status": "error",
                "error": str(error),
                "error_type": type(error).__name__,
            }
        )
        print(f"Error. Saved field match log: {log_path}")
    finally:
        save_json(log_path, manifest)


if __name__ == "__main__":
    main()
