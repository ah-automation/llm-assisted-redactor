import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import yaml
from openai import OpenAI
from PIL import Image, ImageDraw


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def make_run_paths(config, image_path):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    image_stem = image_path.stem

    output_dir = Path(config.get("output_dir", "output"))
    logs_dir = Path(config.get("logs_dir", "logs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    overlay_path = output_dir / f"{image_stem}-field-matches-{timestamp}.png"
    log_path = logs_dir / f"{image_stem}-field-matches-{timestamp}.json"
    return overlay_path, log_path


def compact_fields(document_definition):
    fields = []
    for field in document_definition.get("fields", []):
        fields.append(
            {
                "id": field.get("id"),
                "label": field.get("label"),
                "description": field.get("description"),
                "anchors": field.get("anchors", []),
                "location_hints": field.get("location_hints", []),
                "redact_instruction": field.get("redact_instruction"),
            }
        )
    return fields


def compact_fragments(ocr_manifest):
    fragments = []
    for fragment in ocr_manifest.get("fragments", []):
        item = {
            "id": fragment.get("id"),
            "box": fragment.get("box"),
            "confidence": fragment.get("confidence"),
        }
        if "text" in fragment:
            item["text"] = fragment.get("text")
        elif "text_length" in fragment:
            item["text_length"] = fragment.get("text_length")
        fragments.append(item)
    return fragments


def build_association_prompt(document_definition, ocr_manifest):
    request = {
        "document": {
            "id": document_definition.get("id"),
            "label": document_definition.get("label"),
            "description": document_definition.get("description"),
            "hints": document_definition.get("document_hints", []),
        },
        "fields": compact_fields(document_definition),
        "ocr_fragments": compact_fragments(ocr_manifest),
    }

    return (
        "You are matching OCR fragments to configured document fields.\n"
        "The OCR text may contain typos. Use field anchors, layout hints, text similarity, and coordinates.\n"
        "Return only valid JSON. Do not include markdown or explanations.\n"
        "Do not transcribe or correct any PII values.\n"
        "Use OCR fragment ids only.\n"
        "For each field, choose the OCR fragments that contain the field value.\n"
        "Anchor fragments are optional and should be labels/headings near the value.\n"
        "If a field is not visible or cannot be matched, use empty arrays.\n"
        "Use this exact JSON shape:\n"
        "{\n"
        '  "matches": [\n'
        "    {\n"
        '      "field_id": "passport_number",\n'
        '      "value_fragment_ids": ["ocr_0001"],\n'
        '      "anchor_fragment_ids": ["ocr_0002"],\n'
        '      "confidence": 0.0,\n'
        '      "notes": "short non-PII reason"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Input JSON:\n"
        f"{json.dumps(request, indent=2)}"
    )


def build_field_association_prompt(document_definition, field, ocr_manifest):
    request = {
        "document": {
            "id": document_definition.get("id"),
            "label": document_definition.get("label"),
            "description": document_definition.get("description"),
        },
        "field": {
            "id": field.get("id"),
            "label": field.get("label"),
            "description": field.get("description"),
            "anchors": field.get("anchors", []),
            "location_hints": field.get("location_hints", []),
            "redact_instruction": field.get("redact_instruction"),
        },
        "ocr_fragments": compact_fragments(ocr_manifest),
    }

    return (
        "You are matching OCR fragments to one configured document field.\n"
        "The OCR text may contain typos. Use field anchors, layout hints, text similarity, and coordinates.\n"
        "Return only valid JSON. Do not include markdown or explanations.\n"
        "Do not transcribe or correct any PII values.\n"
        "Use OCR fragment ids only.\n"
        "Choose the OCR fragments that contain the field value.\n"
        "Anchor fragments are optional and should be labels/headings near the value.\n"
        "If the field is not visible or cannot be matched, use empty arrays.\n"
        "Return exactly one match object in the matches array.\n"
        "Use this exact JSON shape:\n"
        "{\n"
        '  "matches": [\n'
        "    {\n"
        f'      "field_id": "{field.get("id")}",\n'
        '      "value_fragment_ids": ["ocr_0001"],\n'
        '      "anchor_fragment_ids": ["ocr_0002"],\n'
        '      "confidence": 0.0,\n'
        '      "notes": "short non-PII reason"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Input JSON:\n"
        f"{json.dumps(request, separators=(',', ':'))}"
    )


def call_llm(config, prompt):
    vlm_config = config["vlm"]
    client = OpenAI(
        base_url=vlm_config["base_url"],
        api_key=vlm_config.get("api_key", "lm-studio"),
    )

    response = client.chat.completions.create(
        model=vlm_config["model"],
        temperature=vlm_config.get("temperature", 0),
        max_tokens=vlm_config.get("max_tokens", 1000),
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


def associate_fields(config, document_definition, ocr_manifest):
    matches = []
    field_results = []

    for field in document_definition.get("fields", []):
        field_id = field.get("id")
        prompt = build_field_association_prompt(document_definition, field, ocr_manifest)
        raw_response, diagnostic = call_llm(config, prompt)
        field_result = {
            "field_id": field_id,
            "llm_diagnostic": diagnostic,
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
                    "raw_response": raw_response,
                }
            )

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


def validate_matches(matches, field_ids, fragment_ids):
    valid_matches = []
    rejected_matches = []

    for index, match in enumerate(matches):
        if not isinstance(match, dict):
            rejected_matches.append({"index": index, "error": "Match must be an object."})
            continue

        field_id = match.get("field_id")
        if field_id not in field_ids:
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

        confidence = match.get("confidence")
        if not is_number(confidence):
            confidence = None

        valid_matches.append(
            {
                "field_id": field_id,
                "value_fragment_ids": value_ids,
                "anchor_fragment_ids": anchor_ids,
                "confidence": confidence,
                "notes": str(match.get("notes", ""))[:160],
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


def build_redaction_boxes(matches, fragments_by_id):
    redaction_boxes = []
    for match in matches:
        boxes = [
            fragments_by_id[fragment_id]["box"]
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
    parser.add_argument("--ocr-log", required=True, help="Path to an OCR JSON log created with --include-text.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--document-definition", required=True, help="Path to a document definition YAML file.")
    args = parser.parse_args()

    image_path = Path(args.image)
    config_path = Path(args.config)
    ocr_log_path = Path(args.ocr_log)
    document_definition_path = Path(args.document_definition)

    config = load_yaml(config_path)
    ocr_manifest = load_json(ocr_log_path)
    document_definition = load_yaml(document_definition_path)
    overlay_path, log_path = make_run_paths(config, image_path)

    manifest = {
        "image": str(image_path),
        "ocr_log": str(ocr_log_path),
        "ocr_source_image": ocr_manifest.get("image"),
        "config": str(config_path),
        "document_definition": str(document_definition_path),
        "document_type": document_definition.get("id"),
        "model": config.get("vlm", {}).get("model"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "match_overlay": str(overlay_path),
        "status": "started",
    }

    try:
        matches, field_results = associate_fields(config, document_definition, ocr_manifest)
        manifest["field_results"] = field_results

        field_ids = {field.get("id") for field in document_definition.get("fields", [])}
        fragments = ocr_manifest.get("fragments", [])
        fragments_by_id = {fragment["id"]: fragment for fragment in fragments}
        valid_matches, rejected_matches = validate_matches(matches, field_ids, set(fragments_by_id))
        redaction_boxes = build_redaction_boxes(valid_matches, fragments_by_id)
        save_match_overlay(image_path, overlay_path, redaction_boxes)

        manifest.update(
            {
                "status": "completed",
                "valid_matches": valid_matches,
                "rejected_matches": rejected_matches,
                "redaction_boxes": redaction_boxes,
                "valid_match_count": len(valid_matches),
                "rejected_match_count": len(rejected_matches),
                "redaction_box_count": len(redaction_boxes),
            }
        )
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
