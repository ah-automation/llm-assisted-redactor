import argparse
import base64
from io import BytesIO
import json
import math
from datetime import datetime
from pathlib import Path

import yaml
from openai import OpenAI
from PIL import Image, ImageDraw


class VlmResponseError(Exception):
    def __init__(self, message, raw_response):
        super().__init__(message)
        self.raw_response = raw_response


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
    for field_id, override in (document_definition.get("field_overrides") or {}).items():
        fields[field_id] = merge_dicts(fields.get(field_id, {}), override)

    document_definition["fields"] = fields
    document_definition.pop("field_overrides", None)
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


def load_config(config_path):
    return load_yaml(config_path)


def image_to_data_url(image_path):
    with Image.open(image_path) as image:
        png_image = image.convert("RGB")
        buffer = BytesIO()
        png_image.save(buffer, format="PNG")

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def join_text_items(items):
    if isinstance(items, list):
        return "; ".join(str(item) for item in items)
    return str(items or "")


def build_config_prompt(pii_targets, image_width, image_height):
    target_lines = []
    for target in pii_targets:
        target_lines.append(
            "\n".join(
                [
                    f"- id: {target.get('id')}",
                    f"  label: {target.get('label')}",
                    f"  description: {target.get('description')}",
                    f"  hints: {join_text_items(target.get('hints', []))}",
                ]
            )
        )

    targets = "\n".join(target_lines)
    return (
        "Return the final JSON immediately. Do not explain your work.\n"
        "Do not reason step by step. Do not describe or transcribe text from the image.\n"
        "You are helping redact PII from one image.\n"
        f"The image size is {image_width} pixels wide by {image_height} pixels tall.\n"
        "Use pixel coordinates from this exact image size.\n"
        "Coordinate origin is the top-left corner: x increases to the right, y increases downward.\n"
        "Each box should tightly cover the visible text/value region for that target, with only small padding.\n"
        "Do not box labels unless the label and value cannot be separated.\n"
        "Do not box portraits, backgrounds, icons, or decorative elements.\n"
        "Find only these configured PII targets:\n"
        f"{targets}\n\n"
        "If a target appears more than once, return one detection for each appearance.\n"
        "Return only valid JSON. Do not include markdown or explanations.\n"
        "Do not transcribe or include actual PII text values in the response.\n"
        "For each detection, return the configured target id as target_id.\n"
        "If no configured PII targets are visible, return {\"boxes\": []}.\n"
        "Use this exact shape:\n"
        '{ "boxes": ['
        '{ "target_id": "dob", "x1": 0, "y1": 0, "x2": 100, "y2": 50 }'
        "] }\n"
        "Coordinates must be pixel coordinates in the original image."
    )


def build_document_field_prompt(document_definition, field, image_width, image_height):
    return (
        "Return the final JSON immediately. Do not explain your work.\n"
        "Do not reason step by step. Do not describe or transcribe text from the image.\n"
        f"The image size is {image_width} pixels wide by {image_height} pixels tall.\n"
        "Use pixel coordinates from this exact image size.\n"
        "Coordinate origin is the top-left corner: x increases to the right, y increases downward.\n"
        "Return boxes around the requested field value only unless instructed otherwise.\n"
        "Use tight boxes with only small padding.\n"
        "Do not include actual PII text values in the response.\n\n"
        f"Document type: {document_definition.get('label')}\n"
        f"Document description: {document_definition.get('description')}\n"
        f"Document hints: {join_text_items(document_definition.get('document_hints', []))}\n\n"
        "Locate only this one configured field:\n"
        f"- id: {field.get('id')}\n"
        f"  label: {field.get('label')}\n"
        f"  type: {field.get('type')}\n"
        f"  description: {field.get('description')}\n"
        f"  anchors: {join_text_items(field.get('anchors', []))}\n"
        f"  match_hints: {join_text_items(field.get('match_hints', []))}\n"
        f"  redaction: {field.get('redaction', {})}\n\n"
        "If the field appears more than once, return one detection for each appearance.\n"
        "If this field is not visible, return {\"boxes\": []}.\n"
        "Return only valid JSON. Do not include markdown or explanations.\n"
        "Use this exact shape:\n"
        '{ "boxes": ['
        f'{{ "target_id": "{field.get("id")}", "x1": 0, "y1": 0, "x2": 100, "y2": 50 }}'
        "] }\n"
        "Every returned target_id must exactly match the configured field id."
    )


def call_vlm(config, image_path, prompt):
    vlm_config = config["vlm"]
    client = OpenAI(
        base_url=vlm_config["base_url"],
        api_key=vlm_config.get("api_key", "lm-studio"),
    )

    response = client.chat.completions.create(
        model=vlm_config["model"],
        temperature=vlm_config.get("temperature", 0),
        max_tokens=vlm_config.get("max_tokens", 1000),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a direct JSON bounding-box detector. "
                    "Return only the requested JSON object. "
                    "Never include actual PII values."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image_path)},
                    },
                ],
            }
        ],
    )

    choice = response.choices[0]
    content = choice.message.content

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

    return content or "", diagnostic


def parse_boxes(raw_response):
    if not raw_response.strip():
        raise ValueError("VLM returned an empty response.")

    parsed = json.loads(raw_response)
    if not isinstance(parsed, dict):
        raise ValueError("JSON response must be an object.")

    boxes = parsed.get("boxes")
    if not isinstance(boxes, list):
        raise ValueError("JSON response must contain a 'boxes' list.")
    return boxes


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def get_target_ids(pii_targets):
    return {
        target.get("id")
        for target in pii_targets
        if isinstance(target, dict) and isinstance(target.get("id"), str)
    }


def safe_box_for_log(box):
    if not isinstance(box, dict):
        return box

    safe_keys = {"target_id", "x1", "y1", "x2", "y2"}
    logged_box = {key: box.get(key) for key in safe_keys if key in box}
    extra_keys = sorted(key for key in box if key not in safe_keys)

    if extra_keys:
        logged_box["unexpected_keys"] = extra_keys

    return logged_box


def validate_box(box, image_width, image_height, target_ids):
    if not isinstance(box, dict):
        return None, "Box must be an object."

    target_id = box.get("target_id")
    if not isinstance(target_id, str) or not target_id.strip():
        return None, "Box target_id must be a non-empty string."
    target_id = target_id.strip()

    if target_id not in target_ids:
        return None, f"Unknown target_id '{target_id}'."

    coordinates = {}
    for key in ["x1", "y1", "x2", "y2"]:
        value = box.get(key)
        if not is_number(value):
            return None, f"Coordinate '{key}' must be a number."
        coordinates[key] = int(round(value))

    x1 = coordinates["x1"]
    y1 = coordinates["y1"]
    x2 = coordinates["x2"]
    y2 = coordinates["y2"]

    if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
        return None, "Box is outside the image bounds."

    if x2 <= x1 or y2 <= y1:
        return None, "Box must have positive width and height."

    return {"target_id": target_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2}, None


def validate_boxes(boxes, image_width, image_height, target_ids):
    valid_boxes = []
    rejected_boxes = []

    for index, box in enumerate(boxes):
        valid_box, error = validate_box(box, image_width, image_height, target_ids)
        if valid_box:
            valid_boxes.append(valid_box)
        else:
            rejected_boxes.append(
                {
                    "index": index,
                    "box": safe_box_for_log(box),
                    "error": error,
                }
            )

    return valid_boxes, rejected_boxes


def redact_image(image_path, output_path, boxes):
    with Image.open(image_path) as image:
        redacted = image.convert("RGB")
        draw = ImageDraw.Draw(redacted)
        for box in boxes:
            draw.rectangle(
                [box["x1"], box["y1"], box["x2"], box["y2"]],
                fill="black",
            )
        redacted.save(output_path)


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

    output_path = output_dir / f"{image_stem}-redacted-{timestamp}.png"
    debug_path = output_dir / f"{image_stem}-debug-{timestamp}.png"
    log_path = logs_dir / f"{image_stem}-manifest-{timestamp}.json"
    return output_path, debug_path, log_path


def draw_box_label(draw, box):
    label = box["target_id"]
    x1 = box["x1"]
    y1 = box["y1"]
    text_top = max(0, y1 - 14)
    draw.rectangle([x1, text_top, x1 + 8 * len(label) + 6, text_top + 13], fill="black")
    draw.text((x1 + 3, text_top), label, fill="white")


def save_debug_overlay(image_path, debug_path, boxes):
    with Image.open(image_path) as image:
        overlay = image.convert("RGB")
        draw = ImageDraw.Draw(overlay)
        for box in boxes:
            draw.rectangle(
                [box["x1"], box["y1"], box["x2"], box["y2"]],
                outline="red",
                width=3,
            )
            draw_box_label(draw, box)
        overlay.save(debug_path)


def locate_with_config_targets(config, image_path, image_width, image_height):
    pii_targets = config.get("pii_targets")
    if not pii_targets:
        raise ValueError(
            "No pii_targets were found in config.yaml. "
            "Use run_pipeline.py for the current OCR + LLM workflow, "
            "or pass --document-definition to this legacy VLM script."
        )

    prompt = build_config_prompt(pii_targets, image_width, image_height)
    raw_response, vlm_diagnostic = call_vlm(config, image_path, prompt)
    try:
        boxes = parse_boxes(raw_response)
    except (json.JSONDecodeError, ValueError) as error:
        raise VlmResponseError(str(error), raw_response) from error

    target_ids = get_target_ids(pii_targets)
    valid_boxes, rejected_boxes = validate_boxes(boxes, image_width, image_height, target_ids)

    return {
        "valid_boxes": valid_boxes,
        "rejected_boxes": rejected_boxes,
        "vlm_diagnostic": vlm_diagnostic,
    }


def locate_with_document_definition(config, document_definition, image_path, image_width, image_height):
    valid_boxes = []
    rejected_boxes = []
    field_results = []

    for field in iter_fields(document_definition):
        field_id = field.get("id")
        prompt = build_document_field_prompt(document_definition, field, image_width, image_height)
        raw_response, vlm_diagnostic = call_vlm(config, image_path, prompt)

        field_result = {
            "field_id": field_id,
            "vlm_diagnostic": vlm_diagnostic,
            "status": "started",
        }

        try:
            boxes = parse_boxes(raw_response)
            field_valid_boxes, field_rejected_boxes = validate_boxes(
                boxes,
                image_width,
                image_height,
                {field_id},
            )
            valid_boxes.extend(field_valid_boxes)
            rejected_boxes.extend(
                {
                    "field_id": field_id,
                    **rejected_box,
                }
                for rejected_box in field_rejected_boxes
            )
            field_result.update(
                {
                    "status": "completed",
                    "valid_box_count": len(field_valid_boxes),
                    "rejected_box_count": len(field_rejected_boxes),
                }
            )
        except (json.JSONDecodeError, ValueError) as error:
            field_result.update(
                {
                    "status": "vlm_response_error",
                    "error": "VLM response was not valid JSON in the expected shape.",
                    "error_details": str(error),
                    "raw_response": raw_response,
                }
            )

        field_results.append(field_result)

    return {
        "valid_boxes": valid_boxes,
        "rejected_boxes": rejected_boxes,
        "field_results": field_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Redact PII from one image using a local VLM.")
    parser.add_argument("--image", required=True, help="Path to one image file.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument(
        "--document-definition",
        help="Optional YAML document definition for document-aware field localization.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    config_path = Path(args.config)
    document_definition_path = Path(args.document_definition) if args.document_definition else None

    config = load_config(config_path)
    document_definition = load_document_definition(document_definition_path) if document_definition_path else None
    output_path, debug_path, log_path = make_run_paths(config, image_path)

    manifest = {
        "image": str(image_path),
        "config": str(config_path),
        "output": str(output_path),
        "debug_overlay": str(debug_path),
        "model": config.get("vlm", {}).get("model"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "started",
    }
    if document_definition_path:
        manifest["document_definition"] = str(document_definition_path)
        manifest["document_type"] = document_definition.get("id")

    try:
        with Image.open(image_path) as image:
            image_width, image_height = image.size

        try:
            if document_definition:
                result = locate_with_document_definition(
                    config,
                    document_definition,
                    image_path,
                    image_width,
                    image_height,
                )
            else:
                result = locate_with_config_targets(config, image_path, image_width, image_height)
        except VlmResponseError as error:
            manifest.update(
                {
                    "status": "vlm_response_error",
                    "error": "VLM response was not valid JSON in the expected shape.",
                    "error_details": str(error),
                    "raw_response": error.raw_response,
                }
            )
            print(f"VLM response was malformed. Saved log: {log_path}")
            return

        valid_boxes = result["valid_boxes"]
        rejected_boxes = result["rejected_boxes"]
        if "vlm_diagnostic" in result:
            manifest["vlm_diagnostic"] = result["vlm_diagnostic"]
        if "field_results" in result:
            manifest["field_results"] = result["field_results"]

        save_debug_overlay(image_path, debug_path, valid_boxes)
        redact_image(image_path, output_path, valid_boxes)

        manifest.update(
            {
                "status": "completed",
                "image_size": {"width": image_width, "height": image_height},
                "valid_boxes": valid_boxes,
                "rejected_boxes": rejected_boxes,
                "valid_box_count": len(valid_boxes),
                "rejected_box_count": len(rejected_boxes),
            }
        )
        print(f"Saved redacted image: {output_path}")
        print(f"Saved log: {log_path}")
    except Exception as error:
        manifest.update(
            {
                "status": "error",
                "error": str(error),
                "error_type": type(error).__name__,
            }
        )
        print(f"Error. Saved log: {log_path}")
    finally:
        save_json(log_path, manifest)


if __name__ == "__main__":
    main()
