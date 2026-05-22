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


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def image_to_data_url(image_path):
    with Image.open(image_path) as image:
        png_image = image.convert("RGB")
        buffer = BytesIO()
        png_image.save(buffer, format="PNG")

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def build_prompt(pii_targets, image_width, image_height):
    target_lines = []
    for target in pii_targets:
        hints = target.get("hints", [])
        if isinstance(hints, list):
            hints_text = "; ".join(str(hint) for hint in hints)
        else:
            hints_text = str(hints)

        target_lines.append(
            "\n".join(
                [
                    f"- id: {target.get('id')}",
                    f"  label: {target.get('label')}",
                    f"  description: {target.get('description')}",
                    f"  hints: {hints_text}",
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


def call_vlm(config, image_path, image_width, image_height):
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
                        "text": build_prompt(config["pii_targets"], image_width, image_height),
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
    log_path = logs_dir / f"{image_stem}-manifest-{timestamp}.json"
    return output_path, log_path


def main():
    parser = argparse.ArgumentParser(description="Redact PII from one image using a local VLM.")
    parser.add_argument("--image", required=True, help="Path to one image file.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    args = parser.parse_args()

    image_path = Path(args.image)
    config_path = Path(args.config)

    config = load_config(config_path)
    output_path, log_path = make_run_paths(config, image_path)

    manifest = {
        "image": str(image_path),
        "config": str(config_path),
        "output": str(output_path),
        "model": config.get("vlm", {}).get("model"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "started",
    }

    try:
        with Image.open(image_path) as image:
            image_width, image_height = image.size

        target_ids = get_target_ids(config["pii_targets"])
        raw_response, vlm_diagnostic = call_vlm(config, image_path, image_width, image_height)
        manifest["vlm_diagnostic"] = vlm_diagnostic
        try:
            boxes = parse_boxes(raw_response)
        except (json.JSONDecodeError, ValueError) as error:
            manifest.update(
                {
                    "status": "vlm_response_error",
                    "error": "VLM response was not valid JSON in the expected shape.",
                    "error_details": str(error),
                    "raw_response": raw_response,
                }
            )
            print(f"VLM response was malformed. Saved log: {log_path}")
            return

        valid_boxes, rejected_boxes = validate_boxes(boxes, image_width, image_height, target_ids)
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
