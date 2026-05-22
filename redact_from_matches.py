import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml
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

    output_path = output_dir / f"{image_stem}-hybrid-redacted-{timestamp}.png"
    log_path = logs_dir / f"{image_stem}-hybrid-redaction-{timestamp}.json"
    return output_path, log_path


def validate_box(item, image_width, image_height):
    box = item.get("box")
    if not isinstance(box, dict):
        return None, "Missing box."

    required_keys = ["x1", "y1", "x2", "y2"]
    if any(key not in box for key in required_keys):
        return None, "Box is missing one or more coordinates."

    try:
        x1 = int(round(float(box["x1"])))
        y1 = int(round(float(box["y1"])))
        x2 = int(round(float(box["x2"])))
        y2 = int(round(float(box["y2"])))
    except (TypeError, ValueError):
        return None, "Box coordinates must be numbers."

    if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
        return None, "Box is outside image bounds."

    if x2 <= x1 or y2 <= y1:
        return None, "Box must have positive width and height."

    return {
        "field_id": item.get("field_id"),
        "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "source_fragment_ids": item.get("source_fragment_ids", []),
        "confidence": item.get("confidence"),
    }, None


def validate_redaction_boxes(redaction_boxes, image_width, image_height):
    valid_boxes = []
    rejected_boxes = []

    for index, item in enumerate(redaction_boxes):
        valid_box, error = validate_box(item, image_width, image_height)
        if valid_box:
            valid_boxes.append(valid_box)
        else:
            rejected_boxes.append(
                {
                    "index": index,
                    "field_id": item.get("field_id") if isinstance(item, dict) else None,
                    "error": error,
                }
            )

    return valid_boxes, rejected_boxes


def redact_image(image_path, output_path, boxes):
    with Image.open(image_path) as image:
        redacted = image.convert("RGB")
        draw = ImageDraw.Draw(redacted)

        for item in boxes:
            box = item["box"]
            draw.rectangle([box["x1"], box["y1"], box["x2"], box["y2"]], fill="black")

        redacted.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Redact an image using boxes from a field-match log.")
    parser.add_argument("--image", required=True, help="Path to the source image.")
    parser.add_argument("--matches-log", required=True, help="Path to a field-matches JSON log.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    args = parser.parse_args()

    image_path = Path(args.image)
    matches_log_path = Path(args.matches_log)
    config_path = Path(args.config)

    config = load_yaml(config_path)
    matches_manifest = load_json(matches_log_path)
    output_path, log_path = make_run_paths(config, image_path)

    manifest = {
        "image": str(image_path),
        "matches_log": str(matches_log_path),
        "config": str(config_path),
        "output": str(output_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "started",
    }

    try:
        with Image.open(image_path) as image:
            image_width, image_height = image.size

        redaction_boxes = matches_manifest.get("redaction_boxes", [])
        valid_boxes, rejected_boxes = validate_redaction_boxes(
            redaction_boxes,
            image_width,
            image_height,
        )
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
        print(f"Saved hybrid redacted image: {output_path}")
        print(f"Saved hybrid redaction log: {log_path}")
    except Exception as error:
        manifest.update(
            {
                "status": "error",
                "error": str(error),
                "error_type": type(error).__name__,
            }
        )
        print(f"Error. Saved hybrid redaction log: {log_path}")
    finally:
        save_json(log_path, manifest)


if __name__ == "__main__":
    main()
