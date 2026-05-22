import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml
from PIL import Image, ImageDraw
from rapidocr import RapidOCR


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def make_run_paths(config, image_path):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    image_stem = image_path.stem

    output_dir = Path(config.get("output_dir", "output"))
    logs_dir = Path(config.get("logs_dir", "logs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    debug_path = output_dir / f"{image_stem}-ocr-debug-{timestamp}.png"
    log_path = logs_dir / f"{image_stem}-ocr-{timestamp}.json"
    return debug_path, log_path


def to_number(value):
    return int(round(float(value)))


def normalize_box(box):
    points = [[to_number(x), to_number(y)] for x, y in box]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return {
        "points": points,
        "rect": {
            "x1": min(xs),
            "y1": min(ys),
            "x2": max(xs),
            "y2": max(ys),
        },
    }


def get_result_values(result):
    if hasattr(result, "boxes") and hasattr(result, "txts"):
        boxes = result.boxes if result.boxes is not None else []
        texts = result.txts if result.txts is not None else []
        scores = getattr(result, "scores", None)
        scores = scores if scores is not None else []
        return boxes, texts, scores

    if isinstance(result, tuple) and result:
        rows = result[0] or []
        boxes = [row[0] for row in rows]
        texts = [row[1][0] for row in rows]
        scores = [row[1][1] for row in rows]
        return boxes, texts, scores

    return [], [], []


def run_ocr(image_path):
    engine = RapidOCR()
    result = engine(str(image_path))
    boxes, texts, scores = get_result_values(result)

    fragments = []
    for index, box in enumerate(boxes):
        text = texts[index] if index < len(texts) else ""
        score = scores[index] if index < len(scores) else None
        normalized_box = normalize_box(box)
        fragments.append(
            {
                "id": f"ocr_{index + 1:04d}",
                "box": normalized_box["rect"],
                "points": normalized_box["points"],
                "confidence": float(score) if score is not None else None,
                "text": str(text),
            }
        )

    return fragments


def draw_label(draw, fragment):
    label = fragment["id"]
    x1 = fragment["box"]["x1"]
    y1 = fragment["box"]["y1"]
    text_top = max(0, y1 - 14)
    draw.rectangle([x1, text_top, x1 + 8 * len(label) + 6, text_top + 13], fill="black")
    draw.text((x1 + 3, text_top), label, fill="white")


def save_debug_overlay(image_path, debug_path, fragments):
    with Image.open(image_path) as image:
        overlay = image.convert("RGB")
        draw = ImageDraw.Draw(overlay)

        for fragment in fragments:
            points = [tuple(point) for point in fragment["points"]]
            if len(points) >= 4:
                draw.line(points + [points[0]], fill="red", width=2)
            else:
                box = fragment["box"]
                draw.rectangle([box["x1"], box["y1"], box["x2"], box["y2"]], outline="red", width=2)
            draw_label(draw, fragment)

        overlay.save(debug_path)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def strip_text_if_needed(fragments, include_text):
    if include_text:
        return fragments

    safe_fragments = []
    for fragment in fragments:
        safe_fragment = {
            key: value
            for key, value in fragment.items()
            if key != "text"
        }
        safe_fragment["text_length"] = len(fragment["text"])
        safe_fragments.append(safe_fragment)

    return safe_fragments


def main():
    parser = argparse.ArgumentParser(description="Run local OCR on one image and save debug artifacts.")
    parser.add_argument("--image", required=True, help="Path to one image file.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include recognized OCR text in the JSON log. Use only with safe sample images.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    config_path = Path(args.config)
    config = load_config(config_path)
    debug_path, log_path = make_run_paths(config, image_path)

    manifest = {
        "image": str(image_path),
        "config": str(config_path),
        "ocr_engine": "rapidocr",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "debug_overlay": str(debug_path),
        "include_text": args.include_text,
        "status": "started",
    }

    try:
        with Image.open(image_path) as image:
            width, height = image.size

        fragments = run_ocr(image_path)
        save_debug_overlay(image_path, debug_path, fragments)

        manifest.update(
            {
                "status": "completed",
                "image_size": {"width": width, "height": height},
                "fragment_count": len(fragments),
                "fragments": strip_text_if_needed(fragments, args.include_text),
            }
        )
        print(f"Saved OCR debug overlay: {debug_path}")
        print(f"Saved OCR log: {log_path}")
    except Exception as error:
        manifest.update(
            {
                "status": "error",
                "error": str(error),
                "error_type": type(error).__name__,
            }
        )
        print(f"Error. Saved OCR log: {log_path}")
    finally:
        save_json(log_path, manifest)


if __name__ == "__main__":
    main()
