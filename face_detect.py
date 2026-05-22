from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def get_face_config(config):
    return config.get("face_detection", {})


def is_enabled(config):
    return bool(get_face_config(config).get("enabled", False))


def expand_box(box, image_width, image_height, expand_percent):
    width = box["x2"] - box["x1"]
    height = box["y2"] - box["y1"]
    pad_x = round(width * expand_percent)
    pad_y = round(height * expand_percent)

    return {
        "x1": max(0, box["x1"] - pad_x),
        "y1": max(0, box["y1"] - pad_y),
        "x2": min(image_width, box["x2"] + pad_x),
        "y2": min(image_height, box["y2"] + pad_y),
    }


def detect_faces(image_path, config):
    face_config = get_face_config(config)
    model_path = Path(face_config.get("model_path", "models/face_detection_yunet_2023mar.onnx"))
    if not model_path.exists():
        raise FileNotFoundError(f"Face detection model not found: {model_path}")

    with Image.open(image_path) as pil_image:
        rgb_image = pil_image.convert("RGB")
        image = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)

    image_height, image_width = image.shape[:2]
    detector = cv2.FaceDetectorYN_create(
        str(model_path),
        "",
        (image_width, image_height),
        float(face_config.get("score_threshold", 0.6)),
        float(face_config.get("nms_threshold", 0.3)),
        int(face_config.get("top_k", 5000)),
    )

    _, faces = detector.detect(image)
    if faces is None:
        return []

    expand_percent = float(face_config.get("expand_box_percent", 0.0))
    detections = []
    for index, face in enumerate(faces):
        x, y, width, height = face[:4]
        score = float(face[-1])
        box = {
            "x1": int(round(x)),
            "y1": int(round(y)),
            "x2": int(round(x + width)),
            "y2": int(round(y + height)),
        }
        expanded = expand_box(box, image_width, image_height, expand_percent)
        detections.append(
            {
                "field_id": "face",
                "box": expanded,
                "source": "opencv_yunet",
                "source_index": index,
                "confidence": score,
            }
        )

    return detections
