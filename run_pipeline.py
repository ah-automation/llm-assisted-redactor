import argparse
from datetime import datetime
from pathlib import Path

import associate_fields
import document_router
import face_detect
import ocr_image
import redact_from_matches


def run_pipeline(image_path, config_path, document_definition_path, include_text, document_definitions_dir):
    image_path = Path(image_path)
    config_path = Path(config_path)
    document_definition_path = Path(document_definition_path) if document_definition_path else None
    document_definitions_dir = Path(document_definitions_dir)

    config = ocr_image.load_config(config_path)

    ocr_debug_path, ocr_log_path = ocr_image.make_run_paths(config, image_path)
    ocr_fragments = ocr_image.run_ocr(image_path)
    ocr_image.save_debug_overlay(image_path, ocr_debug_path, ocr_fragments)

    with ocr_image.Image.open(image_path) as image:
        width, height = image.size

    ocr_manifest = {
        "image": str(image_path),
        "config": str(config_path),
        "ocr_engine": "rapidocr",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "debug_overlay": str(ocr_debug_path),
        "include_text": include_text,
        "status": "completed",
        "image_size": {"width": width, "height": height},
        "fragment_count": len(ocr_fragments),
        "fragments": ocr_image.strip_text_if_needed(ocr_fragments, include_text),
    }
    ocr_image.save_json(ocr_log_path, ocr_manifest)

    if not include_text:
        raise ValueError("Field association currently requires OCR text. Run with --include-text.")

    routing_result = {
        "enabled": False,
        "status": "skipped",
        "reason": "A document definition was provided explicitly.",
    }
    if document_definition_path is None:
        document_definition_path, routing_result = document_router.route_document(
            ocr_manifest,
            document_definitions_dir,
        )
        if document_definition_path is None:
            raise ValueError(
                f"Document routing failed: {routing_result['status']} - {routing_result.get('reason', '')}"
            )
        routing_result["enabled"] = True

    document_definition = associate_fields.load_document_definition(document_definition_path)
    match_overlay_path, match_log_path = associate_fields.make_run_paths(config, image_path)
    matches, field_results = associate_fields.associate_fields(
        config,
        document_definition,
        ocr_manifest,
    )

    fields_by_id = {field.get("id"): field for field in associate_fields.iter_fields(document_definition)}
    fragments_by_id = {fragment["id"]: fragment for fragment in ocr_manifest["fragments"]}
    valid_matches, rejected_matches = associate_fields.validate_matches(
        matches,
        fields_by_id,
        fragments_by_id,
        document_definition,
    )
    repeat_matches, repeat_result = associate_fields.find_repeat_matches(
        config,
        document_definition,
        ocr_manifest,
        valid_matches,
        fields_by_id,
        fragments_by_id,
    )
    valid_repeat_matches, rejected_repeat_matches = associate_fields.validate_matches(
        repeat_matches,
        fields_by_id,
        fragments_by_id,
        document_definition,
    )
    redaction_boxes = associate_fields.build_redaction_boxes(valid_matches, fragments_by_id, fields_by_id)
    repeat_redaction_boxes = associate_fields.build_redaction_boxes(
        valid_repeat_matches,
        fragments_by_id,
        fields_by_id,
    )
    redaction_boxes.extend(repeat_redaction_boxes)
    face_boxes = []
    face_error = None
    if face_detect.is_enabled(config):
        try:
            face_boxes = face_detect.detect_faces(image_path, config)
            redaction_boxes.extend(face_boxes)
        except Exception as error:
            face_error = {
                "error": str(error),
                "error_type": type(error).__name__,
            }

    associate_fields.save_match_overlay(image_path, match_overlay_path, redaction_boxes)

    match_manifest = {
        "image": str(image_path),
        "ocr_log": str(ocr_log_path),
        "ocr_source_image": ocr_manifest.get("image"),
        "config": str(config_path),
        "document_definition": str(document_definition_path),
        "document_type": document_definition.get("id"),
        "routing": routing_result,
        "model": config.get("vlm", {}).get("model"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "match_overlay": str(match_overlay_path),
        "status": "completed",
        "field_results": field_results,
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
        "face_detection": {
            "enabled": face_detect.is_enabled(config),
            "status": "error" if face_error else "completed",
            "detections": face_boxes,
            "detection_count": len(face_boxes),
            "error": face_error,
        },
        "valid_match_count": len(valid_matches),
        "rejected_match_count": len(rejected_matches),
        "redaction_box_count": len(redaction_boxes),
    }
    associate_fields.save_json(match_log_path, match_manifest)

    redacted_path, redaction_log_path = redact_from_matches.make_run_paths(config, image_path)
    valid_boxes, rejected_boxes = redact_from_matches.validate_redaction_boxes(
        redaction_boxes,
        width,
        height,
    )
    redact_from_matches.redact_image(image_path, redacted_path, valid_boxes)

    redaction_manifest = {
        "image": str(image_path),
        "matches_log": str(match_log_path),
        "config": str(config_path),
        "output": str(redacted_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "image_size": {"width": width, "height": height},
        "valid_boxes": valid_boxes,
        "rejected_boxes": rejected_boxes,
        "valid_box_count": len(valid_boxes),
        "rejected_box_count": len(rejected_boxes),
    }
    redact_from_matches.save_json(redaction_log_path, redaction_manifest)

    return {
        "ocr_log": ocr_log_path,
        "ocr_debug_overlay": ocr_debug_path,
        "match_log": match_log_path,
        "match_overlay": match_overlay_path,
        "redaction_log": redaction_log_path,
        "redacted_image": redacted_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Run OCR, field association, and redaction in sequence.")
    parser.add_argument("--image", required=True, help="Path to one image file.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument(
        "--document-definition",
        help="Path to a document definition YAML file. If omitted, OCR-based routing is used.",
    )
    parser.add_argument(
        "--document-definitions-dir",
        default="document_definitions",
        help="Folder containing routable document definition YAML files.",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include OCR text in the OCR log. Required for association during this POC.",
    )
    args = parser.parse_args()

    outputs = run_pipeline(
        args.image,
        args.config,
        args.document_definition,
        args.include_text,
        args.document_definitions_dir,
    )

    print(f"Saved OCR log: {outputs['ocr_log']}")
    print(f"Saved OCR debug overlay: {outputs['ocr_debug_overlay']}")
    print(f"Saved field match log: {outputs['match_log']}")
    print(f"Saved field match overlay: {outputs['match_overlay']}")
    print(f"Saved redaction log: {outputs['redaction_log']}")
    print(f"Saved redacted image: {outputs['redacted_image']}")


if __name__ == "__main__":
    main()
