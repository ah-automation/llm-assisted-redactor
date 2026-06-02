import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import io
import json
import logging
from pathlib import Path
import sys
import traceback
from uuid import uuid4

from redactor import associate_fields
from redactor import document_definition_validator
from redactor import document_router
from redactor import face_detect
from redactor import ocr_image
from redactor import redact_from_matches


REVIEW_STATUSES = {"unsupported_document", "ambiguous_document", "low_confidence", "needs_review"}
OUT_OF_SCOPE_EXTENSIONS = {
    ".pdf": "PDF input is out of scope for this image-only POC.",
    ".heic": "HEIC/HEIF input is not supported by the current local image stack.",
    ".heif": "HEIC/HEIF input is not supported by the current local image stack.",
}


class RedactionRunError(Exception):
    def __init__(self, manifest, original_error):
        super().__init__(str(original_error))
        self.manifest = manifest
        self.original_error = original_error


class UnsupportedInputFormatError(ValueError):
    pass


def validate_input_format(image_path):
    suffix = image_path.suffix.lower()
    if suffix in OUT_OF_SCOPE_EXTENSIONS:
        raise UnsupportedInputFormatError(
            f"Unsupported input format '{suffix}'. {OUT_OF_SCOPE_EXTENSIONS[suffix]} "
            "Use a supported image format such as .jpg, .jpeg, .png, .webp, .avif, .bmp, .tif, or .tiff."
        )


def get_llm_config(config):
    return config["llm"]


def make_run_paths(config, image_path):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_suffix = uuid4().hex[:8]
    run_id = f"{timestamp}-{unique_suffix}-{image_path.stem}"

    output_dir = Path(config.get("output_dir", "output"))
    logs_dir = Path(config.get("logs_dir", "logs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_id": run_id,
        "manifest": logs_dir / f"{run_id}-manifest.json",
        "redacted_image": output_dir / f"{run_id}-redacted.png",
        "ocr_debug_overlay": output_dir / f"{run_id}-ocr-debug.png",
        "match_overlay": output_dir / f"{run_id}-field-matches.png",
    }


def get_redacted_field_ids(valid_boxes):
    return {
        box.get("field_id")
        for box in valid_boxes
        if isinstance(box, dict) and box.get("field_id")
    }


def group_is_satisfied(group, redacted_field_ids):
    any_of = group.get("any_of") or []
    all_of = group.get("all_of") or []

    any_of_satisfied = bool(any_of) and any(field_id in redacted_field_ids for field_id in any_of)
    all_of_satisfied = bool(all_of) and all(field_id in redacted_field_ids for field_id in all_of)

    if any_of and all_of:
        return any_of_satisfied or all_of_satisfied
    if any_of:
        return any_of_satisfied
    if all_of:
        return all_of_satisfied
    return True


def get_review_result(document_definition, valid_boxes):
    review_config = document_definition.get("review") or {}
    required_fields = review_config.get("required_fields") or []
    required_groups = review_config.get("required_groups") or []
    redacted_field_ids = {
        field_id
        for field_id in get_redacted_field_ids(valid_boxes)
        if field_id
    }

    missing_fields = sorted(
        field_id for field_id in required_fields if field_id not in redacted_field_ids
    )
    missing_groups = [
        {
            "id": group.get("id"),
            "label": group.get("label"),
            "any_of": group.get("any_of") or [],
            "all_of": group.get("all_of") or [],
        }
        for group in required_groups
        if not group_is_satisfied(group, redacted_field_ids)
    ]

    needs_review = bool(missing_fields or missing_groups)
    return {
        "status": "needs_review" if needs_review else "passed",
        "missing_fields": missing_fields,
        "missing_groups": missing_groups,
        "redacted_fields": sorted(redacted_field_ids),
    }


def redact_image_file(image_path, config_path, document_definition_path, document_definitions_dir):
    image_path = Path(image_path)
    config_path = Path(config_path)
    document_definition_path = Path(document_definition_path) if document_definition_path else None
    document_definitions_dir = Path(document_definitions_dir)

    config = ocr_image.load_config(config_path)
    debug_enabled = ocr_image.is_debug_enabled(config)
    run_paths = make_run_paths(config, image_path)
    manifest_path = run_paths["manifest"]
    redacted_path = run_paths["redacted_image"]
    ocr_debug_path = run_paths["ocr_debug_overlay"]
    match_overlay_path = run_paths["match_overlay"]

    manifest = {
        "run_id": run_paths["run_id"],
        "image": str(image_path),
        "config": str(config_path),
        "output": str(redacted_path),
        "model": get_llm_config(config).get("model"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "debug": ocr_image.debug_manifest(config),
        "status": "started",
        "artifacts": {
            "manifest": str(manifest_path),
            "redacted_image": str(redacted_path),
            "ocr_debug_overlay": str(ocr_debug_path) if debug_enabled else None,
            "match_overlay": str(match_overlay_path) if debug_enabled else None,
        },
    }

    try:
        validate_input_format(image_path)
        ocr_fragments = ocr_image.run_ocr(image_path)
        if debug_enabled:
            ocr_image.save_debug_overlay(image_path, ocr_debug_path, ocr_fragments)

        with ocr_image.Image.open(image_path) as image:
            width, height = image.size

        ocr_manifest_for_processing = {
            "image": str(image_path),
            "config": str(config_path),
            "ocr_engine": "rapidocr",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "debug": ocr_image.debug_manifest(config),
            "debug_overlay": str(ocr_debug_path) if debug_enabled else None,
            "include_text": True,
            "status": "completed",
            "image_size": {"width": width, "height": height},
            "fragment_count": len(ocr_fragments),
            "fragments": ocr_fragments,
        }
        manifest["ocr"] = {
            **ocr_manifest_for_processing,
            "include_text": debug_enabled,
            "fragments": ocr_image.strip_text_if_needed(ocr_fragments, debug_enabled),
        }

        routing_result = {
            "enabled": False,
            "status": "skipped",
            "reason": "A document definition was provided explicitly.",
        }
        if document_definition_path is None:
            document_definition_path, routing_result = document_router.route_document_with_llm(
                config,
                ocr_manifest_for_processing,
                document_definitions_dir,
            )
            manifest["routing"] = routing_result
            if document_definition_path is None:
                manifest["status"] = routing_result.get("status", "routing_failed")
                raise ValueError(
                    f"Document routing failed: {routing_result['status']} - {routing_result.get('reason', '')}"
                )
            routing_result["enabled"] = True

        document_definition_validator.require_valid_document_definition(document_definition_path)
        document_definition = associate_fields.load_document_definition(document_definition_path)
        manifest["routing"] = routing_result
        manifest["document_definition"] = str(document_definition_path)
        manifest["document_type"] = document_definition.get("id")

        matches, field_results = associate_fields.associate_fields(
            config,
            document_definition,
            ocr_manifest_for_processing,
        )

        fields_by_id = {field.get("id"): field for field in associate_fields.iter_fields(document_definition)}
        fragments_by_id = {fragment["id"]: fragment for fragment in ocr_manifest_for_processing["fragments"]}
        valid_matches, rejected_matches = associate_fields.validate_matches(
            matches,
            fields_by_id,
            fragments_by_id,
            include_notes=debug_enabled,
        )
        repeat_matches, repeat_result = associate_fields.find_repeat_matches(
            config,
            document_definition,
            ocr_manifest_for_processing,
            valid_matches,
            fields_by_id,
            fragments_by_id,
        )
        valid_repeat_matches, rejected_repeat_matches = associate_fields.validate_matches(
            repeat_matches,
            fields_by_id,
            fragments_by_id,
            include_notes=debug_enabled,
        )
        text_redaction_boxes = associate_fields.build_redaction_boxes(valid_matches, fragments_by_id, fields_by_id)
        repeat_redaction_boxes = associate_fields.build_redaction_boxes(
            valid_repeat_matches,
            fragments_by_id,
            fields_by_id,
        )
        text_redaction_boxes.extend(repeat_redaction_boxes)
        redaction_boxes = list(text_redaction_boxes)

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

        if debug_enabled:
            associate_fields.save_match_overlay(image_path, match_overlay_path, redaction_boxes)

        valid_boxes, rejected_boxes = redact_from_matches.validate_redaction_boxes(
            redaction_boxes,
            width,
            height,
        )
        review_result = get_review_result(document_definition, valid_boxes)
        needs_review = review_result["status"] == "needs_review"
        should_write_output = not needs_review or debug_enabled
        if should_write_output:
            redact_from_matches.redact_image(image_path, redacted_path, valid_boxes)

        output_value = str(redacted_path) if should_write_output else None
        manifest["output"] = output_value
        manifest["artifacts"]["redacted_image"] = output_value

        manifest.update(
            {
                "status": "needs_review" if needs_review else "completed",
                "association": {
                    "field_results": field_results,
                    "valid_matches": valid_matches,
                    "rejected_matches": rejected_matches,
                    "redaction_boxes": text_redaction_boxes,
                    "valid_match_count": len(valid_matches),
                    "rejected_match_count": len(rejected_matches),
                    "redaction_box_count": len(text_redaction_boxes),
                },
                "repeat_detection": {
                    **repeat_result,
                    "valid_matches": valid_repeat_matches,
                    "rejected_matches": rejected_repeat_matches,
                    "redaction_boxes": repeat_redaction_boxes,
                    "valid_match_count": len(valid_repeat_matches),
                    "rejected_match_count": len(rejected_repeat_matches),
                    "redaction_box_count": len(repeat_redaction_boxes),
                },
                "face_detection": {
                    "enabled": face_detect.is_enabled(config),
                    "status": "error" if face_error else "completed",
                    "detections": face_boxes,
                    "detection_count": len(face_boxes),
                    "error": face_error,
                },
                "review": {
                    **review_result,
                    "partial_output_written": bool(needs_review and should_write_output),
                },
                "redaction": {
                    "status": "completed" if should_write_output else "skipped_needs_review",
                    "image_size": {"width": width, "height": height},
                    "valid_boxes": valid_boxes,
                    "rejected_boxes": rejected_boxes,
                    "valid_box_count": len(valid_boxes),
                    "rejected_box_count": len(rejected_boxes),
                },
            }
        )
    except Exception as error:
        status = manifest.get("status")
        if status in {"started", "completed"}:
            if isinstance(error, UnsupportedInputFormatError):
                manifest["status"] = "unsupported_format"
            elif isinstance(error, document_definition_validator.DocumentDefinitionValidationError):
                manifest["status"] = "definition_error"
            else:
                manifest["status"] = "error"
        output_path = manifest.get("output")
        if output_path and not Path(output_path).exists():
            manifest["output"] = None
            manifest["artifacts"]["redacted_image"] = None
        manifest.update(
            {
                "error": str(error),
                "error_type": type(error).__name__,
            }
        )
        if isinstance(error, document_definition_validator.DocumentDefinitionValidationError):
            manifest["definition_errors"] = error.errors
        redact_from_matches.save_json(manifest_path, manifest)
        raise RedactionRunError(manifest, error) from error

    redact_from_matches.save_json(manifest_path, manifest)

    return {
        "ocr_debug_overlay": ocr_debug_path if debug_enabled else None,
        "match_overlay": match_overlay_path if debug_enabled else None,
        "manifest": manifest_path,
        "redacted_image": redacted_path,
    }


def build_summary(manifest):
    redaction = manifest.get("redaction") or {}
    routing = manifest.get("routing") or {}
    review = manifest.get("review") or {}
    return {
        "status": manifest.get("status", "error"),
        "manifest": (manifest.get("artifacts") or {}).get("manifest"),
        "output": manifest.get("output"),
        "document_type": manifest.get("document_type"),
        "document_definition": manifest.get("document_definition"),
        "routing_status": routing.get("status"),
        "review_status": review.get("status"),
        "missing_review_fields": review.get("missing_fields", []),
        "missing_review_groups": [
            group.get("id")
            for group in review.get("missing_groups", [])
            if isinstance(group, dict) and group.get("id")
        ],
        "partial_output_written": review.get("partial_output_written", False),
        "redaction_box_count": redaction.get("valid_box_count", 0),
        "rejected_box_count": redaction.get("rejected_box_count", 0),
        "error": manifest.get("error"),
        "error_type": manifest.get("error_type"),
        "definition_errors": manifest.get("definition_errors", []),
    }


def exit_code_for_status(status):
    if status == "completed":
        return 0
    if status in REVIEW_STATUSES:
        return 2
    return 1


def print_verbose_summary(summary):
    print(f"Status: {summary['status']}")
    if summary.get("document_type"):
        print(f"Document type: {summary['document_type']}")
    if summary.get("routing_status"):
        print(f"Routing status: {summary['routing_status']}")
    if summary.get("review_status"):
        print(f"Review status: {summary['review_status']}")
    if summary.get("missing_review_fields"):
        print(f"Missing review fields: {', '.join(summary['missing_review_fields'])}")
    if summary.get("missing_review_groups"):
        print(f"Missing review groups: {', '.join(summary['missing_review_groups'])}")
    if summary.get("manifest"):
        print(f"Saved manifest: {summary['manifest']}")
    if summary.get("output"):
        if summary["status"] == "completed":
            print(f"Saved redacted image: {summary['output']}")
        else:
            print(f"Output path: {summary['output']}")
    if summary.get("redaction_box_count") is not None:
        print(f"Redaction boxes: {summary['redaction_box_count']}")
    if summary.get("error"):
        print(f"Error: {summary['error']}")
    if summary.get("definition_errors"):
        print("Definition errors:")
        for error in summary["definition_errors"]:
            print(f"- {error}")


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
        "--verbose",
        action="store_true",
        help="Print human-readable status messages instead of a compact JSON summary.",
    )
    args = parser.parse_args()

    manifest = None
    original_error = None
    captured_output = io.StringIO()
    if not args.verbose:
        logging.disable(logging.CRITICAL)

    try:
        if args.verbose:
            print("[INFO] Running OCR and local LLM checks. This may take a few minutes...")
            outputs = redact_image_file(
                args.image,
                args.config,
                args.document_definition,
                args.document_definitions_dir,
            )
        else:
            with redirect_stdout(captured_output), redirect_stderr(captured_output):
                outputs = redact_image_file(
                    args.image,
                    args.config,
                    args.document_definition,
                    args.document_definitions_dir,
                )
        manifest = redact_from_matches.load_json(outputs["manifest"])
    except RedactionRunError as error:
        manifest = error.manifest
        original_error = error.original_error
    except Exception as error:
        original_error = error
        manifest = {
            "status": "error",
            "error": str(error),
            "error_type": type(error).__name__,
        }
    finally:
        if not args.verbose:
            logging.disable(logging.NOTSET)

    summary = build_summary(manifest)
    if args.verbose:
        print_verbose_summary(summary)
        if original_error:
            print("Traceback:", file=sys.stderr)
            traceback.print_exception(
                type(original_error),
                original_error,
                original_error.__traceback__,
                file=sys.stderr,
            )
    else:
        print(json.dumps(summary, separators=(",", ":")))

    exit_code = exit_code_for_status(summary["status"])
    if original_error and args.verbose:
        print(f"Exit code: {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
