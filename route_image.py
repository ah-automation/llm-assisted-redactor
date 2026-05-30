import argparse
from datetime import datetime
from pathlib import Path

from PIL import Image

import document_router
import ocr_image


def make_route_log_path(config, image_path):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logs_dir = Path(config.get("logs_dir", "logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f"{image_path.stem}-routing-{timestamp}.json"


def main():
    parser = argparse.ArgumentParser(description="Experiment with OCR and LLM-assisted document routing.")
    parser.add_argument("--image", required=True, help="Path to one image file.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument(
        "--document-definitions-dir",
        default="document_definitions",
        help="Folder containing routable document definition YAML files.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    config_path = Path(args.config)
    definitions_dir = Path(args.document_definitions_dir)
    config = ocr_image.load_config(config_path)
    log_path = make_route_log_path(config, image_path)

    manifest = {
        "image": str(image_path),
        "config": str(config_path),
        "document_definitions_dir": str(definitions_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "started",
    }

    try:
        with Image.open(image_path) as image:
            width, height = image.size

        fragments = ocr_image.run_ocr(image_path)
        ocr_manifest = {
            "image": str(image_path),
            "image_size": {"width": width, "height": height},
            "fragment_count": len(fragments),
            "fragments": fragments,
        }

        llm_path, llm_result = document_router.route_document_with_llm(
            config,
            ocr_manifest,
            definitions_dir,
        )

        manifest.update(
            {
                "status": "completed",
                "image_size": {"width": width, "height": height},
                "fragment_count": len(fragments),
                "llm_route": llm_result,
            }
        )

        print("LLM route:")
        print(f"  status: {llm_result.get('status')}")
        print(f"  definition: {llm_path}")
        print(f"  type: {llm_result.get('selected_document_type')}")
        print(f"  confidence: {llm_result.get('confidence')}")
        print(f"  threshold: {llm_result.get('confidence_threshold')}")
        print(f"Saved routing log: {log_path}")
    except Exception as error:
        manifest.update(
            {
                "status": "error",
                "error": str(error),
                "error_type": type(error).__name__,
            }
        )
        print(f"Error. Saved routing log: {log_path}")
    finally:
        ocr_image.save_json(log_path, manifest)


if __name__ == "__main__":
    main()
