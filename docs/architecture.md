# Architecture

This project is a local OCR plus local LLM redaction pipeline for known document types.

The core idea is to let OCR handle coordinates and let the LLM handle semantic association. OCR is good at saying "this text fragment is at this box." The LLM is useful for deciding whether a noisy OCR fragment belongs to a configured field such as passport number, date of birth, account number, or customer name.

## Main Flow

The main entry point is:

```powershell
python redact.py --image input\sample.png --config config.yaml
```

High-level flow:

1. Load `config.yaml`.
2. Validate that the input is not an explicitly unsupported format.
3. Run RapidOCR against the image.
4. If no document definition is provided, route the document using local LLM routing.
5. Load the selected YAML document definition.
6. Ask the local LLM to associate OCR fragments with configured fields.
7. Optionally run a second pass for repeated values.
8. Optionally run local face detection.
9. Validate and merge redaction boxes.
10. Draw solid black rectangles over accepted boxes.
11. Save a redacted PNG and one JSON manifest.
12. Print a compact JSON summary to stdout.

## Components

`redact.py`

The user-facing command. It coordinates the full pipeline, writes the manifest, and prints the automation-friendly JSON summary.

`redactor/ocr_image.py`

Runs RapidOCR and normalizes OCR output into fragments:

- fragment id
- bounding box
- polygon points
- confidence
- text, kept in memory for processing

With debug disabled, saved manifests strip OCR text and keep only metadata such as text length.

`redactor/document_router.py`

Uses a local LLM call to select the best document definition. Routing candidates come from YAML files that opt into routing.

`redactor/associate_fields.py`

Builds prompts for each configured field and asks the local LLM to choose OCR fragment ids for that field. It validates returned field ids and fragment ids before redaction.

`redactor/face_detect.py`

Uses OpenCV YuNet for local face/photo detection when enabled in `config.yaml`.

`redactor/redact_from_matches.py`

Validates redaction boxes and writes the final redacted image.

## Local LLM Role

The LLM does not draw boxes directly. Instead, it receives OCR fragment ids, text, and coordinates, then returns fragment ids that should be redacted.

This keeps localization deterministic:

- OCR supplies boxes.
- The LLM supplies document-aware association.
- Python validates and draws the final redactions.

## Logging And Output

Each normal run writes one manifest:

```text
logs/YYYYMMDD-HHMMSS-uniqueid-image_name-manifest.json
```

Each normal run writes one redacted image:

```text
output/YYYYMMDD-HHMMSS-uniqueid-image_name-redacted.png
```

`config.yaml` may be passed from any local or network-accessible path with `--config`. `logs_dir` and `output_dir` may also point to local or network locations, provided the Python process has the necessary read/write permissions.

Default stdout is compact JSON:

```json
{"status":"completed","manifest":"logs\\...","output":"output\\...","document_type":"passport.common","review_status":"passed"}
```

Use `--verbose` for human-readable output, library logs, and tracebacks during setup or troubleshooting.

## Debug Mode

`config.yaml` controls debug behavior:

```yaml
debug:
  enabled: false
```

When debug is off, saved manifests avoid OCR text, raw model responses, LLM notes, and visual overlays.

When debug is on, the pipeline may save OCR text, raw model responses, and debug images. This is useful for development but may expose sensitive document contents.

## Error Handling

The script returns:

- exit code `0` for completed runs
- exit code `1` for errors
- exit code `2` for unsupported document, ambiguous document, low-confidence route, or review-needed statuses

If a document definition's review policy is not satisfied, the manifest status is `needs_review`. In normal mode, the redacted image is not written; in debug mode, a partial redacted image may be written for troubleshooting.

Unsupported input formats such as PDF and HEIC/HEIF fail before OCR starts and return exit code `1`.

## Current Scope

The pipeline is designed for short Latin-script scanned/image documents where OCR can extract the relevant text. It is not designed for long PDFs, large multi-page records, or compliance-grade unattended redaction.
