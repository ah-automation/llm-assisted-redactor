# Local Document PII Redactor

A local-first proof of concept for redacting personally identifiable information from scanned or photographed document images.

This project combines local OCR, local LLM-assisted document understanding, configurable document definitions, and simple black-box redaction. It was built as a learning project and portfolio piece for regulated-environment-friendly document processing, where images and model calls stay on the local machine.

This is not production compliance software.

## What It Does

The tool accepts one image, detects the document type, finds configured PII fields, and writes a redacted PNG plus a JSON manifest.

```powershell
python redact.py --image input\sample.png --config config.yaml
```

By default, stdout is a compact JSON summary suitable for automation tools:

```json
{"status":"completed","manifest":"logs\\...","output":"output\\...","document_type":"passport.common","review_status":"passed"}
```

For manual troubleshooting, use:

```powershell
python redact.py --image input\sample.png --config config.yaml --verbose
```

## Supported Document Families

Current document definitions include:

- Passports
- Driver's licenses
- Invoices, statements, and bills
- Cheques
- Health insurance cards

The definitions are intentionally configurable. The goal is not one universal PII detector; it is document-aware redaction for known document types.

## Local-First Architecture

The pipeline runs locally:

1. RapidOCR extracts text fragments and coordinates.
2. A local OpenAI-compatible LLM endpoint, initially LM Studio, routes the document type when no definition is provided.
3. The local LLM associates noisy OCR fragments with configured fields in YAML document definitions.
4. Local validation turns accepted OCR fragments into redaction boxes.
5. Optional local face detection redacts detected faces/photos.
6. Pillow writes a redacted PNG using solid black rectangles.
7. A JSON manifest records metadata, model diagnostics, selected definition, boxes, and output paths.

## Installation

Create and activate a Python environment, then install dependencies:

```powershell
pip install -r requirements.txt
```

You also need:

- LM Studio running a local model through its OpenAI-compatible server
- `config.yaml` updated with the model name served by LM Studio
- Optional face detection model at `models/face_detection_yunet_2023mar.onnx` when face detection is enabled

The project requires a local chat-completions endpoint. LM Studio is the tested setup; it uses an OpenAI-compatible API format but runs locally on your machine. Other local runtimes may work if they support compatible chat completions and structured JSON responses.

Tested primarily with `google/gemma-4-26b-a4b` in LM Studio. Other local instruction models may work if they support the OpenAI-compatible chat completions API and reliably return strict JSON, but they have not been validated for this project.

## Configuration

The default runtime settings live in `config.yaml`.

Important values:

- `llm.base_url`: local OpenAI-compatible endpoint, usually `http://localhost:1234/v1`
- `llm.model`: the exact model name from LM Studio
- `llm.temperature`: recommended `0`
- `output_dir`: where redacted images are written
- `logs_dir`: where manifests are written
- `debug.enabled`: when `true`, OCR text, raw model responses, and overlays may be saved locally

Keep `debug.enabled: false` for normal runs.

The config file does not need to live in the project folder. Pass any local or network-accessible path with `--config`, as long as the Python process has permission to read it.

## Running

With automatic document routing:

```powershell
python redact.py --image input\sample.png --config config.yaml
```

With an explicit document definition:

```powershell
python redact.py --image input\sample.png --config config.yaml --document-definition document_definitions\passports\common.yaml
```

Human-readable output:

```powershell
python redact.py --image input\sample.png --config config.yaml --verbose
```

## Outputs

The main script writes:

- redacted image: `output\YYYYMMDD-HHMMSS-uniqueid-image_name-redacted.png`
- manifest: `logs\YYYYMMDD-HHMMSS-uniqueid-image_name-manifest.json`

`output_dir` and `logs_dir` can point to local folders or network locations, as long as the Python process has permission to create files there.

Exit codes:

- `0`: completed
- `1`: error
- `2`: unsupported document, ambiguous document, low-confidence route, or review-needed status

Unsupported input formats such as PDF and HEIC/HEIF return exit code `1`.

## Input Formats

Tested formats:

- `.jpg`
- `.jpeg`
- `.png`
- `.webp`
- `.avif`

Likely supported when available in the local Pillow installation:

- `.bmp`
- `.tif`
- `.tiff`

Explicitly unsupported/out of scope:

- `.pdf`
- `.heic`
- `.heif`

Output is always PNG.

## Privacy Notes

Do not commit real documents, output images, debug overlays, or manifests that may contain PII.

Normal manifests avoid storing OCR text and raw LLM responses. Debug mode is useful during development, but it may save sensitive text and image overlays locally.

## Limitations

- Image-only input; no PDF support.
- OCR quality strongly affects results.
- The current version is intended for Latin-script documents.
- Handwriting is not a primary target.
- Very dense documents can exceed local model context limits.
- Barcodes and QR codes are not detected or redacted.
- Local LLM behavior depends on the model, context size, and LM Studio settings.
- Redaction is best-effort and should be reviewed before relying on it.
- This project is a POC, not a compliance-certified redaction product.

## More Docs

- [Architecture](docs/architecture.md)
- [Document Definitions](docs/document_definitions.md)
- [Safe Examples](examples/README.md)
