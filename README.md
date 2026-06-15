# Local Document PII Redactor

A local-first proof of concept for redacting personally identifiable information from scanned or photographed document images.

This project combines local OCR, local LLM-assisted document understanding, configurable document definitions, and black-box redaction. It was built as a learning project and portfolio piece for regulated-environment-friendly document processing, where images and model calls stay on the local machine.

This is not production compliance software. A production-grade solution would require stronger models, validation datasets, versioned prompts/model settings, and formal human review/compliance controls.

## Why This Exists

Traditional redaction often leans on OCR, regex, configuration, and custom parsing code. Those pipelines can become brittle: one small change can cascade through the rules, and edge cases often require fuzzy judgment rather than exact matching.

This project investigates whether local LLM-assisted inference can simplify that association layer while keeping documents private. It also explores what smaller local models can contribute to complex document work: they cannot compete with large frontier models on raw capability, but they offer major cost savings and can run inside private or regulated environments.

## What It Does

The tool accepts one image, finds configured PII fields, and writes a redacted PNG plus a JSON manifest. It can either route the document type with the local LLM or use an explicit document definition supplied by the caller.

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
- Credit cards
- Invoices, statements, and bills
- Cheques
- Health insurance cards

The goal is not one universal PII detector; it is document-aware redaction for known document types.

## Recommended Demo Path

For repeatable testing and portfolio demos, use an explicit document definition:

```powershell
python redact.py --image input\sample.png --config config.yaml --document-definition document_definitions\passports\common.yaml
```

Automatic routing is included to demonstrate local LLM document classification, but it is less stable than passing a known definition.

## Local-First Architecture

The pipeline runs locally:

1. RapidOCR extracts text fragments and coordinates.
2. A local OpenAI-compatible LLM endpoint, initially LM Studio, optionally routes the document type when no definition is provided.
3. The local LLM associates noisy OCR fragments with configured fields in YAML document definitions.
4. Optional family-specific fallback detection looks for leftover opaque identifiers after known fields are handled.
5. Local validation turns accepted OCR fragments into redaction boxes.
6. Optional local face detection redacts detected faces/photos.
7. Pillow writes a redacted PNG using solid black rectangles.
8. A JSON manifest records metadata, model diagnostics, selected definition, boxes, and output paths.

## Installation

Create and activate a Python environment, then install dependencies:

```powershell
pip install -r requirements.txt
```

You also need:

- LM Studio running a local model through its OpenAI-compatible server
- `config.yaml` updated with the model name served by LM Studio
- Optional face detection model at `models/face_detection_yunet_2023mar.onnx` when face detection is enabled

The project requires a local chat-completions endpoint. LM Studio with `google/gemma-4-26b-a4b` is the tested setup; it uses an OpenAI-compatible API format but runs locally on your machine. Other local runtimes and instruction models may work if they support compatible chat completions and reliably return strict JSON, but they should be treated as unvalidated until tested against your document set.

Recommended LM Studio setup:

- Start the local OpenAI-compatible server.
- Load `google/gemma-4-26b-a4b`.
- Set the context length to at least `4096`; larger values may help dense documents but should be tested for speed and stability.
- Disable thinking/reasoning mode for this workflow.
- Use temperature `0`.
- Make sure `llm.model` in `config.yaml` exactly matches the model name served by LM Studio.

The bundled YuNet face detection model is MIT licensed and sourced from OpenCV Zoo. See [models/README.md](models/README.md) for attribution.

## Configuration

The default runtime settings live in `config.yaml`.

Important values:

- `llm.base_url`: local OpenAI-compatible endpoint, usually `http://localhost:1234/v1`
- `llm.model`: the exact model name from LM Studio
- `llm.temperature`: recommended `0`
- `output_dir`: where redacted images are written
- `logs_dir`: where manifests are written
- `debug.enabled`: when `true`, assume OCR text, raw model responses, and overlays contain PII

Keep `debug.enabled: false` for normal runs.

The config file does not need to live in the project folder. Pass any local or network-accessible path with `--config`, as long as the Python process has permission to read it.

## Running

With automatic document routing:

```powershell
python redact.py --image input\sample.png --config config.yaml
```

With an explicit document definition, recommended for stable testing:

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

Do not commit real documents, output images, debug overlays, or manifests from real documents.

Normal manifests omit OCR text and raw LLM responses. Debug mode writes troubleshooting artifacts that should be treated as containing PII when real documents are used.

## Limitations

- Image-only input; no PDF support.
- OCR quality strongly affects results.
- The current version is intended for Latin-script documents.
- Handwriting is not a primary target.
- Very dense documents can exceed local model context limits.
- Barcodes and QR codes are not detected or redacted.
- Redaction intentionally uses full OCR boxes to avoid partial-character leakage; this may over-redact nearby labels or adjacent text.
- Automatic routing is experimental and can be inconsistent; explicit document definitions are more stable.
- Local LLM behavior depends on the model, context size, and LM Studio settings.
- Model changes may require prompt and document-definition retesting.
- Redaction is best-effort and should be reviewed before relying on it.

## More Docs

- [Architecture](docs/architecture.md)
- [Document Definitions](docs/document_definitions.md)
