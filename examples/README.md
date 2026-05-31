# Examples

This folder is reserved for safe examples only.

Do not add real identity documents, real invoices, real cheques, real insurance cards, output images, debug overlays, or manifests containing actual PII.

## Safe Example Commands

Automatic routing:

```powershell
python redact.py --image input\synthetic_passport.png --config config.yaml
```

Explicit definition:

```powershell
python redact.py --image input\synthetic_passport.png --config config.yaml --document-definition document_definitions\passports\common.yaml
```

Verbose troubleshooting:

```powershell
python redact.py --image input\synthetic_passport.png --config config.yaml --verbose
```

## Automation Summary Example

Default stdout is a compact JSON object:

```json
{
  "status": "completed",
  "manifest": "logs\\20260531-120000-ab12cd34-synthetic_passport-manifest.json",
  "output": "output\\20260531-120000-ab12cd34-synthetic_passport-redacted.png",
  "document_type": "passport.common",
  "document_definition": "document_definitions\\passports\\common.yaml",
  "routing_status": "completed",
  "redaction_box_count": 12,
  "rejected_box_count": 0,
  "error": null,
  "error_type": null
}
```

## Safe Manifest Snippet

Normal manifests should not include OCR text or raw model responses when `debug.enabled` is `false`.

```json
{
  "status": "completed",
  "document_type": "passport.common",
  "ocr": {
    "include_text": false,
    "fragment_count": 42,
    "fragments": [
      {
        "id": "ocr_0001",
        "box": {"x1": 10, "y1": 20, "x2": 80, "y2": 40},
        "confidence": 0.99,
        "text_length": 8
      }
    ]
  },
  "redaction": {
    "valid_box_count": 12,
    "rejected_box_count": 0
  }
}
```

## Future Safe Samples

Good sample candidates:

- synthetic passport-like image
- synthetic driver's license-like image
- synthetic invoice
- synthetic cheque
- synthetic health insurance card

Samples should be clearly fake and should avoid resembling a real person's actual document.
