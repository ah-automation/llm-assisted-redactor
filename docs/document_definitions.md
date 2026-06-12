# Document Definitions

Document definitions are YAML files that describe known document types and the fields that should be redacted.

They live under `document_definitions/` and are grouped by document family:

```text
document_definitions/
  cheques/
    common.yaml
  health_insurance_cards/
    common.yaml
  invoices/
    common.yaml
  licenses/
    common.yaml
    ohio.yaml
  passports/
    common.yaml
    netherlands.yaml
```

## Common And Extended Definitions

Use `common.yaml` for fields shared across a document family.

Use an extended YAML file when a country, region, vendor, or screen has special labels or layout behavior.

Example:

```yaml
extends: "common.yaml"

id: "license.ohio"
label: "Ohio driver's license"

field_overrides:
  license_number:
    anchors_add:
      - "4dNO"
      - "4d NO"
```

The inheritance model is intentionally simple:

- one parent through `extends`
- child paths resolve relative to the current YAML file
- child values override parent values
- `field_overrides` are keyed by field id

## Routing

Definitions participate in automatic routing when they provide routing markers:

```yaml
routing:
  markers:
    strong:
      - "Passport"
      - "Passport No"
    weak:
      - "Nationality"
      - "Date of birth"
```

When `redact.py` is called without `--document-definition`, OCR snippets and routing markers are sent to the local LLM. Routing happens in two stages: first the LLM chooses a document family from each folder's `common.yaml`, then it optionally chooses a variant from that same folder.

Markers should be conceptual signals, not exhaustive regex-style rules. Extended definitions participate in variant routing only when they add variant-specific markers with `strong_add` or `weak_add`. If no variant clearly matches, the family `common.yaml` is used.

Automatic routing is useful for demonstrating local LLM classification, but it is model-dependent. For repeatable tests or demos, provide `--document-definition` explicitly.

## Fields

A field describes one redaction target.

Example:

```yaml
fields:
  date_of_birth:
    label: "Date of birth"
    description: "The holder's date of birth."
    anchors:
      - "Date of birth"
      - "DOB"
```

Important properties:

- `label`: human-readable field name
- `description`: conceptual explanation for the LLM
- `anchors`: labels or headings associated with the field
- `match_hints`: extra guidance for difficult fields
- `max_value_fragments`: optional cap when a value may span more than one OCR fragment; default is `1`
- `repeat_detection`: optional `true`/`false` flag for fields that may appear again without a clear label; default is `false`

Text fields use OCR boxes and solid black rectangles by default. The pipeline intentionally redacts full OCR boxes to avoid partial-character leakage; this may over-redact nearby labels or adjacent text.

Optional field-level redaction settings:

```yaml
redaction:
  trim_leading_token_lengths:
    - 3
```

`trim_leading_token_lengths` trims a leading all-caps token from a selected OCR box when OCR combines a short label/code and the value into one fragment. Use it only for document-specific cleanup after testing.

## Review Policy

The optional `review` section describes which redactions must succeed before a run can be considered complete.

```yaml
review:
  required_fields:
    - "license_number"
    - "date_of_birth"
  required_groups:
    - id: "holder_name"
      label: "Cardholder name"
      any_of:
        - "name_full"
      all_of:
        - "surname"
        - "given_names"
```

`required_fields` are individual fields that must produce at least one valid redaction box.

`required_groups` handle either/or requirements. The example above passes if `name_full` is redacted, or if both `surname` and `given_names` are redacted.

If a review requirement is not met, the run returns `needs_review` with exit code `2`. In normal mode, no redacted image is written for that run. In debug mode, a partial redacted image may be written for troubleshooting.

## Field Overrides

Extended definitions should avoid repeating the full parent definition. Use `field_overrides` instead.

```yaml
field_overrides:
  passport_number:
    anchors_add:
      - "Documentnummer"
```

Supported additive patterns include:

- `anchors_add`
- `match_hints_add`

## Validation

Validate a document definition before using it in the main pipeline:

```powershell
python -m redactor.document_definition_validator --definition document_definitions\licenses\common.yaml
```

The validator loads inherited definitions, checks the supported YAML shape, and confirms review fields/groups reference known field ids.

## Repeat Detection

Some document definitions enable a second LLM pass:

```yaml
fields:
  license_number:
    label: "Driver's license number"
    repeat_detection: true
```

This pass only looks for repeats or near-repeats of values already matched in the first pass for fields with `repeat_detection: true`. It should not discover unrelated new PII categories.

This is useful for documents where the same identifier appears in more than one place, sometimes without a nearby label.

## Fallback Detection

Some document families can enable a final LLM-assisted fallback pass:

```yaml
fallback_detection:
  opaque_identifier:
    enabled: true
    label: "Opaque machine-readable identifier"
    max_value_fragments: 3
    description: "A long opaque identifier, side number, audit number, vertical number, or machine-readable string."
    match_hints:
      - "Run only on OCR fragments that were not already matched by configured fields."
      - "Redact long standalone alphanumeric strings that appear to be machine-readable identifiers."
```

Fallback detection runs after normal field association and repeat detection. It only sees OCR fragments that have not already been selected, so it is best used as a conservative family-specific safeguard for leftover opaque identifiers.

This should not be used as a universal PII detector. Keep fallback detectors narrow, document-family-specific, and easy to review.

## Adding A New Document Family

Suggested process:

1. Create `document_definitions/<family>/common.yaml`.
2. Add routing markers that identify the family.
3. Add a small set of high-value PII fields first.
4. Test with synthetic or safe sample images.
5. Add field hints only where the LLM repeatedly fails.
6. Create extended definitions only when a specific document variant needs them.
7. Add fallback detection only when a document family has common leftover opaque identifiers.

Keep YAML definitions readable. If a definition starts to become a large rule engine, step back and decide whether the field should be document-specific, out of scope, or handled by another local detector.

## Current Document Families

Passports:

- common passport biographical data page fields
- Netherlands-specific extension

Licenses:

- common driver's license fields
- Ohio-specific extension

Invoices:

- customer names, addresses, account/reference fields, and contact details

Credit cards:

- front-of-card number, cardholder name, visible card dates, and front security code when present

Cheques:

- account holder, payee, routing/account-style identifiers, and cheque-related PII

Health insurance cards:

- member names, card/member ids, policy/group/certificate-style identifiers, sex, and date of birth
