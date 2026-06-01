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

Definitions can opt into automatic routing:

```yaml
routing:
  enabled: true
  markers:
    strong:
      - "Passport"
      - "Passport No"
    weak:
      - "Nationality"
      - "Date of birth"
```

When `redact.py` is called without `--document-definition`, OCR snippets and routing markers are sent to the local LLM. The LLM chooses one supported definition or returns an unsupported/ambiguous status.

Markers should be conceptual signals, not exhaustive regex-style rules.

## Fields

A field describes one redaction target.

Example:

```yaml
fields:
  date_of_birth:
    label: "Date of birth"
    type: "text"
    description: "The holder's date of birth."
    anchors:
      - "Date of birth"
      - "DOB"
    redaction:
      mode: "ocr_box"
```

Important properties:

- `label`: human-readable field name
- `type`: simple category such as `text` or `mrz`
- `description`: conceptual explanation for the LLM
- `anchors`: labels or headings associated with the field
- `match_hints`: extra guidance for difficult fields
- `max_value_fragments`: number of OCR fragments the LLM may choose
- `redaction.mode`: currently `ocr_box`

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
- `excluded_fragment_tags_add`

## Fragment Tags

Fragment tags mark OCR fragments with simple local rules before they reach validation.

Example:

```yaml
fragment_tags:
  mrz:
    label: "Machine-readable zone text"
    text_contains:
      - "<"
```

Tags are useful when a document has known regions or special text patterns, such as passport MRZ lines.

## Repeat Detection

Some document definitions enable a second LLM pass:

```yaml
repeat_detection:
  enabled: true
  field_ids:
    - "license_number"
```

This pass only looks for repeats or near-repeats of values already matched in the first pass. It should not discover unrelated new PII categories.

This is useful for documents where the same identifier appears in more than one place, sometimes without a nearby label.

## Adding A New Document Family

Suggested process:

1. Create `document_definitions/<family>/common.yaml`.
2. Add routing markers that identify the family.
3. Add a small set of high-value PII fields first.
4. Test with synthetic or safe sample images.
5. Add field hints only where the LLM repeatedly fails.
6. Create extended definitions only when a specific document variant needs them.

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

Cheques:

- account holder, payee, routing/account-style identifiers, and cheque-related PII

Health insurance cards:

- member names, card/member ids, policy/group/certificate-style identifiers, sex, and date of birth
