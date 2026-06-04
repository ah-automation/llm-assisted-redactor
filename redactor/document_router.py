from pathlib import Path
import json

from redactor import associate_fields
from openai import OpenAI


MAX_LLM_TEXT_SNIPPETS = 160
DEFAULT_CONFIDENCE_THRESHOLD = 0.75


def get_routing_config(config):
    routing = config.get("routing") or {}
    return routing if isinstance(routing, dict) else {}


def get_confidence_threshold(config):
    threshold = get_routing_config(config).get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
    try:
        return float(threshold)
    except (TypeError, ValueError):
        return DEFAULT_CONFIDENCE_THRESHOLD


def is_debug_enabled(config):
    debug = config.get("debug") or {}
    return bool(debug.get("enabled", False)) if isinstance(debug, dict) else False


def add_raw_response_if_debug(target, config, raw_response):
    if is_debug_enabled(config):
        target["raw_response"] = raw_response


def get_llm_config(config):
    return config["llm"]


def has_routing_markers(path):
    raw_definition = associate_fields.load_yaml(path)
    routing = raw_definition.get("routing") or {}
    markers = routing.get("markers") or {}
    marker_keys = ("strong", "weak", "strong_add", "weak_add")
    return any(markers.get(key) for key in marker_keys)


def build_route_candidate(path, scope):
    document_definition = associate_fields.load_document_definition(path)
    routing = document_definition.get("routing") or {}
    markers = routing.get("markers") or {}
    return {
        "document_definition": str(path),
        "document_type": document_definition.get("id"),
        "label": document_definition.get("label"),
        "description": document_definition.get("description"),
        "definition_scope": scope,
        "markers": {
            "strong": markers.get("strong", []),
            "weak": markers.get("weak", []),
        },
    }


def build_family_route_candidates(definitions_dir):
    candidates = []
    definitions_dir = Path(definitions_dir)
    for path in sorted(definitions_dir.glob("*/common.yaml")):
        if has_routing_markers(path):
            candidates.append(build_route_candidate(path, "common_family"))
    return candidates


def build_variant_route_candidates(common_definition_path):
    common_definition_path = Path(common_definition_path)
    candidates = [build_route_candidate(common_definition_path, "common_family")]
    for path in sorted(common_definition_path.parent.glob("*.yaml")):
        if path.name == "common.yaml":
            continue
        if has_routing_markers(path):
            candidates.append(build_route_candidate(path, "specific_variant"))
    return candidates


def build_llm_route_candidates(definitions_dir):
    candidates = []
    for family_candidate in build_family_route_candidates(definitions_dir):
        candidates.extend(build_variant_route_candidates(family_candidate["document_definition"]))
    return candidates


def ocr_text_snippets(ocr_manifest):
    snippets = []
    for fragment in ocr_manifest.get("fragments", []):
        text = str(fragment.get("text", "")).strip()
        if text:
            snippets.append(text)
    return snippets[:MAX_LLM_TEXT_SNIPPETS]


def get_llm_route_response_format():
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "document_route",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["completed", "unsupported_document", "ambiguous_document"],
                    },
                    "document_type": {"type": "string"},
                    "document_definition": {"type": "string"},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": [
                    "status",
                    "document_type",
                    "document_definition",
                    "confidence",
                    "reason",
                ],
                "additionalProperties": False,
            },
        },
    }


def call_llm_router(config, prompt):
    llm_config = get_llm_config(config)
    client = OpenAI(
        base_url=llm_config["base_url"],
        api_key=llm_config.get("api_key", "lm-studio"),
    )

    response = client.chat.completions.create(
        model=llm_config["model"],
        temperature=llm_config.get("temperature", 0),
        max_tokens=llm_config.get("max_tokens", 1000),
        response_format=get_llm_route_response_format(),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document routing engine. "
                    "Choose one supported document type using OCR snippets and routing markers. "
                    "Return only JSON. Do not repeat OCR text or PII in the reason."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    choice = response.choices[0]
    content = choice.message.content or ""
    diagnostic = {
        "finish_reason": choice.finish_reason,
        "message_role": choice.message.role,
        "content_is_empty": not bool(content),
    }
    if response.usage:
        diagnostic["completion_tokens"] = response.usage.completion_tokens
        token_details = getattr(response.usage, "completion_tokens_details", None)
        if token_details:
            diagnostic["reasoning_tokens"] = getattr(token_details, "reasoning_tokens", None)

    return content, diagnostic


def parse_llm_route(raw_response):
    cleaned = associate_fields.clean_json_response(raw_response)
    if not cleaned:
        raise ValueError("LLM returned an empty routing response.")

    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("LLM routing response must be an object.")
    return parsed


def route_with_candidates(config, ocr_manifest, definitions_dir, candidates, stage, instructions):
    confidence_threshold = get_confidence_threshold(config)
    request = {
        "ocr_text_snippets": ocr_text_snippets(ocr_manifest),
        "document_types": candidates,
        "confidence_threshold": confidence_threshold,
    }
    prompt = (
        f"{instructions}\n"
        "Markers are conceptual and OCR may contain typos, missing spaces, or merged words.\n"
        "Use document_type and document_definition exactly from the provided list.\n"
        "If no type fits, use status unsupported_document and empty strings for document_type/document_definition.\n"
        "If more than one type fits equally, use status ambiguous_document and empty strings for document_type/document_definition.\n"
        f"{json.dumps(request, separators=(',', ':'))}"
    )

    raw_response, diagnostic = call_llm_router(config, prompt)
    result = {
        "status": "started",
        "stage": stage,
        "definitions_dir": str(definitions_dir),
        "candidate_count": len(candidates),
        "confidence_threshold": confidence_threshold,
        "llm_diagnostic": diagnostic,
    }

    try:
        parsed = parse_llm_route(raw_response)
    except (json.JSONDecodeError, ValueError) as error:
        result.update(
            {
                "status": "llm_response_error",
                "error": "LLM routing response was not valid JSON in the expected shape.",
                "error_details": str(error),
            }
        )
        add_raw_response_if_debug(result, config, raw_response)
        return None, result

    known_paths = {candidate["document_definition"] for candidate in candidates}
    selected_path = parsed.get("document_definition", "")
    if parsed.get("status") == "completed" and selected_path not in known_paths:
        result.update(
            {
                "status": "llm_response_error",
                "error": "LLM selected an unknown document definition.",
                "selected_document_definition": selected_path,
            }
        )
        add_raw_response_if_debug(result, config, raw_response)
        return None, result

    confidence = parsed.get("confidence")
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.0

    result.update(
        {
            "status": parsed.get("status"),
            "selected_document_type": parsed.get("document_type", ""),
            "selected_document_definition": selected_path,
            "confidence": confidence_value,
            "reason": parsed.get("reason", "") if is_debug_enabled(config) else "",
        }
    )

    if result["status"] != "completed":
        return None, result

    if confidence_value < confidence_threshold:
        result.update(
            {
                "status": "low_confidence",
                "reason": "LLM selected a document definition below the configured confidence threshold.",
            }
        )
        return None, result

    return Path(selected_path), result


def route_document_with_llm(config, ocr_manifest, definitions_dir):
    family_candidates = build_family_route_candidates(definitions_dir)
    family_path, family_result = route_with_candidates(
        config,
        ocr_manifest,
        definitions_dir,
        family_candidates,
        "family",
        (
            "Select the best document family. Each candidate is a common-family definition. "
            "Use unsupported_document only when none of the document families fit the OCR snippets."
        ),
    )
    if family_path is None:
        family_result["routing_strategy"] = "family_then_variant"
        return None, family_result

    variant_candidates = build_variant_route_candidates(family_path)
    if len(variant_candidates) == 1:
        family_result.update(
            {
                "routing_strategy": "family_then_variant",
                "family_routing": family_result.copy(),
                "variant_routing": {
                    "status": "skipped",
                    "reason": "No variant definitions with routing markers are available for this family.",
                    "candidate_count": 1,
                },
            }
        )
        return family_path, family_result

    variant_path, variant_result = route_with_candidates(
        config,
        ocr_manifest,
        definitions_dir,
        variant_candidates,
        "variant",
        (
            "A document family has already been selected. Choose a specific-variant definition only when "
            "OCR evidence clearly supports that variant. Otherwise choose the common-family definition. "
            "Do not return unsupported_document just because a specific country, state, province, vendor, "
            "or variant definition is missing."
        ),
    )

    if variant_path is None:
        selected_path = family_path
        selected_definition = associate_fields.load_document_definition(selected_path)
        combined_result = {
            "status": "completed",
            "routing_strategy": "family_then_variant",
            "definitions_dir": str(definitions_dir),
            "candidate_count": len(family_candidates) + len(variant_candidates),
            "confidence_threshold": get_confidence_threshold(config),
            "selected_document_type": selected_definition.get("id"),
            "selected_document_definition": str(selected_path),
            "confidence": family_result.get("confidence", 0.0),
            "reason": "Variant routing did not select a specific variant; using the common-family definition."
            if is_debug_enabled(config)
            else "",
            "family_routing": family_result,
            "variant_routing": variant_result,
        }
        return selected_path, combined_result

    variant_result.update(
        {
            "routing_strategy": "family_then_variant",
            "family_routing": family_result,
            "variant_routing": variant_result.copy(),
        }
    )
    return variant_path, variant_result
