from pathlib import Path

import associate_fields


DEFAULT_STRONG_WEIGHT = 5
DEFAULT_WEAK_WEIGHT = 1


def normalize_text(value):
    return "".join(str(value).casefold().split())


def ocr_text_blob(ocr_manifest):
    parts = []
    for fragment in ocr_manifest.get("fragments", []):
        text = fragment.get("text")
        if text:
            parts.append(str(text))
    return "\n".join(parts)


def marker_matches(text_blob, markers):
    normalized_blob = normalize_text(text_blob)
    matches = []
    for marker in markers or []:
        if normalize_text(marker) in normalized_blob:
            matches.append(marker)
    return matches


def score_definition(document_definition, text_blob):
    routing = document_definition.get("routing") or {}
    markers = routing.get("markers") or {}

    strong_matches = marker_matches(text_blob, markers.get("strong", []))
    weak_matches = marker_matches(text_blob, markers.get("weak", []))

    strong_weight = routing.get("strong_weight", DEFAULT_STRONG_WEIGHT)
    weak_weight = routing.get("weak_weight", DEFAULT_WEAK_WEIGHT)
    score = len(strong_matches) * strong_weight + len(weak_matches) * weak_weight

    return {
        "document_type": document_definition.get("id"),
        "label": document_definition.get("label"),
        "score": score,
        "strong_matches": strong_matches,
        "weak_matches": weak_matches,
        "threshold": routing.get("threshold", 1),
    }


def iter_routable_definition_paths(definitions_dir):
    definitions_dir = Path(definitions_dir)
    for path in sorted(definitions_dir.rglob("*.yaml")):
        raw_definition = associate_fields.load_yaml(path)
        routing = raw_definition.get("routing") or {}
        if routing.get("enabled", False):
            yield path


def route_document(ocr_manifest, definitions_dir):
    text_blob = ocr_text_blob(ocr_manifest)
    candidates = []

    for path in iter_routable_definition_paths(definitions_dir):
        document_definition = associate_fields.load_document_definition(path)
        candidate = score_definition(document_definition, text_blob)
        candidate["document_definition"] = str(path)
        candidates.append(candidate)

    candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)
    viable_candidates = [
        candidate
        for candidate in candidates
        if candidate["score"] >= candidate["threshold"]
    ]

    result = {
        "status": "unsupported_document",
        "definitions_dir": str(definitions_dir),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }

    if not viable_candidates:
        result["reason"] = "No routable document definition met its routing threshold."
        return None, result

    best_candidate = viable_candidates[0]
    tied_candidates = [
        candidate
        for candidate in viable_candidates
        if candidate["score"] == best_candidate["score"]
    ]
    if len(tied_candidates) > 1:
        result.update(
            {
                "status": "ambiguous_document",
                "reason": "Multiple document definitions tied for the best routing score.",
                "tied_candidates": tied_candidates,
            }
        )
        return None, result

    result.update(
        {
            "status": "completed",
            "selected_document_definition": best_candidate["document_definition"],
            "selected_document_type": best_candidate["document_type"],
            "selected_score": best_candidate["score"],
            "selected_strong_matches": best_candidate["strong_matches"],
            "selected_weak_matches": best_candidate["weak_matches"],
        }
    )
    return Path(best_candidate["document_definition"]), result
