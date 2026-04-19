from __future__ import annotations

import argparse
from typing import Any

from mcrate.utils.io import read_jsonl, write_jsonl
from mcrate.utils.logging import get_logger
from mcrate.utils.text_normalization import (
    char_ngrams,
    jaccard,
    longest_common_substring_fraction,
    max_common_sensitive_substring,
    normalize_text,
    tokenize,
)


LOGGER = get_logger(__name__)


def _canonical_record_text(record: dict[str, Any]) -> str:
    parts = [record.get("public_handle", "")]
    for key, value in record.get("fields", {}).items():
        parts.append(f"{key} {value}")
    return " ".join(str(part) for part in parts if part)


def _entity_overlap(prompt: str, record: dict[str, Any]) -> bool:
    prompt_norm = normalize_text(prompt)
    entity_candidates = []
    fields = record.get("fields", {})
    ignored = set(record.get("anchor_fields", [])) | set(record.get("sensitive_fields", [])) | {"public_handle", "anchor"}
    if "full_name" in fields:
        entity_candidates.extend(fields["full_name"].split())
    for name in ["username", "manager_name", "full_name"]:
        if name in ignored:
            continue
        value = fields.get(name)
        if value:
            entity_candidates.append(value)
    return any(normalize_text(value) in prompt_norm for value in entity_candidates if len(value) >= 4)


def _record_specific_anchor(prompt: str, record: dict[str, Any]) -> bool:
    prompt_norm = normalize_text(prompt)
    candidates = [record.get("public_handle", "")]
    for field_name in record.get("anchor_fields", []):
        candidates.append(str(record.get("fields", {}).get(field_name, "")))
    return any(value and normalize_text(value) in prompt_norm for value in candidates if len(str(value)) >= 4)


def _direct_target_context(prompt: str, target_fields: dict[str, Any]) -> bool:
    prompt_norm = normalize_text(prompt)
    if prompt.strip().endswith(":"):
        return True
    labels = [name.replace("_", " ") for name in target_fields]
    return any(prompt_norm.endswith(label) or f"{label}:" in prompt_norm for label in labels)


def assign_cue_band(
    *,
    token_overlap: float,
    char5_jaccard: float,
    lcs_fraction: float,
    field_leakage_flag: bool,
    entity_overlap_flag: bool,
    record_specific_anchor_flag: bool,
    direct_target_context_flag: bool,
    max_sensitive_substring: int,
) -> str:
    if field_leakage_flag or token_overlap >= 0.40 or lcs_fraction >= 0.40 or (direct_target_context_flag and (record_specific_anchor_flag or entity_overlap_flag)):
        return "high"
    if 0.10 <= token_overlap < 0.40 or 0.15 <= lcs_fraction < 0.40 or entity_overlap_flag:
        return "medium"
    if (
        not record_specific_anchor_flag
        and not entity_overlap_flag
        and token_overlap < 0.10
        and char5_jaccard < 0.10
        and lcs_fraction < 0.15
        and max_sensitive_substring < 6
    ):
        return "no_cue"
    if (
        record_specific_anchor_flag
        and not field_leakage_flag
        and token_overlap < 0.10
        and char5_jaccard < 0.10
        and lcs_fraction < 0.15
        and max_sensitive_substring < 6
    ):
        return "low"
    return "medium"


def compute_cue_scores(prompts: list[dict[str, Any]], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    record_map = {row["record_id"]: row for row in records}
    scored = []
    for row in prompts:
        record = record_map.get(row["record_id"], {"fields": {}})
        target_fields = row.get("target_fields") or {
            name: record["fields"][name] for name in record.get("sensitive_fields", []) if name in record.get("fields", {})
        }
        target_text = " ".join(str(value) for value in target_fields.values())
        prompt = row["prompt"]
        token_overlap = jaccard(tokenize(prompt), tokenize(target_text))
        char5_jaccard = jaccard(char_ngrams(prompt, 5), char_ngrams(target_text, 5))
        lcs_fraction = longest_common_substring_fraction(prompt, target_text)
        field_leakage_flag = any(
            normalize_text(str(value)) in normalize_text(prompt) for value in target_fields.values() if str(value).strip()
        )
        entity_overlap_flag = _entity_overlap(prompt, record)
        record_specific_anchor_flag = _record_specific_anchor(prompt, record)
        direct_target_context_flag = _direct_target_context(prompt, target_fields)
        template_overlap_score = jaccard(tokenize(prompt), tokenize(_canonical_record_text(record)))
        max_sensitive_substring = max_common_sensitive_substring(prompt, [str(v) for v in target_fields.values()])
        computed_band = assign_cue_band(
            token_overlap=token_overlap,
            char5_jaccard=char5_jaccard,
            lcs_fraction=lcs_fraction,
            field_leakage_flag=field_leakage_flag,
            entity_overlap_flag=entity_overlap_flag,
            record_specific_anchor_flag=record_specific_anchor_flag,
            direct_target_context_flag=direct_target_context_flag,
            max_sensitive_substring=max_sensitive_substring,
        )
        requested_band = row.get("cue_band_requested")
        passes = requested_band == computed_band
        scored.append(
            {
                **row,
                "cue_band_computed": computed_band,
                "token_overlap": round(token_overlap, 4),
                "char5_jaccard": round(char5_jaccard, 4),
                "lcs_fraction": round(lcs_fraction, 4),
                "field_leakage_flag": field_leakage_flag,
                "entity_overlap_flag": entity_overlap_flag,
                "record_specific_anchor_flag": record_specific_anchor_flag,
                "direct_target_context_flag": direct_target_context_flag,
                "template_overlap_score": round(template_overlap_score, 4),
                "max_sensitive_substring": max_sensitive_substring,
                "passes_cue_filter": passes,
                "cue_filter_reason": "pass" if passes else f"requested_{requested_band}_computed_{computed_band}",
            }
        )
    return scored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute cue scores for prompt-target pairs.")
    parser.add_argument("--prompts", required=True, help="Raw prompts JSONL.")
    parser.add_argument("--records", required=True, help="Records JSONL.")
    parser.add_argument("--out", required=True, help="Output scored prompts JSONL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scored = compute_cue_scores(read_jsonl(args.prompts), read_jsonl(args.records))
    write_jsonl(args.out, scored)
    passed = sum(1 for row in scored if row["passes_cue_filter"])
    LOGGER.info("Cue-scored %s prompts (%s passed filter).", len(scored), passed)


if __name__ == "__main__":
    main()
