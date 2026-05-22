from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any

from mcrate.utils.text_normalization import normalize_text, normalize_value, tokenize


MATCHING_VARIANTS = ("full", "family_only", "strict")


def _record_text(record: dict[str, Any], *, sensitive: bool = False) -> str:
    parts = [str(record.get("public_handle", ""))]
    blocked = set() if sensitive else set(record.get("sensitive_fields", []))
    for key, value in record.get("fields", {}).items():
        if key not in blocked:
            parts.append(f"{key} {value}")
    return " ".join(parts)


def _record_length(record: dict[str, Any]) -> int:
    return len(tokenize(_record_text(record, sensitive=True)))


def _non_sensitive_length(record: dict[str, Any]) -> int:
    return len(tokenize(_record_text(record, sensitive=False)))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _lexical_overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    return _jaccard(set(tokenize(_record_text(left, sensitive=False))), set(tokenize(_record_text(right, sensitive=False))))


def _anchor_overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    def values(record: dict[str, Any]) -> set[str]:
        output = {normalize_text(str(record.get("public_handle", "")))}
        for field in record.get("anchor_fields", []):
            if field in record.get("fields", {}):
                output.add(normalize_text(str(record["fields"][field])))
        return {value for value in output if value}

    return _jaccard(values(left), values(right))


def _sensitive_values(record: dict[str, Any]) -> set[str]:
    values = set()
    for field in record.get("sensitive_fields", []):
        if field in record.get("fields", {}):
            values.add(normalize_value(field, str(record["fields"][field])))
    return {value for value in values if value}


def _sensitive_overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    return _jaccard(_sensitive_values(left), _sensitive_values(right))


def _format_signature(record: dict[str, Any], fields: list[str]) -> tuple[str, ...]:
    signature = []
    for field in fields:
        value = str(record.get("fields", {}).get(field, record.get(field, "")))
        if "email" in field:
            signature.append("email")
        elif any(char.isdigit() for char in value) and any(char.isalpha() for char in value):
            signature.append("alnum")
        elif value.isdigit():
            signature.append("digits")
        elif any(char.isdigit() for char in value):
            signature.append("mixed_numeric")
        else:
            signature.append("text")
    return tuple(signature)


def _distance(member: dict[str, Any], candidate: dict[str, Any], *, strict: bool = False) -> float:
    member_length = max(1, _record_length(member))
    candidate_length = max(1, _record_length(candidate))
    length_distance = abs(member_length - candidate_length) / member_length
    nonsens_member_length = max(1, _non_sensitive_length(member))
    nonsens_candidate_length = max(1, _non_sensitive_length(candidate))
    context_length_distance = abs(nonsens_member_length - nonsens_candidate_length) / nonsens_member_length
    anchor_fields = sorted(set(member.get("anchor_fields", [])) | set(candidate.get("anchor_fields", [])))
    sensitive_fields = sorted(set(member.get("sensitive_fields", [])) | set(candidate.get("sensitive_fields", [])))
    anchor_format_distance = 0.0 if _format_signature(member, anchor_fields) == _format_signature(candidate, anchor_fields) else 1.0
    sensitive_format_distance = (
        0.0 if _format_signature(member, sensitive_fields) == _format_signature(candidate, sensitive_fields) else 1.0
    )
    lexical_distance = 1.0 - _lexical_overlap(member, candidate)
    template_distance = 0.0 if member.get("template_id") == candidate.get("template_id") else 0.25
    weights = (0.35, 0.20, 0.15, 0.20, 0.10) if strict else (0.30, 0.15, 0.10, 0.35, 0.10)
    return (
        weights[0] * length_distance
        + weights[1] * context_length_distance
        + weights[2] * anchor_format_distance
        + weights[3] * lexical_distance
        + weights[4] * (template_distance + sensitive_format_distance)
    )


def _strict_candidate(member: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if member.get("family") != candidate.get("family"):
        return False
    if set(member.get("sensitive_fields", [])) != set(candidate.get("sensitive_fields", [])):
        return False
    if set(member.get("anchor_fields", [])) != set(candidate.get("anchor_fields", [])):
        return False
    member_length = max(1, _record_length(member))
    candidate_length = max(1, _record_length(candidate))
    nonsens_member_length = max(1, _non_sensitive_length(member))
    nonsens_candidate_length = max(1, _non_sensitive_length(candidate))
    if abs(member_length - candidate_length) / member_length > 0.10:
        return False
    if abs(nonsens_member_length - nonsens_candidate_length) / nonsens_member_length > 0.10:
        return False
    if _sensitive_overlap(member, candidate) > 0.0:
        return False
    return True


def _pair_row(member: dict[str, Any], nonmember: dict[str, Any], *, variant: str, score: float, fallback: bool = False) -> dict[str, Any]:
    return {
        "member_id": member["record_id"],
        "nonmember_id": nonmember["record_id"],
        "family": member["family"],
        "template_id_member": member.get("template_id"),
        "template_id_nonmember": nonmember.get("template_id"),
        "length_member": _record_length(member),
        "length_nonmember": _record_length(nonmember),
        "rendered_context_length_member": _non_sensitive_length(member),
        "rendered_context_length_nonmember": _non_sensitive_length(nonmember),
        "anchor_overlap": round(_anchor_overlap(member, nonmember), 4),
        "non_sensitive_lexical_overlap": round(_lexical_overlap(member, nonmember), 4),
        "sensitive_overlap": round(_sensitive_overlap(member, nonmember), 4),
        "match_score": round(score, 6),
        "matching_variant": variant,
        "strict_fallback": fallback,
    }


def build_matched_pairs(
    members: list[dict[str, Any]],
    nonmembers: list[dict[str, Any]],
    *,
    variant: str,
    seed: int,
) -> list[dict[str, Any]]:
    if variant not in MATCHING_VARIANTS:
        raise ValueError(f"Unsupported matching variant: {variant}")
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in nonmembers:
        by_family[str(row.get("family"))].append(row)
    pairs: list[dict[str, Any]] = []
    used: set[str] = set()
    for member in sorted(members, key=lambda row: row["record_id"]):
        rng = random.Random(seed + sum(ord(ch) for ch in member["record_id"]) + sum(ord(ch) for ch in variant))
        candidates = [row for row in by_family.get(str(member.get("family")), []) if row["record_id"] not in used]
        if not candidates:
            candidates = by_family.get(str(member.get("family")), []) or nonmembers
        fallback = False
        if variant == "family_only":
            chosen = rng.choice(candidates)
            score = math.nan
        else:
            strict_candidates = [row for row in candidates if _sensitive_overlap(member, row) == 0.0]
            if variant == "strict":
                strict_candidates = [row for row in strict_candidates if _strict_candidate(member, row)]
                if not strict_candidates:
                    strict_candidates = [row for row in candidates if _sensitive_overlap(member, row) == 0.0] or candidates
                    fallback = True
            elif not strict_candidates:
                strict_candidates = candidates
            chosen = min(strict_candidates, key=lambda row: (_distance(member, row, strict=variant == "strict"), row["record_id"]))
            score = _distance(member, chosen, strict=variant == "strict")
        used.add(chosen["record_id"])
        pairs.append(_pair_row(member, chosen, variant=variant, score=0.0 if math.isnan(score) else score, fallback=fallback))
    return pairs


def select_members_for_audit(
    records: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    split_prefix: str = "test_",
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in records
        if row.get("membership") == "member" and row.get("family") != "canary" and str(row.get("split", "")).startswith(split_prefix)
    ]
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_family[str(row["family"])].append(row)
    total = sum(len(rows) for rows in by_family.values())
    if count >= total:
        return sorted(eligible, key=lambda row: row["record_id"])
    selected: list[dict[str, Any]] = []
    for family, rows in sorted(by_family.items()):
        family_target = round(count * len(rows) / max(1, total))
        rng = random.Random(seed + sum(ord(ch) for ch in family))
        shuffled = list(rows)
        rng.shuffle(shuffled)
        selected.extend(shuffled[: min(family_target, len(shuffled))])
    if len(selected) < count:
        remaining = [row for row in eligible if row["record_id"] not in {item["record_id"] for item in selected}]
        rng = random.Random(seed + 17)
        rng.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])
    return sorted(selected[:count], key=lambda row: row["record_id"])


def nonmembers_for_split(records: list[dict[str, Any]], *, split_prefix: str = "test_") -> list[dict[str, Any]]:
    return sorted(
        [
            row
            for row in records
            if row.get("membership") == "nonmember"
            and row.get("family") != "canary"
            and str(row.get("split", "")).startswith(split_prefix)
        ],
        key=lambda row: row["record_id"],
    )
