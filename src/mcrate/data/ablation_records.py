from __future__ import annotations

import copy
import random
from collections import defaultdict
from typing import Any


ABLATION_NONE = "none"
ABLATION_SENSITIVE_VALUE_SHUFFLE = "sensitive_value_shuffle"
ABLATION_ANCHOR_SHUFFLE = "anchor_shuffle"

SUPPORTED_ABLATIONS = {
    ABLATION_NONE,
    "original",
    ABLATION_SENSITIVE_VALUE_SHUFFLE,
    ABLATION_ANCHOR_SHUFFLE,
}


def derangement_indices(size: int, rng: random.Random) -> list[int]:
    """Return a deterministic derangement for size > 1."""
    if size <= 1:
        return list(range(size))
    indices = list(range(size))
    for _ in range(100):
        shuffled = list(indices)
        rng.shuffle(shuffled)
        if all(left != right for left, right in zip(indices, shuffled)):
            return shuffled
    # Rotation is a deterministic fallback and is a derangement for size > 1.
    return indices[1:] + indices[:1]


def _member_family_groups(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record.get("membership") == "member" and record.get("family") != "canary":
            groups[str(record["family"])].append(index)
    return groups


def _sensitive_fields_for(records: list[dict[str, Any]], indices: list[int]) -> list[str]:
    fields: set[str] = set()
    for index in indices:
        fields.update(str(name) for name in records[index].get("sensitive_fields", []))
    return sorted(fields)


def _anchor_fields_for(records: list[dict[str, Any]], indices: list[int]) -> list[str]:
    fields: set[str] = {"public_handle"}
    for index in indices:
        fields.update(str(name) for name in records[index].get("anchor_fields", []))
    return sorted(fields)


def transform_records_for_ablation(
    records: list[dict[str, Any]],
    *,
    ablation_name: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create train-insertion records for destructive ablations.

    The returned record table is intended for rendering/training only. Audit
    targets should remain the original untransformed records so original-target
    lift can collapse when the train-time association is destroyed.
    """

    normalized = ablation_name or ABLATION_NONE
    if normalized == "original":
        normalized = ABLATION_NONE
    if normalized not in SUPPORTED_ABLATIONS:
        raise ValueError(f"Unsupported ablation_name: {ablation_name}")
    transformed = copy.deepcopy(records)
    if normalized == ABLATION_NONE:
        return transformed, []

    manifest: list[dict[str, Any]] = []
    groups = _member_family_groups(transformed)
    for family, indices in sorted(groups.items()):
        rng = random.Random(seed + sum(ord(ch) for ch in family) + sum(ord(ch) for ch in normalized))
        permutation = derangement_indices(len(indices), rng)
        source_by_index = {indices[pos]: indices[permutation[pos]] for pos in range(len(indices))}

        if normalized == ABLATION_SENSITIVE_VALUE_SHUFFLE:
            sensitive_fields = _sensitive_fields_for(transformed, indices)
            for target_index in indices:
                donor_index = source_by_index[target_index]
                target = transformed[target_index]
                donor = transformed[donor_index]
                field_changes: dict[str, dict[str, Any]] = {}
                for field_name in sensitive_fields:
                    if field_name not in target.get("fields", {}) or field_name not in donor.get("fields", {}):
                        continue
                    original_value = target["fields"][field_name]
                    shuffled_value = donor["fields"][field_name]
                    target["fields"][field_name] = shuffled_value
                    field_changes[field_name] = {
                        "original_value": original_value,
                        "inserted_value": shuffled_value,
                        "donor_record_id": donor["record_id"],
                    }
                if field_changes:
                    manifest.append(
                        {
                            "ablation_name": normalized,
                            "record_id": target["record_id"],
                            "family": family,
                            "donor_record_id": donor["record_id"],
                            "changed_fields": field_changes,
                        }
                    )

        if normalized == ABLATION_ANCHOR_SHUFFLE:
            anchor_fields = _anchor_fields_for(transformed, indices)
            for target_index in indices:
                donor_index = source_by_index[target_index]
                target = transformed[target_index]
                donor = transformed[donor_index]
                field_changes: dict[str, dict[str, Any]] = {}
                for field_name in anchor_fields:
                    if field_name == "public_handle":
                        original_value = target.get("public_handle")
                        shuffled_value = donor.get("public_handle")
                        if shuffled_value:
                            target["public_handle"] = shuffled_value
                    else:
                        if field_name not in target.get("fields", {}) or field_name not in donor.get("fields", {}):
                            continue
                        original_value = target["fields"][field_name]
                        shuffled_value = donor["fields"][field_name]
                        target["fields"][field_name] = shuffled_value
                    field_changes[field_name] = {
                        "original_value": original_value,
                        "inserted_value": shuffled_value,
                        "donor_record_id": donor["record_id"],
                    }
                if field_changes:
                    manifest.append(
                        {
                            "ablation_name": normalized,
                            "record_id": target["record_id"],
                            "family": family,
                            "donor_record_id": donor["record_id"],
                            "changed_fields": field_changes,
                        }
                    )
    return transformed, manifest
