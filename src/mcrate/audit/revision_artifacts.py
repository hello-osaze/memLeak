from __future__ import annotations

import csv
import gzip
import json
import platform
from pathlib import Path
from typing import Any

from mcrate.audit.aggregate_results import aggregate
from mcrate.utils.io import dump_yaml, ensure_parent, load_yaml, read_jsonl, write_jsonl, write_text
from mcrate.utils.text_normalization import normalize_text


def budget_from_generation_config(path: str | Path) -> int:
    try:
        config = load_yaml(path)
    except FileNotFoundError:
        config = {}
    if "num_return_sequences" in config:
        return int(config["num_return_sequences"])
    name = Path(path).stem
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else 1


def _revision_row(
    *,
    condition: str,
    seed: int,
    generation_config_path: str,
    ablation_name: str,
    matching_variant: str,
    cue_filter_strength: str,
    cue_variant: str,
    scoring_mode: str,
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    logprob_by_cue = {row["cue_band"]: row for row in summary.get("logprob_table", [])}
    output = []
    for row in summary.get("cue_table", []):
        logprob = logprob_by_cue.get(row["cue_band"], {})
        output.append(
            {
                "condition": condition,
                "seed": seed,
                "ablation_name": ablation_name,
                "matching_variant": matching_variant,
                "cue_filter_strength": cue_filter_strength,
                "cue_variant": cue_variant,
                "scoring_mode": scoring_mode,
                "cue_band": row["cue_band"],
                "budget_B": budget_from_generation_config(generation_config_path),
                "n_member": row["member_tasks"],
                "n_nonmember": row["nonmember_tasks"],
                "r_mem": row["member_extraction"],
                "r_non": row["nonmember_extraction"],
                "delta_mem": row["lift"],
                "delta_mem_CI": row["lift_ci95"],
                "exact_record_mem": row["member_record_exact"],
                "exact_record_non": row["nonmember_record_exact"],
                "field_F1_mem": row.get("member_field_f1", 0.0),
                "field_F1_non": row.get("nonmember_field_f1", 0.0),
                "teacher_forced_delta": logprob.get("delta", 0.0),
            }
        )
    return output


def write_revision_result_tables(
    *,
    rows: list[dict[str, Any]],
    jsonl_path: str | Path,
    csv_path: str | Path,
) -> None:
    baseline: dict[tuple[Any, ...], float] = {}
    for row in rows:
        key = (
            row["condition"],
            row["seed"],
            row["matching_variant"],
            row["cue_filter_strength"],
            row["cue_variant"],
            row["scoring_mode"],
            row["cue_band"],
            row["budget_B"],
        )
        if row["ablation_name"] in {"none", "original"}:
            baseline[key] = float(row["delta_mem"])
    enriched = []
    for row in rows:
        key = (
            row["condition"],
            row["seed"],
            row["matching_variant"],
            row["cue_filter_strength"],
            row["cue_variant"],
            row["scoring_mode"],
            row["cue_band"],
            row["budget_B"],
        )
        original = baseline.get(key)
        if original in {None, 0.0}:
            collapse_ratio = None
        else:
            collapse_ratio = round(float(row["delta_mem"]) / float(original), 6)
        enriched.append({**row, "collapse_ratio": collapse_ratio})
    write_jsonl(jsonl_path, enriched)
    if not enriched:
        write_text(csv_path, "")
        return
    target = ensure_parent(csv_path)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(enriched[0].keys()))
        writer.writeheader()
        writer.writerows(enriched)


def collect_audit_revision_rows(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit in units:
        if unit.get("phase") not in {"audit", "adaptive_attack"}:
            continue
        payload = unit.get("payload", {})
        score_path = Path(payload.get("score_path", ""))
        if not score_path.exists():
            continue
        scores = read_jsonl(score_path)
        if not scores:
            continue
        summary = aggregate(scores)
        base_rows = _revision_row(
            condition=payload.get("base_condition", payload.get("condition", "unknown")),
            seed=int(payload.get("seed", 0)),
            generation_config_path=payload.get("base_generation_config_path") or payload.get("generated_generation_config_path", ""),
            ablation_name=payload.get("ablation_name", "none"),
            matching_variant=payload.get("matching_variant", "full"),
            cue_filter_strength=payload.get("cue_filter_strength", "legacy"),
            cue_variant=payload.get("cue_variant", "full"),
            scoring_mode=payload.get("scoring_mode", "legacy"),
            summary=summary,
        )
        for row in base_rows:
            if unit.get("phase") == "adaptive_attack":
                row.update(
                    {
                        "attack_type": payload.get("attack_type"),
                        "attack_objective": payload.get("objective"),
                        "generation_budget_G": payload.get("generation_budget"),
                        "number_of_prompts_used": payload.get("number_of_prompts_used"),
                        "number_of_generations_used": payload.get("generation_budget"),
                    }
                )
            rows.append(row)
    return rows


def archive_raw_generations(generation_paths: list[str | Path], out_path: str | Path) -> dict[str, Any]:
    target = ensure_parent(out_path)
    written = 0
    with gzip.open(target, "wt", encoding="utf-8") as handle:
        for path in generation_paths:
            source = Path(path)
            if not source.exists():
                continue
            for row in read_jsonl(source):
                payload = {
                    **row,
                    "completion": row.get("completion", row.get("output_text", "")),
                    "normalized_completion": normalize_text(str(row.get("output_text", row.get("completion", "")))),
                    "matched_fields": row.get("field_matches", {}),
                    "exact_record_match": row.get("record_exact"),
                    "any_sensitive_match": row.get("any_sensitive_match"),
                    "scoring_version": row.get("scoring_mode", "unknown"),
                }
                handle.write(json.dumps(payload, sort_keys=False) + "\n")
                written += 1
    return {"raw_generation_rows": written, "archive_path": str(target)}


def write_runtime_stub(path: str | Path) -> None:
    lines = [
        "# Hardware And Runtime",
        "",
        f"- Host platform: `{platform.platform()}`",
        f"- Python: `{platform.python_version()}`",
        "- GPU type: recorded by run environment if available",
        "- CUDA/PyTorch/Transformers versions: recorded by run environment if available",
        "- Training, audit, adaptive attack, provenance runtimes: populated from unit markers after execution",
        "",
    ]
    write_text(path, "\n".join(lines))


def write_generation_config_artifact(generation_configs: list[str], out_path: str | Path) -> None:
    payload = {"generation_configs": []}
    for path in generation_configs:
        config = load_yaml(path)
        payload["generation_configs"].append({"source": str(path), **config})
    dump_yaml(out_path, payload)
