from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

from mcrate.utils.io import read_jsonl, write_json, write_text
from mcrate.utils.logging import get_logger
from mcrate.utils.stats import bootstrap_ci, safe_mean


LOGGER = get_logger(__name__)


def _collapse_to_task(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        grouped[row["task_id"]].append(row)
    collapsed = []
    for rows in grouped.values():
        sample = rows[0]
        collapsed.append(
            {
                "task_id": sample["task_id"],
                "record_id": sample["record_id"],
                "family": sample["family"],
                "membership": sample["membership"],
                "condition": sample["condition"],
                "cue_band": sample["cue_band"],
                "any_sensitive_match": any(row["any_sensitive_match"] for row in rows),
                "record_exact": any(row["record_exact"] for row in rows),
                "field_f1": safe_mean([row["field_f1"] for row in rows]),
            }
        )
    return collapsed


def _rate(items: list[dict[str, Any]], metric: str) -> float:
    if not items:
        return 0.0
    return sum(1.0 for row in items if row[metric]) / len(items)


def aggregate(scores: list[dict[str, Any]]) -> dict[str, Any]:
    task_rows = _collapse_to_task(scores)
    by_condition_cue = defaultdict(list)
    by_condition_family = defaultdict(list)
    for row in task_rows:
        by_condition_cue[(row["condition"], row["cue_band"], row["membership"])].append(row)
        by_condition_family[(row["condition"], row["family"], row["membership"])].append(row)

    cue_table = []
    keys = sorted({(row["condition"], row["cue_band"]) for row in task_rows})
    for condition, cue_band in keys:
        member = by_condition_cue.get((condition, cue_band, "member"), [])
        nonmember = by_condition_cue.get((condition, cue_band, "nonmember"), [])
        member_rate = _rate(member, "any_sensitive_match")
        nonmember_rate = _rate(nonmember, "any_sensitive_match")
        lift = member_rate - nonmember_rate
        combined = [{"value": 1 if row["any_sensitive_match"] else 0} for row in member]
        ci_low, ci_high = bootstrap_ci(combined, lambda sample: sum(x["value"] for x in sample) / max(1, len(sample))) if combined else (0.0, 0.0)
        cue_table.append(
            {
                "condition": condition,
                "cue_band": cue_band,
                "member_extraction": round(member_rate, 4),
                "nonmember_extraction": round(nonmember_rate, 4),
                "lift": round(lift, 4),
                "member_ci95": [round(ci_low, 4), round(ci_high, 4)],
            }
        )

    family_table = []
    family_keys = sorted({(row["condition"], row["family"]) for row in task_rows if row["membership"] == "member"})
    for condition, family in family_keys:
        member = by_condition_family.get((condition, family, "member"), [])
        nonmember = by_condition_family.get((condition, family, "nonmember"), [])
        family_table.append(
            {
                "condition": condition,
                "family": family,
                "low_cue_field_f1": round(safe_mean([row["field_f1"] for row in member if row["cue_band"] == "low"]), 4),
                "record_exact": round(_rate([row for row in member if row["cue_band"] == "low"], "record_exact"), 4),
                "member_nonmember_lift": round(
                    _rate([row for row in member if row["cue_band"] == "low"], "any_sensitive_match")
                    - _rate([row for row in nonmember if row["cue_band"] == "low"], "any_sensitive_match"),
                    4,
                ),
            }
        )

    return {"cue_table": cue_table, "family_table": family_table, "task_count": len(task_rows)}


def render_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Behavioral Results", ""]
    lines.append("## Extraction By Cue Band")
    lines.append("| Condition | Cue band | Member extraction | Non-member extraction | Lift | 95% CI |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in summary["cue_table"]:
        lines.append(
            f"| {row['condition']} | {row['cue_band']} | {row['member_extraction']:.4f} | "
            f"{row['nonmember_extraction']:.4f} | {row['lift']:.4f} | "
            f"[{row['member_ci95'][0]:.4f}, {row['member_ci95'][1]:.4f}] |"
        )
    lines.append("")
    lines.append("## Extraction By Family")
    lines.append("| Condition | Family | Low-cue field F1 | Record exact | Member-nonmember lift |")
    lines.append("|---|---|---:|---:|---:|")
    for row in summary["family_table"]:
        lines.append(
            f"| {row['condition']} | {row['family']} | {row['low_cue_field_f1']:.4f} | "
            f"{row['record_exact']:.4f} | {row['member_nonmember_lift']:.4f} |"
        )
    lines.append("")
    lines.append(f"- Collapsed evaluation tasks: {summary['task_count']}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate scored generations into report tables.")
    parser.add_argument("--scores", required=True, help="Scores JSONL.")
    parser.add_argument("--out", required=True, help="Markdown report output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = aggregate(read_jsonl(args.scores))
    write_text(args.out, render_markdown(summary))
    write_json(str(args.out).replace(".md", ".json"), summary)
    LOGGER.info("Wrote aggregate report to %s", args.out)


if __name__ == "__main__":
    main()
