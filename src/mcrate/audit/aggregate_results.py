from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

from mcrate.utils.io import read_jsonl, write_json, write_text
from mcrate.utils.logging import get_logger
from mcrate.utils.stats import agresti_caffo_diff_ci, safe_mean, wilson_ci


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
                "field_f1": max((float(row["field_f1"]) for row in rows), default=0.0),
                "max_target_logprob": max(
                    (float(row["target_logprob"]) for row in rows if row.get("target_logprob") is not None),
                    default=None,
                ),
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
    logprob_table = []
    keys = sorted({(row["condition"], row["cue_band"]) for row in task_rows})
    for condition, cue_band in keys:
        member = by_condition_cue.get((condition, cue_band, "member"), [])
        nonmember = by_condition_cue.get((condition, cue_band, "nonmember"), [])
        member_successes = sum(1 for row in member if row["any_sensitive_match"])
        nonmember_successes = sum(1 for row in nonmember if row["any_sensitive_match"])
        member_rate = safe_mean([1.0 if row["any_sensitive_match"] else 0.0 for row in member])
        nonmember_rate = safe_mean([1.0 if row["any_sensitive_match"] else 0.0 for row in nonmember])
        lift = member_rate - nonmember_rate
        member_ci_low, member_ci_high = wilson_ci(member_successes, len(member))
        nonmember_ci_low, nonmember_ci_high = wilson_ci(nonmember_successes, len(nonmember))
        lift_ci_low, lift_ci_high = agresti_caffo_diff_ci(
            member_successes,
            len(member),
            nonmember_successes,
            len(nonmember),
        )
        cue_table.append(
            {
                "condition": condition,
                "cue_band": cue_band,
                "member_extraction": round(member_rate, 4),
                "nonmember_extraction": round(nonmember_rate, 4),
                "lift": round(lift, 4),
                "member_successes": member_successes,
                "member_tasks": len(member),
                "nonmember_successes": nonmember_successes,
                "nonmember_tasks": len(nonmember),
                "member_ci95": [round(member_ci_low, 4), round(member_ci_high, 4)],
                "nonmember_ci95": [round(nonmember_ci_low, 4), round(nonmember_ci_high, 4)],
                "lift_ci95": [round(lift_ci_low, 4), round(lift_ci_high, 4)],
            }
        )
        member_logprobs = [float(row["max_target_logprob"]) for row in member if row.get("max_target_logprob") is not None]
        nonmember_logprobs = [float(row["max_target_logprob"]) for row in nonmember if row.get("max_target_logprob") is not None]
        logprob_table.append(
            {
                "condition": condition,
                "cue_band": cue_band,
                "member_mean_max_target_logprob": round(safe_mean(member_logprobs), 4),
                "nonmember_mean_max_target_logprob": round(safe_mean(nonmember_logprobs), 4),
                "delta": round(safe_mean(member_logprobs) - safe_mean(nonmember_logprobs), 4),
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

    return {"cue_table": cue_table, "family_table": family_table, "logprob_table": logprob_table, "task_count": len(task_rows)}


def render_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Behavioral Results", ""]
    lines.append("## Extraction By Cue Band")
    lines.append("| Condition | Cue band | Member extraction | Non-member extraction | Lift | Member 95% CI | Lift 95% CI |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in summary["cue_table"]:
        lines.append(
            f"| {row['condition']} | {row['cue_band']} | {row['member_extraction']:.4f} | "
            f"{row['nonmember_extraction']:.4f} | {row['lift']:.4f} | "
            f"[{row['member_ci95'][0]:.4f}, {row['member_ci95'][1]:.4f}] | "
            f"[{row['lift_ci95'][0]:.4f}, {row['lift_ci95'][1]:.4f}] |"
        )
    lines.append("")
    lines.append("## Teacher-Forced Target Logprob By Cue Band")
    lines.append("| Condition | Cue band | Member mean max target logprob | Non-member mean max target logprob | Delta |")
    lines.append("|---|---|---:|---:|---:|")
    for row in summary.get("logprob_table", []):
        lines.append(
            f"| {row['condition']} | {row['cue_band']} | {row['member_mean_max_target_logprob']:.4f} | "
            f"{row['nonmember_mean_max_target_logprob']:.4f} | {row['delta']:.4f} |"
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
