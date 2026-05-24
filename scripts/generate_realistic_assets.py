from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as stats
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle

from mcrate.utils.stats import agresti_caffo_diff_ci
from mcrate_plot_style import CONDITION_COLORS
from mcrate_plot_style import CONDITION_MARKERS
from mcrate_plot_style import set_style


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_ROOT = ROOT / "study_runs" / "workshop_realistic_main_c4_100m"
REPORT_DIR = ROOT / "reports"
TABLE_DIR = REPORT_DIR / "tables"
FIG_DIR = REPORT_DIR / "figures"

CONDITIONS = ["c0_clean", "c1_exact_1x", "c2_exact_10x", "c3_fuzzy_5x", "c4_redacted"]
CONDITION_LABELS = {
    "c0_clean": "C0-clean",
    "c1_exact_1x": "C1-exact-1x",
    "c2_exact_10x": "C2-exact-10x",
    "c3_fuzzy_5x": "C3-fuzzy-5x",
    "c4_redacted": "C4-redacted",
}
CUE_BANDS = ["high", "medium", "low", "no_cue"]
BUDGETS = [1, 5, 20]
MAIN_LIFT_CI_COLUMNS = [
    ("high", 1, "High $B{=}1$"),
    ("high", 5, "High $B{=}5$"),
    ("high", 20, "High $B{=}20$"),
    ("low", 20, "Low $B{=}20$"),
    ("no_cue", 20, "No cue $B{=}20$"),
]

MPL_COLORS = CONDITION_COLORS
MARKERS = CONDITION_MARKERS


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return stats.mean(values) if values else 0.0


def sd(values: list[float]) -> float:
    return stats.stdev(values) if len(values) > 1 else 0.0


def sem(values: list[float]) -> float:
    return sd(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def ci95(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    m = mean(values)
    if len(values) == 1:
        return (m, m)
    # t_{0.975, df=2}; all current paper runs use three seeds.
    t_crit = 4.303 if len(values) == 3 else 1.96
    half = t_crit * sem(values)
    return (m - half, m + half)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_reports(run_root: Path) -> list[dict[str, Any]]:
    behavior_root = run_root / "reports" / "behavioral"
    reports: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for path in sorted((behavior_root / condition).glob("seed_*__budget*.json")):
            seed = int(path.name.split("__")[0].replace("seed_", ""))
            budget = int(path.name.split("__budget")[1].replace(".json", ""))
            payload = read_json(path)
            cue = {row["cue_band"]: row for row in payload["cue_table"]}
            logprob = {row["cue_band"]: row for row in payload.get("logprob_table", [])}
            family = {row["family"]: row for row in payload.get("family_table", [])}
            reports.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "budget": budget,
                    "cue": cue,
                    "logprob": logprob,
                    "family": family,
                    "task_count": payload.get("task_count"),
                }
            )
    expected = len(CONDITIONS) * 3 * len(BUDGETS)
    if len(reports) != expected:
        raise RuntimeError(f"Expected {expected} behavioral reports, found {len(reports)} under {behavior_root}")
    return reports


def cue_values(reports: list[dict[str, Any]], *, condition: str, budget: int, cue_band: str, metric: str) -> list[float]:
    return [
        float(report["cue"][cue_band][metric])
        for report in reports
        if report["condition"] == condition and report["budget"] == budget
    ]


def logprob_values(reports: list[dict[str, Any]], *, condition: str, cue_band: str) -> list[float]:
    return [
        float(report["logprob"][cue_band]["delta"])
        for report in reports
        if report["condition"] == condition and report["budget"] == 1 and cue_band in report["logprob"]
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        write_text(path, "")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def high_cue_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for budget in BUDGETS:
        c0 = mean(cue_values(reports, condition="c0_clean", budget=budget, cue_band="high", metric="lift"))
        c4 = mean(cue_values(reports, condition="c4_redacted", budget=budget, cue_band="high", metric="lift"))
        for condition in CONDITIONS:
            lifts = cue_values(reports, condition=condition, budget=budget, cue_band="high", metric="lift")
            member = cue_values(reports, condition=condition, budget=budget, cue_band="high", metric="member_extraction")
            nonmember = cue_values(reports, condition=condition, budget=budget, cue_band="high", metric="nonmember_extraction")
            lo, hi = ci95(lifts)
            rows.append(
                {
                    "condition": condition,
                    "label": CONDITION_LABELS[condition],
                    "budget": budget,
                    "n_seeds": len(lifts),
                    "mean_high_lift": round(mean(lifts), 4),
                    "sd_high_lift": round(sd(lifts), 4),
                    "ci95_low": round(lo, 4),
                    "ci95_high": round(hi, 4),
                    "mean_member_extraction": round(mean(member), 4),
                    "mean_nonmember_extraction": round(mean(nonmember), 4),
                    "delta_vs_c0": round(mean(lifts) - c0, 4),
                    "delta_vs_c4": round(mean(lifts) - c4, 4),
                }
            )
    return rows


def cue_gating_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for condition in CONDITIONS:
        for budget in BUDGETS:
            row: dict[str, Any] = {"condition": condition, "label": CONDITION_LABELS[condition], "budget": budget}
            for cue_band in CUE_BANDS:
                lifts = cue_values(reports, condition=condition, budget=budget, cue_band=cue_band, metric="lift")
                row[f"{cue_band}_lift"] = round(mean(lifts), 4)
            rows.append(row)
    return rows


def seed_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for report in sorted(reports, key=lambda r: (r["condition"], r["seed"], r["budget"])):
        cue = report["cue"]
        rows.append(
            {
                "condition": report["condition"],
                "seed": report["seed"],
                "budget": report["budget"],
                "high_member": cue["high"]["member_extraction"],
                "high_nonmember": cue["high"]["nonmember_extraction"],
                "high_lift": cue["high"]["lift"],
                "medium_lift": cue["medium"]["lift"],
                "low_lift": cue["low"]["lift"],
                "no_cue_lift": cue["no_cue"]["lift"],
            }
        )
    return rows


def logprob_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for condition in CONDITIONS:
        for cue_band in CUE_BANDS:
            values = logprob_values(reports, condition=condition, cue_band=cue_band)
            rows.append(
                {
                    "condition": condition,
                    "label": CONDITION_LABELS[condition],
                    "cue_band": cue_band,
                    "n_seeds": len(values),
                    "mean_logprob_delta": round(mean(values), 4),
                    "sd_logprob_delta": round(sd(values), 4),
                }
            )
    return rows


def family_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for report in reports:
        for family, row in report["family"].items():
            key = (report["condition"], report["budget"], family)
            grouped[key]["low_cue_field_f1"].append(float(row["low_cue_field_f1"]))
            grouped[key]["record_exact"].append(float(row["record_exact"]))
            grouped[key]["member_nonmember_lift"].append(float(row["member_nonmember_lift"]))
    rows = []
    for condition in CONDITIONS:
        for budget in BUDGETS:
            for family in ["account", "event", "identity"]:
                values = grouped[(condition, budget, family)]
                rows.append(
                    {
                        "condition": condition,
                        "label": CONDITION_LABELS[condition],
                        "budget": budget,
                        "family": family,
                        "low_cue_field_f1": round(mean(values["low_cue_field_f1"]), 4),
                        "record_exact": round(mean(values["record_exact"]), 4),
                        "member_nonmember_lift": round(mean(values["member_nonmember_lift"]), 4),
                    }
                )
    return rows


def _fmt4(value: float) -> str:
    if abs(value) < 0.00005:
        value = 0.0
    return f"{value:.4f}"


def main_lift_ci_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pooled Agresti-Caffo CIs for the compact main lift table."""
    rows = []
    for condition in CONDITIONS:
        for cue_band, budget, label in MAIN_LIFT_CI_COLUMNS:
            seed_reports = [
                report
                for report in reports
                if report["condition"] == condition and int(report["budget"]) == budget
            ]
            member_successes = 0
            member_tasks = 0
            nonmember_successes = 0
            nonmember_tasks = 0
            seed_lifts = []
            for report in seed_reports:
                cue_row = report["cue"][cue_band]
                member_successes += int(cue_row["member_successes"])
                member_tasks += int(cue_row["member_tasks"])
                nonmember_successes += int(cue_row["nonmember_successes"])
                nonmember_tasks += int(cue_row["nonmember_tasks"])
                seed_lifts.append(float(cue_row["lift"]))
            pooled_lift = (member_successes / member_tasks) - (nonmember_successes / nonmember_tasks)
            ci_low, ci_high = agresti_caffo_diff_ci(
                member_successes,
                member_tasks,
                nonmember_successes,
                nonmember_tasks,
            )
            rows.append(
                {
                    "condition": condition,
                    "label": CONDITION_LABELS[condition],
                    "cue_band": cue_band,
                    "budget": budget,
                    "column_label": label,
                    "mean_lift": round(mean(seed_lifts), 4),
                    "pooled_lift": round(pooled_lift, 4),
                    "ci95_low": round(ci_low, 4),
                    "ci95_high": round(ci_high, 4),
                    "member_successes": member_successes,
                    "member_tasks": member_tasks,
                    "nonmember_successes": nonmember_successes,
                    "nonmember_tasks": nonmember_tasks,
                }
            )
    return rows


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_latex_tables(
    high_rows: list[dict[str, Any]],
    cue_rows: list[dict[str, Any]],
    lift_ci_rows: list[dict[str, Any]],
) -> None:
    high_lookup = {(row["condition"], int(row["budget"])): row for row in high_rows}
    cue_lookup = {(row["condition"], int(row["budget"])): row for row in cue_rows}

    high_lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Condition & Budget 1 & Budget 5 & Budget 20 \\\\",
        "\\midrule",
    ]
    for condition in CONDITIONS:
        values = [float(high_lookup[(condition, budget)]["mean_high_lift"]) for budget in BUDGETS]
        high_lines.append(
            f"{CONDITION_LABELS[condition]} & {values[0]:.4f} & {values[1]:.4f} & {values[2]:.4f} \\\\"
        )
    high_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )

    delta_lines = [
        "\\begin{tabular}{rrrrr}",
        "\\toprule",
        "Budget & C1 $- C0$ & C2 $- C0$ & C3 $- C0$ & C3 / C2 \\\\",
        "\\midrule",
    ]
    for budget in BUDGETS:
        c1 = float(high_lookup[("c1_exact_1x", budget)]["delta_vs_c0"])
        c2 = float(high_lookup[("c2_exact_10x", budget)]["delta_vs_c0"])
        c3 = float(high_lookup[("c3_fuzzy_5x", budget)]["delta_vs_c0"])
        ratio = c3 / c2 if c2 else 0.0
        delta_lines.append(f"{budget} & {c1:.4f} & {c2:.4f} & {c3:.4f} & {ratio:.2f} \\\\")
    delta_lines.extend(["\\bottomrule", "\\end{tabular}"])

    c3_lines = [
        "\\begin{tabular}{rrrrr}",
        "\\toprule",
        "Budget & High & Medium & Low & No cue \\\\",
        "\\midrule",
    ]
    for budget in BUDGETS:
        row = cue_lookup[("c3_fuzzy_5x", budget)]
        c3_lines.append(
            f"{budget} & {float(row['high_lift']):.4f} & {float(row['medium_lift']):.4f} & "
            f"{float(row['low_lift']):.4f} & {float(row['no_cue_lift']):.4f} \\\\"
        )
    c3_lines.extend(["\\bottomrule", "\\end{tabular}"])

    write_text(TABLE_DIR / "realistic_high_cue_lift.tex", "\n".join(high_lines) + "\n")
    write_text(TABLE_DIR / "realistic_delta_vs_clean.tex", "\n".join(delta_lines) + "\n")
    write_text(TABLE_DIR / "realistic_c3_cue_gating.tex", "\n".join(c3_lines) + "\n")

    ci_lookup = {
        (row["condition"], row["cue_band"], int(row["budget"])): row
        for row in lift_ci_rows
    }
    ci_lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Condition & High $B{=}1$ & High $B{=}5$ & High $B{=}20$ & Low $B{=}20$ & No cue $B{=}20$ \\",
        r"\midrule",
    ]
    for condition in CONDITIONS:
        cells = []
        for cue_band, budget, _ in MAIN_LIFT_CI_COLUMNS:
            row = ci_lookup[(condition, cue_band, budget)]
            cells.append(
                f"${_fmt4(float(row['mean_lift']))}$ "
                f"[${_fmt4(float(row['ci95_low']))}$, ${_fmt4(float(row['ci95_high']))}$]"
            )
        ci_lines.append(f"{CONDITION_LABELS[condition]} & " + " & ".join(cells) + r" \\")
    ci_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\caption{Main member-over-nonmember lift estimates with pooled Agresti--Caffo 95\% confidence intervals for the member--nonmember rate difference.}",
            r"\label{tab:main-lift-summary}",
            r"\end{table}",
        ]
    )
    write_text(TABLE_DIR / "realistic_main_lift_ci.tex", "\n".join(ci_lines) + "\n")


def write_summary_md(
    high_rows: list[dict[str, Any]],
    cue_rows: list[dict[str, Any]],
    log_rows: list[dict[str, Any]],
    lift_ci_rows: list[dict[str, Any]],
    outfile: Path,
) -> None:
    high_lookup = {(row["condition"], int(row["budget"])): row for row in high_rows}
    c3_rows = [high_lookup[("c3_fuzzy_5x", budget)] for budget in BUDGETS]

    high_table_rows = []
    for condition in CONDITIONS:
        row = [CONDITION_LABELS[condition]]
        for budget in BUDGETS:
            item = high_lookup[(condition, budget)]
            row.append(f"{float(item['mean_high_lift']):.4f}")
        high_table_rows.append(row)

    delta_rows = []
    for budget in BUDGETS:
        c1 = high_lookup[("c1_exact_1x", budget)]["delta_vs_c0"]
        c2 = high_lookup[("c2_exact_10x", budget)]["delta_vs_c0"]
        c3 = high_lookup[("c3_fuzzy_5x", budget)]["delta_vs_c0"]
        delta_rows.append(
            [
                str(budget),
                f"{float(c1):.4f}",
                f"{float(c2):.4f}",
                f"{float(c3):.4f}",
                f"{float(c3) / float(c1):.2f}x" if float(c1) else "",
                f"{float(c3) / float(c2):.2f}x" if float(c2) else "",
            ]
        )

    cue_lookup = {(row["condition"], int(row["budget"])): row for row in cue_rows}
    c3_cue_rows = []
    for budget in BUDGETS:
        item = cue_lookup[("c3_fuzzy_5x", budget)]
        c3_cue_rows.append(
            [
                str(budget),
                f"{float(item['high_lift']):.4f}",
                f"{float(item['medium_lift']):.4f}",
                f"{float(item['low_lift']):.4f}",
                f"{float(item['no_cue_lift']):.4f}",
            ]
        )

    log_lookup = {(row["condition"], row["cue_band"]): row for row in log_rows}
    log_table_rows = []
    for condition in CONDITIONS:
        row = [CONDITION_LABELS[condition]]
        for cue_band in CUE_BANDS:
            row.append(f"{float(log_lookup[(condition, cue_band)]['mean_logprob_delta']):.4f}")
        log_table_rows.append(row)

    text = f"""# Realistic Main C4 100M Tables And Figures

Source run: `study_runs/workshop_realistic_main_c4_100m`

All tables and figures in this file are generated only from:

`study_runs/workshop_realistic_main_c4_100m/reports/behavioral/*/*.json`

Generated artifacts:

- `reports/tables/realistic_high_cue_lift.csv`
- `reports/tables/realistic_cue_gating.csv`
- `reports/tables/realistic_seed_high_cue.csv`
- `reports/tables/realistic_logprob_delta.csv`
- `reports/tables/realistic_family_metrics.csv`
- `reports/tables/realistic_main_lift_ci.csv`
- `reports/tables/realistic_high_cue_lift.tex`
- `reports/tables/realistic_delta_vs_clean.tex`
- `reports/tables/realistic_c3_cue_gating.tex`
- `reports/tables/realistic_main_lift_ci.tex`
- `reports/figures/realistic_high_cue_lift.svg`
- `reports/figures/realistic_high_cue_lift.pdf`
- `reports/figures/realistic_cue_gating.svg`
- `reports/figures/realistic_cue_gating.pdf`
- `reports/figures/realistic_cue_gating_heatmap.svg`
- `reports/figures/realistic_cue_gating_heatmap.pdf`
- `reports/figures/realistic_c3_budget_curve.svg`
- `reports/figures/realistic_c3_budget_curve.pdf`
- `reports/figures/realistic_logprob_delta.svg`
- `reports/figures/realistic_logprob_delta.pdf`
- `reports/figures/realistic_high_cue_rates.svg`
- `reports/figures/realistic_high_cue_rates.pdf`
- `reports/figures/realistic_high_cue_lift_and_rates.svg`
- `reports/figures/realistic_high_cue_lift_and_rates.pdf`
- `reports/figures/realistic_c3_delta_vs_controls.svg`
- `reports/figures/realistic_c3_delta_vs_controls.pdf`
- `reports/figures/realistic_c3_delta_and_gating.svg`
- `reports/figures/realistic_c3_delta_and_gating.pdf`

## High-Cue Lift

{markdown_table(["Condition", "budget1", "budget5", "budget20"], high_table_rows)}

## Main Lift Confidence Intervals

The compact main-text CI table is generated as
`reports/tables/realistic_main_lift_ci.tex`. Its intervals are pooled
Agresti-Caffo 95% confidence intervals for the member-nonmember rate difference.

## Delta Versus Clean

{markdown_table(["Budget", "C1 delta", "C2 delta", "C3 delta", "C3/C1", "C3/C2"], delta_rows)}

## C3 Cue Gating

{markdown_table(["Budget", "high", "medium", "low", "no cue"], c3_cue_rows)}

## Teacher-Forced Logprob Delta

{markdown_table(["Condition", "high", "medium", "low", "no cue"], log_table_rows)}

## Paper Notes

- `C3_fuzzy_5x` reproduces across all three seeds and has the largest high-cue lift at budget 1.
- The C3 effect is cue-gated: high-cue extraction is large, while low/no-cue free-generation lift is near zero.
- At budget 20, C3 high-cue lift weakens because nonmember extraction rises, but the teacher-forced logprob delta remains strongly positive.
- `C4_redacted` stays near zero in behavioral lift, which is the clean negative-control story.
"""
    write_text(outfile, text)


def configure_matplotlib() -> None:
    set_style()


def save_matplotlib_figure(fig: Any, outfile: Path, *, tight: bool = True) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(outfile, **save_kwargs)
    fig.savefig(outfile.with_suffix(".pdf"), **save_kwargs)
    plt.close(fig)


def _condition_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=MPL_COLORS[condition],
            marker=MARKERS[condition],
            linewidth=1.8,
            markersize=4,
            label=CONDITION_LABELS[condition],
        )
        for condition in CONDITIONS
    ]


def _cue_axis_with_weak_band(ax: Any) -> None:
    ax.axvspan(1.5, 3.35, color="#F8FAFC", zorder=0)
    ax.axvline(1.5, color="#94A3B8", linestyle=":", linewidth=1.0, zorder=1)
    ax.text(2.45, 0.086, "weak-cue region", color="#64748B", fontsize=7.5, ha="center", va="top")


def _membership_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color="black", linestyle="-", marker="o", linewidth=1.4, markersize=4, label="Member"),
        Line2D([0], [0], color="black", linestyle="--", marker="o", linewidth=1.4, markersize=4, label="Nonmember"),
    ]


def _draw_error_bars(ax: Any, xs: list[float], lows: list[float], highs: list[float], *, cap_width: float = 0.055) -> None:
    ax.vlines(xs, lows, highs, color="black", linewidth=0.8, zorder=3)
    ax.hlines(lows, [x - cap_width / 2 for x in xs], [x + cap_width / 2 for x in xs], color="black", linewidth=0.8, zorder=3)
    ax.hlines(highs, [x - cap_width / 2 for x in xs], [x + cap_width / 2 for x in xs], color="black", linewidth=0.8, zorder=3)


def _plot_high_cue_lift(
    ax: Any,
    rows: list[dict[str, Any]],
    *,
    show_xlabel: bool = True,
    title: str = "High-cue lift",
) -> None:
    row_index = {(row["condition"], int(row["budget"])): row for row in rows}

    for condition in CONDITIONS:
        means = [float(row_index[(condition, budget)]["mean_high_lift"]) for budget in BUDGETS]
        lows = [float(row_index[(condition, budget)]["ci95_low"]) for budget in BUDGETS]
        highs = [float(row_index[(condition, budget)]["ci95_high"]) for budget in BUDGETS]
        ax.fill_between(BUDGETS, lows, highs, color=MPL_COLORS[condition], alpha=0.13, linewidth=0)
        ax.plot(
            BUDGETS,
            means,
            label=CONDITION_LABELS[condition],
            color=MPL_COLORS[condition],
            marker=MARKERS[condition],
            linewidth=1.9,
            markersize=4.2,
        )

    ax.axhline(0, color="#111827", linestyle="-.", linewidth=0.9)
    ax.text(1.05, 0.003, "no member lift", color="#334155", fontsize=7.5, va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS, [str(budget) for budget in BUDGETS])
    if show_xlabel:
        ax.set_xlabel("Generation budget")
    ax.set_ylabel("High-cue extraction lift")
    ax.set_title(title)
    ax.set_ylim(-0.03, 0.11)
    ax.set_xlim(0.85, 24)


def _plot_high_cue_rates(
    ax: Any,
    rows: list[dict[str, Any]],
    *,
    show_xlabel: bool = True,
    title: str = "High-cue rates",
) -> None:
    row_index = {(row["condition"], int(row["budget"])): row for row in rows}
    for condition in CONDITIONS:
        member = [float(row_index[(condition, budget)]["mean_member_extraction"]) for budget in BUDGETS]
        nonmember = [float(row_index[(condition, budget)]["mean_nonmember_extraction"]) for budget in BUDGETS]
        ax.plot(
            BUDGETS,
            member,
            color=MPL_COLORS[condition],
            marker=MARKERS[condition],
            linewidth=1.6,
            markersize=4.0,
        )
        ax.plot(
            BUDGETS,
            nonmember,
            color=MPL_COLORS[condition],
            marker=MARKERS[condition],
            linestyle="--",
            linewidth=1.4,
            markersize=4.0,
            alpha=0.82,
        )
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS, [str(budget) for budget in BUDGETS])
    if show_xlabel:
        ax.set_xlabel("Generation budget")
    ax.set_ylabel("High-cue extraction rate")
    ax.set_title(title)
    ax.set_ylim(0, 0.50)
    ax.axhline(0, color="black", linewidth=0.8)


def write_high_cue_bar_figure(rows: list[dict[str, Any]], outfile: Path) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    _plot_high_cue_lift(ax, rows, title="High-cue extraction lift")
    ax.legend(
        handles=_condition_legend_handles(),
        ncol=2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
    )
    fig.subplots_adjust(bottom=0.30)
    save_matplotlib_figure(fig, outfile, tight=False)


def write_cue_gating_figure(rows: list[dict[str, Any]], outfile: Path) -> None:
    configure_matplotlib()
    row_index = {(row["condition"], int(row["budget"])): row for row in rows}
    x_positions = list(range(len(CUE_BANDS)))
    x_labels = [cue_band.replace("_", " ") for cue_band in CUE_BANDS]

    fig, axes = plt.subplots(1, len(BUDGETS), figsize=(7.2, 3.15), sharey=True)
    for ax, budget in zip(axes, BUDGETS):
        for condition in CONDITIONS:
            row = row_index[(condition, budget)]
            values = [float(row[f"{cue_band}_lift"]) for cue_band in CUE_BANDS]
            ax.plot(
                x_positions,
                values,
                color=MPL_COLORS[condition],
                marker=MARKERS[condition],
                linewidth=1.5,
                markersize=4.0,
                label=CONDITION_LABELS[condition],
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x_positions, x_labels, rotation=25, ha="right")
        ax.set_title(f"Budget {budget}")
        ax.set_ylim(-0.012, 0.092)
        ax.set_xlabel("Cue band")
    axes[0].set_ylabel("Extraction lift")
    fig.suptitle("Cue-gated extraction lift", y=0.98, fontsize=10)
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=MPL_COLORS[condition],
                marker=MARKERS[condition],
                linewidth=1.5,
                markersize=4.0,
                label=CONDITION_LABELS[condition],
            )
            for condition in CONDITIONS
        ],
        ncol=5,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.subplots_adjust(bottom=0.30, top=0.82, wspace=0.24)
    save_matplotlib_figure(fig, outfile, tight=False)


def write_cue_gating_heatmap(rows: list[dict[str, Any]], outfile: Path) -> None:
    configure_matplotlib()
    row_index = {(row["condition"], int(row["budget"])): row for row in rows}
    cells = [(budget, cue_band) for budget in BUDGETS for cue_band in CUE_BANDS]
    x_labels = [f"b{budget}\n{cue_band.replace('_', ' ')}" for budget, cue_band in cells]
    max_positive = 0.09

    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    for y_idx, condition in enumerate(CONDITIONS):
        for x_idx, (budget, cue_band) in enumerate(cells):
            value = float(row_index[(condition, budget)][f"{cue_band}_lift"])
            intensity = max(0.08, min(max(value, 0.0) / max_positive, 1.0))
            alpha = 0.14 + 0.78 * intensity
            rect = Rectangle(
                (x_idx - 0.5, y_idx - 0.5),
                1.0,
                1.0,
                facecolor=to_rgba(MPL_COLORS[condition], alpha),
                edgecolor="white",
                linewidth=1.0,
            )
            ax.add_patch(rect)
            ax.text(
                x_idx,
                y_idx,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if intensity > 0.55 else "black",
            )

    ax.set_xlim(-0.5, len(cells) - 0.5)
    ax.set_ylim(len(CONDITIONS) - 0.5, -0.5)
    ax.set_xticks(range(len(cells)), x_labels)
    ax.set_yticks(range(len(CONDITIONS)), [CONDITION_LABELS[c] for c in CONDITIONS])
    for tick, condition in zip(ax.get_yticklabels(), CONDITIONS):
        tick.set_color(MPL_COLORS[condition])
    ax.set_title("Cue-gated extraction lift")
    ax.set_xlabel("Generation budget and cue band")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save_matplotlib_figure(fig, outfile, tight=False)


def write_c3_budget_curve(rows: list[dict[str, Any]], outfile: Path) -> None:
    configure_matplotlib()
    row_index = {(row["condition"], int(row["budget"])): row for row in rows}
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for condition in ["c1_exact_1x", "c2_exact_10x", "c3_fuzzy_5x"]:
        means = [float(row_index[(condition, budget)]["mean_high_lift"]) for budget in BUDGETS]
        ax.plot(
            BUDGETS,
            means,
            marker="o",
            linewidth=1.8,
            label=CONDITION_LABELS[condition],
            color=MPL_COLORS[condition],
        )
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS, [str(budget) for budget in BUDGETS])
    ax.set_xlabel("Generation budget")
    ax.set_ylabel("High-cue extraction lift")
    ax.set_title("Fuzzy versus exact high-cue extraction")
    ax.set_ylim(0.0, 0.095)
    ax.legend(frameon=False)
    save_matplotlib_figure(fig, outfile, tight=False)


def write_logprob_figure(rows: list[dict[str, Any]], outfile: Path) -> None:
    configure_matplotlib()
    row_index = {(row["condition"], row["cue_band"]): row for row in rows}
    x_positions = list(range(len(CUE_BANDS)))

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.axvspan(1.5, 3.35, color="#F8FAFC", zorder=0)
    ax.axvline(1.5, color="#94A3B8", linestyle=":", linewidth=1.0, zorder=1)
    ax.text(2.45, 0.272, "weak-cue region", color="#64748B", fontsize=7.5, ha="center", va="top")
    for condition in CONDITIONS:
        means = [float(row_index[(condition, cue_band)]["mean_logprob_delta"]) for cue_band in CUE_BANDS]
        ax.plot(
            x_positions,
            means,
            label=CONDITION_LABELS[condition],
            color=MPL_COLORS[condition],
            marker=MARKERS[condition],
            linewidth=1.8,
            markersize=4.2,
        )

    ax.axhline(0, color="#111827", linestyle="-.", linewidth=0.9)
    ax.set_xticks(x_positions, [cue.replace("_", " ") for cue in CUE_BANDS])
    ax.set_xlabel("Cue band")
    ax.set_ylabel("Member - nonmember target logprob")
    ax.set_title("Teacher-forced target logprob delta")
    ax.set_ylim(-0.04, 0.29)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save_matplotlib_figure(fig, outfile)


def write_high_cue_rate_figure(rows: list[dict[str, Any]], outfile: Path) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    _plot_high_cue_rates(ax, rows, title="High-cue extraction rates")
    condition_legend = ax.legend(
        handles=_condition_legend_handles(),
        ncol=2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
    )
    ax.add_artist(condition_legend)
    ax.legend(handles=_membership_legend_handles(), frameon=False, loc="upper left")
    fig.subplots_adjust(bottom=0.30)
    save_matplotlib_figure(fig, outfile, tight=False)


def write_high_cue_joint_figure(rows: list[dict[str, Any]], outfile: Path) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=False)
    _plot_high_cue_lift(axes[0], rows, show_xlabel=True, title="A. Member/nonmember lift")
    _plot_high_cue_rates(axes[1], rows, show_xlabel=True, title="B. Absolute rates")
    axes[1].legend(handles=_membership_legend_handles(), frameon=False, loc="upper left")
    fig.legend(
        handles=_condition_legend_handles(),
        ncol=5,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.subplots_adjust(bottom=0.24, wspace=0.33)
    save_matplotlib_figure(fig, outfile, tight=False)


def _plot_c3_delta_vs_controls(
    ax: Any,
    rows: list[dict[str, Any]],
    *,
    title: str = "C3-fuzzy-5x high-cue lift relative to controls",
    show_legend: bool = True,
) -> None:
    by_key = {(row["condition"], int(row["seed"]), int(row["budget"])): float(row["high_lift"]) for row in rows}
    comparators = ["c0_clean", "c4_redacted", "c1_exact_1x", "c2_exact_10x"]
    comparator_labels = {
        "c0_clean": "C3 - C0",
        "c4_redacted": "C3 - C4",
        "c1_exact_1x": "C3 - C1",
        "c2_exact_10x": "C3 - C2",
    }
    colors = {
        "c0_clean": MPL_COLORS["c0_clean"],
        "c4_redacted": MPL_COLORS["c4_redacted"],
        "c1_exact_1x": MPL_COLORS["c1_exact_1x"],
        "c2_exact_10x": MPL_COLORS["c2_exact_10x"],
    }
    for comp_idx, comparator in enumerate(comparators):
        means = []
        lows = []
        highs = []
        seed_deltas_by_budget = []
        for budget in BUDGETS:
            deltas = [
                by_key[("c3_fuzzy_5x", seed, budget)] - by_key[(comparator, seed, budget)]
                for seed in [1, 2, 3]
            ]
            m = mean(deltas)
            lo, hi = ci95(deltas)
            means.append(m)
            lows.append(lo)
            highs.append(hi)
            seed_deltas_by_budget.append(deltas)
        ax.fill_between(BUDGETS, lows, highs, color=colors[comparator], alpha=0.13, linewidth=0)
        ax.plot(
            BUDGETS,
            means,
            color=colors[comparator],
            marker=MARKERS[comparator],
            linewidth=1.8,
            markersize=4.0,
            label=comparator_labels[comparator],
        )
        for budget_idx, deltas in enumerate(seed_deltas_by_budget):
            for seed_idx, delta in enumerate(deltas):
                ax.scatter(
                    BUDGETS[budget_idx] * [0.94, 1.0, 1.06][seed_idx],
                    delta,
                    s=13,
                    marker="o",
                    facecolor=MPL_COLORS["c3_fuzzy_5x"],
                    edgecolor="white",
                    linewidth=0.5,
                    alpha=0.78,
                    zorder=4,
                )

    ax.axhline(0, color="#111827", linestyle="-.", linewidth=0.9)
    ax.text(1.05, 0.003, "C3 matches control", color="#334155", fontsize=7.5, va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS, [str(budget) for budget in BUDGETS])
    ax.set_xlabel("Generation budget")
    ax.set_ylabel("High-cue lift difference")
    ax.set_title(title)
    ax.set_xlim(0.85, 24)
    ax.set_ylim(-0.045, 0.105)
    if show_legend:
        ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20))


def write_c3_delta_vs_controls_figure(rows: list[dict[str, Any]], outfile: Path) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    _plot_c3_delta_vs_controls(ax, rows)
    save_matplotlib_figure(fig, outfile)


def _plot_c3_only_cue_gating(
    ax: Any,
    rows: list[dict[str, Any]],
    *,
    title: str = "C3 cue gating",
    show_legend: bool = False,
) -> None:
    row_index = {(row["condition"], int(row["budget"])): row for row in rows}
    x_positions = list(range(len(CUE_BANDS)))
    x_labels = [cue_band.replace("_", " ") for cue_band in CUE_BANDS]
    _cue_axis_with_weak_band(ax)
    styles = {
        1: ("-", "o", 1.00, 2.0),
        5: ("--", "s", 0.90, 1.9),
        20: (":", "^", 0.62, 1.6),
    }
    for budget in BUDGETS:
        row = row_index[("c3_fuzzy_5x", budget)]
        values = [float(row[f"{cue_band}_lift"]) for cue_band in CUE_BANDS]
        linestyle, marker, alpha, linewidth = styles[budget]
        ax.plot(
            x_positions,
            values,
            color=MPL_COLORS["c3_fuzzy_5x"],
            linestyle=linestyle,
            marker=marker,
            linewidth=linewidth,
            markersize=4.2,
            alpha=alpha,
            label=f"budget {budget}",
        )
        ax.text(
            x_positions[0] + 0.10,
            values[0],
            f"b{budget}",
            color=MPL_COLORS["c3_fuzzy_5x"],
            fontsize=8.5,
            va="center",
            ha="left",
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.7},
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x_positions, x_labels, rotation=20, ha="right")
    ax.set_xlabel("Cue band")
    ax.set_ylabel("Extraction lift")
    ax.set_ylim(-0.005, 0.09)
    ax.set_xlim(-0.08, len(CUE_BANDS) - 0.78)
    ax.set_title(title)
    if show_legend:
        ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20))


def write_c3_delta_and_gating_figure(seed_rows: list[dict[str, Any]], cue_rows: list[dict[str, Any]], outfile: Path) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=False)
    _plot_c3_delta_vs_controls(axes[0], seed_rows, title="A. C3 relative to controls", show_legend=True)
    _plot_c3_only_cue_gating(axes[1], cue_rows, title="B. C3 cue gating", show_legend=False)
    fig.subplots_adjust(bottom=0.30, wspace=0.34)
    save_matplotlib_figure(fig, outfile, tight=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-facing assets for the realistic C4 100M run.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()

    reports = load_reports(args.run_root)
    high = high_cue_rows(reports)
    cue = cue_gating_rows(reports)
    seeds = seed_rows(reports)
    logprob = logprob_rows(reports)
    family = family_rows(reports)
    lift_ci = main_lift_ci_rows(reports)

    write_csv(TABLE_DIR / "realistic_high_cue_lift.csv", high)
    write_csv(TABLE_DIR / "realistic_cue_gating.csv", cue)
    write_csv(TABLE_DIR / "realistic_seed_high_cue.csv", seeds)
    write_csv(TABLE_DIR / "realistic_logprob_delta.csv", logprob)
    write_csv(TABLE_DIR / "realistic_family_metrics.csv", family)
    write_csv(TABLE_DIR / "realistic_main_lift_ci.csv", lift_ci)

    write_high_cue_bar_figure(high, FIG_DIR / "realistic_high_cue_lift.svg")
    write_cue_gating_figure(cue, FIG_DIR / "realistic_cue_gating.svg")
    write_cue_gating_heatmap(cue, FIG_DIR / "realistic_cue_gating_heatmap.svg")
    write_c3_budget_curve(high, FIG_DIR / "realistic_c3_budget_curve.svg")
    write_logprob_figure(logprob, FIG_DIR / "realistic_logprob_delta.svg")
    write_high_cue_rate_figure(high, FIG_DIR / "realistic_high_cue_rates.svg")
    write_high_cue_joint_figure(high, FIG_DIR / "realistic_high_cue_lift_and_rates.svg")
    write_c3_delta_vs_controls_figure(seeds, FIG_DIR / "realistic_c3_delta_vs_controls.svg")
    write_c3_delta_and_gating_figure(seeds, cue, FIG_DIR / "realistic_c3_delta_and_gating.svg")
    write_latex_tables(high, cue, lift_ci)
    write_summary_md(high, cue, logprob, lift_ci, REPORT_DIR / "realistic_main_c4_100m_summary.md")

    print("Generated realistic-run tables and figures.")
    print(f"Summary: {REPORT_DIR / 'realistic_main_c4_100m_summary.md'}")


if __name__ == "__main__":
    main()
