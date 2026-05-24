from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from mcrate_plot_style import CONDITION_COLORS
from mcrate_plot_style import CONDITION_LABELS
from mcrate_plot_style import CONDITION_MARKERS
from mcrate_plot_style import CONDITIONS
from mcrate_plot_style import set_style


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
PUBLIC_DIR = FIG_DIR / "figs_publi"
TABLE_DIR = REPORT_DIR / "tables"

CUE_BANDS = ["high", "medium", "low", "no_cue"]
CUE_LABELS = {"high": "high", "medium": "medium", "low": "low", "no_cue": "no cue"}
BUDGETS = [1, 5, 20]

COLORS = {
    **CONDITION_COLORS,
    "random": "#111827",
    "member": "#2563EB",
    "nonmember": "#64748B",
}
MARKERS = CONDITION_MARKERS
PUBLIC_STEMS = {
    "main_audit_design_schematic": "fig01_audit_design_schematic",
    "main_raw_vs_lift_highcue_b20": "fig02_raw_vs_lift_highcue_b20",
    "main_cue_condition_heatmap_b20": "fig03_cue_condition_heatmap_b20",
    "main_highcue_budget_curves": "fig04_highcue_budget_curves",
    "main_realistic_provenance_recall_at_k": "fig05_realistic_provenance_recall_at_k",
    "main_story_summary_three_panel": "fig06_story_summary_three_panel",
    "main_canary_provenance_recall_at_k": "fig05_canary_provenance_recall_at_k",
    "main_cue_gating_and_provenance": "fig03_05_cue_gating_and_provenance",
    "appendix_teacher_forced_logprob_delta": "figA1_teacher_forced_logprob_delta",
    "appendix_lowcue_family_f1_heatmap": "figA2_lowcue_family_f1_heatmap",
    "appendix_c2_removal_validation": "figA3_c2_removal_validation",
    "appendix_seed_highcue_lift_dotplots": "figA4_seed_highcue_lift_dotplots",
}


def configure_matplotlib() -> None:
    set_style()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig: plt.Figure, stem: str, *, tight: bool = True) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"bbox_inches": "tight"} if tight else {}
    for suffix in [".svg", ".pdf"]:
        fig.savefig(FIG_DIR / f"{stem}{suffix}", **save_kwargs)
        fig.savefig(PUBLIC_DIR / f"{PUBLIC_STEMS.get(stem, stem)}{suffix}", **save_kwargs)
    plt.close(fig)


def condition_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=COLORS[condition],
            marker=MARKERS[condition],
            linewidth=1.8,
            markersize=4.2,
            label=CONDITION_LABELS[condition],
        )
        for condition in CONDITIONS
    ]


def compact_condition_labels() -> list[str]:
    return [CONDITION_LABELS[condition].replace("-", "-\n", 1) for condition in CONDITIONS]


def high_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, str]]:
    return {(row["condition"], int(row["budget"])): row for row in rows}


def cue_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, str]]:
    return {(row["condition"], int(row["budget"])): row for row in rows}


def logprob_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["condition"], row["cue_band"]): row for row in rows}


def lift_ci_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], dict[str, str]]:
    return {(row["condition"], row["cue_band"], int(row["budget"])): row for row in rows}


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def se(values: list[float]) -> float:
    return float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def grouped_seed_values(seed_rows: list[dict[str, str]], metric: str, *, budget: int) -> dict[str, list[float]]:
    grouped = {condition: [] for condition in CONDITIONS}
    for row in seed_rows:
        if int(row["budget"]) == budget:
            grouped[row["condition"]].append(float(row[metric]))
    return grouped


def box(ax: Any, xy: tuple[float, float], text: str, *, width: float = 1.42, height: float = 0.54, fc: str = "#FFFFFF", ec: str = "#CBD5E1") -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.035,rounding_size=0.035",
        linewidth=0.9,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=8.1, color="#0F172A")


def arrow(ax: Any, start: tuple[float, float], end: tuple[float, float], *, color: str = "#64748B") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.95,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def main_audit_design_schematic() -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(9.6, 3.0), constrained_layout=True)
    ax.set_xlim(0, 13.9)
    ax.set_ylim(0, 3.1)
    ax.axis("off")

    box(ax, (0.25, 1.25), "Synthetic\nrecords", width=1.10, height=0.62, fc="#f7f7f7")
    box(ax, (1.85, 2.15), "Member\nrecord", width=1.15, height=0.58, fc="#ffffff", ec=COLORS["c3_fuzzy_5x"])
    box(ax, (1.85, 0.45), "Matched\nnonmember", width=1.15, height=0.58, fc="#ffffff")
    box(ax, (3.50, 2.10), "Inserted into\ntraining", width=1.25, height=0.68, fc="#ffffff", ec=COLORS["c3_fuzzy_5x"])
    box(ax, (3.50, 0.40), "Held out\nof training", width=1.25, height=0.68, fc="#ffffff")
    box(ax, (5.25, 1.25), "Model\nC0--C4", width=0.95, height=0.68, fc="#f7f7f7")
    box(ax, (6.70, 1.25), "Prompt\nmodel", width=0.95, height=0.68, fc="#ffffff")
    box(ax, (8.05, 1.25), "Same cue\nband", width=0.98, height=0.68, fc="#ffffff")
    box(ax, (9.42, 1.25), "Same budget\nB", width=0.98, height=0.68, fc="#ffffff")
    box(ax, (10.78, 1.25), "Extraction\nmatch rule", width=1.16, height=0.68, fc="#ffffff")
    box(ax, (12.48, 1.25), "Score lift\n$\\Delta_{\\mathrm{mem}}$", width=0.98, height=0.68, fc="#f7f7f7")

    arrow(ax, (1.35, 1.56), (1.85, 2.44))
    arrow(ax, (1.35, 1.56), (1.85, 0.74))
    arrow(ax, (3.00, 2.44), (3.50, 2.44), color=COLORS["c3_fuzzy_5x"])
    arrow(ax, (3.00, 0.74), (3.50, 0.74))
    arrow(ax, (4.75, 2.44), (5.25, 1.77), color=COLORS["c3_fuzzy_5x"])
    arrow(ax, (4.75, 0.74), (5.25, 1.41))
    arrow(ax, (6.20, 1.59), (6.70, 1.59))
    arrow(ax, (7.65, 1.59), (8.05, 1.59))
    arrow(ax, (9.03, 1.59), (9.42, 1.59))
    arrow(ax, (10.40, 1.59), (10.78, 1.59))
    arrow(ax, (11.94, 1.59), (12.48, 1.59))

    ax.text(4.12, 2.93, "membership differs", ha="center", va="center", fontsize=7)
    ax.text(9.35, 0.92, "same audit protocol", ha="center", fontsize=7)
    save_figure(fig, "main_audit_design_schematic")


def main_raw_vs_lift_highcue_b20(seed_rows: list[dict[str, str]], lift_ci_rows: list[dict[str, str]]) -> None:
    configure_matplotlib()
    budget = 20
    member = grouped_seed_values(seed_rows, "high_member", budget=budget)
    nonmember = grouped_seed_values(seed_rows, "high_nonmember", budget=budget)
    lift = grouped_seed_values(seed_rows, "high_lift", budget=budget)
    ci = lift_ci_lookup(lift_ci_rows)
    x = np.arange(len(CONDITIONS))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)

    ax = axes[0]
    member_means = [mean(member[c]) for c in CONDITIONS]
    non_means = [mean(nonmember[c]) for c in CONDITIONS]
    ax.bar(x - width / 2, member_means, width, color=[COLORS[c] for c in CONDITIONS], label="Member")
    ax.bar(
        x + width / 2,
        non_means,
        width,
        color="white",
        edgecolor=[COLORS[c] for c in CONDITIONS],
        linewidth=1.2,
        label="Nonmember",
    )
    ax.set_ylabel("Extraction rate")
    ax.set_title("A. Raw high-cue rates")
    ax.set_xticks(x, compact_condition_labels())
    ax.set_ylim(0, 0.50)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    lift_means = [mean(lift[c]) for c in CONDITIONS]
    ci_low = [float(ci[(c, "high", budget)]["ci95_low"]) for c in CONDITIONS]
    ci_high = [float(ci[(c, "high", budget)]["ci95_high"]) for c in CONDITIONS]
    lift_err = np.array(
        [
            [lift_means[idx] - ci_low[idx] for idx in range(len(CONDITIONS))],
            [ci_high[idx] - lift_means[idx] for idx in range(len(CONDITIONS))],
        ]
    )
    ax.bar(
        x,
        lift_means,
        yerr=lift_err,
        capsize=2.8,
        error_kw={"elinewidth": 0.9, "ecolor": "#111827"},
        color=[COLORS[c] for c in CONDITIONS],
        edgecolor="white",
        linewidth=0.8,
    )
    ax.axhline(0, color="#111827", linestyle="-.", linewidth=0.9)
    ax.set_ylabel(r"Member-over-nonmember lift, $\Delta_{\mathrm{mem}}$")
    ax.set_title("B. Exposure-specific lift")
    ax.set_xticks(x, compact_condition_labels())
    ax.set_ylim(-0.025, 0.075)

    save_figure(fig, "main_raw_vs_lift_highcue_b20")


def main_cue_condition_heatmap_b20(cue_rows: list[dict[str, str]]) -> None:
    configure_matplotlib()
    lookup = cue_lookup(cue_rows)
    budget = 20
    values = np.array(
        [[float(lookup[(condition, budget)][f"{cue}_lift"]) for condition in CONDITIONS] for cue in CUE_BANDS]
    )
    vmax = 0.055
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    image = ax.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
    for row_idx, cue in enumerate(CUE_BANDS):
        for col_idx, condition in enumerate(CONDITIONS):
            value = values[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=7.6,
                color="white" if abs(value) > 0.030 else "#0F172A",
            )
    ax.axhline(1.5, color="#64748B", linestyle=":", linewidth=1.0)
    ax.set_xticks(range(len(CONDITIONS)), [CONDITION_LABELS[c] for c in CONDITIONS], rotation=28, ha="right")
    ax.set_yticks(range(len(CUE_BANDS)), [CUE_LABELS[c] for c in CUE_BANDS])
    ax.set_xlabel("Exposure condition")
    ax.set_ylabel("Cue band")
    ax.set_title(r"Cue-gated member-over-nonmember lift at $B=20$")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_ticks([-0.05, -0.025, 0.0, 0.025, 0.05])
    cbar.set_label(r"$\Delta_{\mathrm{mem}}$", fontsize=8)
    fig.tight_layout(pad=0.8)
    save_figure(fig, "main_cue_condition_heatmap_b20", tight=False)


def main_highcue_budget_curves(seed_rows: list[dict[str, str]], lift_ci_rows: list[dict[str, str]]) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(5.25, 3.15))
    x = np.arange(len(BUDGETS))
    ci = lift_ci_lookup(lift_ci_rows)
    for condition in CONDITIONS:
        means = []
        lows = []
        highs = []
        for budget in BUDGETS:
            vals = [
                float(row["high_lift"])
                for row in seed_rows
                if row["condition"] == condition and int(row["budget"]) == budget
            ]
            m = mean(vals)
            means.append(m)
            lows.append(float(ci[(condition, "high", budget)]["ci95_low"]))
            highs.append(float(ci[(condition, "high", budget)]["ci95_high"]))
        ax.fill_between(x, lows, highs, color=COLORS[condition], alpha=0.13, linewidth=0)
        ax.plot(
            x,
            means,
            color=COLORS[condition],
            marker=MARKERS[condition],
            linewidth=2.0,
            markersize=5.0,
            label=CONDITION_LABELS[condition],
        )
    ax.axhline(0, color="#111827", linestyle="-.", linewidth=0.9)
    ax.set_xticks(x, [str(b) for b in BUDGETS])
    ax.set_xlabel(r"Generation budget $B$")
    ax.set_ylabel(r"High-cue lift, $\Delta_{\mathrm{mem}}$")
    ax.set_title("High-cue lift under repeated sampling")
    ax.set_ylim(-0.025, 0.105)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncols=3)
    fig.subplots_adjust(bottom=0.30)
    save_figure(fig, "main_highcue_budget_curves")


def main_story_summary_three_panel(
    seed_rows: list[dict[str, str]],
    cue_rows: list[dict[str, str]],
    lift_ci_rows: list[dict[str, str]],
) -> None:
    """One-glance summary: raw cue confounding, lift, and weak-cue null."""
    configure_matplotlib()
    budget = 20
    ci = lift_ci_lookup(lift_ci_rows)
    cue = cue_lookup(cue_rows)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.6, 3.05),
        gridspec_kw={"width_ratios": [0.84, 1.32, 1.16], "wspace": 0.44},
    )

    ax = axes[0]
    c0_member = mean(
        [
            float(row["high_member"])
            for row in seed_rows
            if row["condition"] == "c0_clean" and int(row["budget"]) == budget
        ]
    )
    c0_nonmember = mean(
        [
            float(row["high_nonmember"])
            for row in seed_rows
            if row["condition"] == "c0_clean" and int(row["budget"]) == budget
        ]
    )
    bars = ax.bar(
        [0, 1],
        [c0_member, c0_nonmember],
        color=[COLORS["member"], "white"],
        edgecolor=[COLORS["member"], COLORS["nonmember"]],
        linewidth=[0.8, 1.2],
        width=0.56,
    )
    for bar, value in zip(bars, [c0_member, c0_nonmember]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{100 * value:.1f}%", ha="center", fontsize=8.0)
    ax.set_xticks([0, 1], ["member", "nonmember"], rotation=18, ha="right")
    ax.set_ylabel("High-cue extraction rate")
    ax.set_title("A. Raw high-cue can be high")
    ax.set_ylim(0, 0.48)
    ax.text(0.5, 0.06, "C0-clean\n(no target exposure)", ha="center", va="bottom", fontsize=8.0, color="#334155")

    ax = axes[1]
    x = np.arange(len(CONDITIONS))
    lift_means = [float(cue[(condition, budget)]["high_lift"]) for condition in CONDITIONS]
    ci_low = [float(ci[(condition, "high", budget)]["ci95_low"]) for condition in CONDITIONS]
    ci_high = [float(ci[(condition, "high", budget)]["ci95_high"]) for condition in CONDITIONS]
    lift_err = np.array(
        [
            [lift_means[idx] - ci_low[idx] for idx in range(len(CONDITIONS))],
            [ci_high[idx] - lift_means[idx] for idx in range(len(CONDITIONS))],
        ]
    )
    ax.bar(
        x,
        lift_means,
        yerr=lift_err,
        capsize=2.6,
        error_kw={"elinewidth": 0.9, "ecolor": "#111827"},
        color=[COLORS[c] for c in CONDITIONS],
        edgecolor="white",
        linewidth=0.8,
    )
    ax.axhline(0, color="#111827", linestyle="-.", linewidth=0.9)
    ax.set_xticks(x, compact_condition_labels())
    ax.set_ylabel(r"High-cue lift, $\Delta_{\mathrm{mem}}$")
    ax.set_title("B. Lift isolates exposure")
    ax.set_ylim(-0.025, 0.075)

    ax = axes[2]
    offsets = np.linspace(-0.24, 0.24, len(CONDITIONS))
    for idx, condition in enumerate(CONDITIONS):
        values = [
            float(cue[(condition, budget)]["low_lift"]),
            float(cue[(condition, budget)]["no_cue_lift"]),
        ]
        xs = np.array([0, 1]) + offsets[idx]
        ax.plot(xs, values, color=COLORS[condition], marker=MARKERS[condition], linewidth=1.4, markersize=4.2)
    ax.axhline(0, color="#111827", linestyle="-.", linewidth=0.9)
    ax.axhspan(-0.001, 0.001, color="#E2E8F0", alpha=0.45, linewidth=0)
    ax.set_xticks([0, 1], ["low cue", "no cue"])
    ax.set_ylabel(r"Lift at $B=20$")
    ax.set_title("C. Weak-cue lift near zero")
    ax.set_ylim(-0.0013, 0.0019)
    ax.text(0.5, 0.00155, "all values <= 0.001", ha="center", va="center", fontsize=7.8, color="#334155")

    fig.legend(
        handles=condition_handles(),
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncols=5,
        fontsize=7.0,
    )
    fig.subplots_adjust(bottom=0.28, left=0.075, right=0.985, top=0.86)
    save_figure(fig, "main_story_summary_three_panel", tight=False)


def provenance_recall_plot(
    *,
    table_path: Path,
    stem: str,
    title: str,
    legend_labels: tuple[str, str],
    full_rank_path: Path | None = None,
) -> None:
    """Use full recall@k if ranks exist; otherwise plot the available recall cutoffs."""
    configure_matplotlib()

    def _optional_random_values(rows: list[dict[str, str]]) -> list[float] | None:
        raw = rows[0].get("random_top10", "").strip()
        if not raw or raw.lower() in {"na", "nan", "none", "pending"}:
            return None
        random_top10 = float(raw)
        n_candidates = 10 / random_top10
        return [1 / n_candidates, random_top10]

    if full_rank_path and full_rank_path.exists():
        rows = read_csv(full_rank_path)
        ks = np.arange(1, 51)
        fig, ax = plt.subplots(figsize=(6.6, 2.9))
        for label, color in [(legend_labels[0], COLORS["c2_exact_10x"]), (legend_labels[1], COLORS["c3_fuzzy_5x"])]:
            ranks = np.array([int(row["true_source_rank"]) for row in rows if row["condition"] == label])
            recall = np.array([(ranks <= k).mean() for k in ks])
            ax.plot(ks, recall, color=color, linewidth=1.9, label=label)
        n_candidates = 526
        ax.plot(ks, np.minimum(ks / n_candidates, 1.0), color=COLORS["random"], linestyle="--", label="Random")
        ax.axvline(10, color="#64748B", linestyle=":", linewidth=1.0)
        ax.set_xlabel(r"Rank cutoff $k$")
        ax.set_ylabel("Recall@k")
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False)
        fig.tight_layout(pad=0.8)
        save_figure(fig, stem, tight=False)
        return

    rows = read_csv(table_path)
    cutoffs = ["Top-1", "Top-10"]
    x = np.arange(len(cutoffs))
    width = 0.24
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    series = [
        (rows[0], COLORS["c2_exact_10x"], -width, legend_labels[0]),
        (rows[1], COLORS["c3_fuzzy_5x"], 0.0, legend_labels[1]),
    ]
    for row, color, offset, label in series:
        targets = int(row["targets"])
        values = [float(row["top1_recall"]), float(row["top10_recall"])]
        counts = [round(values[0] * targets), round(values[1] * targets)]
        bars = ax.bar(x + offset, values, width, color=color, edgecolor="white", linewidth=0.8, label=label)
        for bar, count, target in zip(bars, counts, [targets, targets]):
            height = bar.get_height()
            high_bar = height > 0.86
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height - 0.045 if high_bar else height + 0.025,
                f"{count}/{target}",
                ha="center",
                va="top" if high_bar else "bottom",
                fontsize=7.5,
                color="white" if high_bar else "#334155",
            )
    random_values = _optional_random_values(rows)
    if random_values is not None:
        ax.bar(x + width, random_values, width, color="white", edgecolor=COLORS["random"], linewidth=1.2, hatch="//", label="Random")
    ax.set_xticks(x, cutoffs)
    ax.set_xlabel("Source-rank cutoff")
    ax.set_ylabel("Source recovery recall")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(pad=0.8)
    save_figure(fig, stem, tight=False)


def main_realistic_provenance_recall() -> None:
    provenance_recall_plot(
        table_path=TABLE_DIR / "realistic_provenance.csv",
        stem="main_realistic_provenance_recall_at_k",
        title="Realistic-record source recovery",
        legend_labels=("C2 record", "C3 cluster"),
    )


def main_canary_provenance_recall() -> None:
    provenance_recall_plot(
        table_path=TABLE_DIR / "canary_provenance.csv",
        stem="main_canary_provenance_recall_at_k",
        title="Canary provenance source recovery",
        legend_labels=("C2-exact-10x", "C3-fuzzy-5x"),
        full_rank_path=TABLE_DIR / "canary_provenance_ranks.csv",
    )


def main_cue_gating_and_provenance(cue_rows: list[dict[str, str]]) -> None:
    """Combined native two-panel figure for easier manuscript integration."""
    configure_matplotlib()

    def _optional_random_values(rows: list[dict[str, str]]) -> list[float] | None:
        raw = rows[0].get("random_top10", "").strip()
        if not raw or raw.lower() in {"na", "nan", "none", "pending"}:
            return None
        random_top10 = float(raw)
        n_candidates = 10 / random_top10
        return [1 / n_candidates, random_top10]

    lookup = cue_lookup(cue_rows)
    budget = 20
    heat_values = np.array(
        [[float(lookup[(condition, budget)][f"{cue}_lift"]) for condition in CONDITIONS] for cue in CUE_BANDS]
    )

    prov_rows = read_csv(TABLE_DIR / "realistic_provenance.csv")
    cutoffs = ["Top-1", "Top-10"]
    bar_x = np.arange(len(cutoffs))
    width = 0.24

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.0),
        gridspec_kw={"width_ratios": [1.28, 1.0], "wspace": 0.36},
    )

    ax = axes[0]
    norm = TwoSlopeNorm(vmin=-0.055, vcenter=0.0, vmax=0.055)
    image = ax.imshow(heat_values, cmap="RdBu_r", norm=norm, aspect="auto")
    for row_idx, cue in enumerate(CUE_BANDS):
        for col_idx, condition in enumerate(CONDITIONS):
            value = heat_values[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=6.6,
                color="white" if abs(value) > 0.030 else "#0F172A",
            )
    ax.axhline(1.5, color="#64748B", linestyle=":", linewidth=0.9)
    ax.set_xticks(range(len(CONDITIONS)), [f"C{idx}" for idx in range(len(CONDITIONS))], rotation=0)
    ax.set_yticks(range(len(CUE_BANDS)), [CUE_LABELS[c] for c in CUE_BANDS])
    ax.set_xlabel("Exposure condition")
    ax.set_ylabel("Cue band")
    ax.set_title(r"A. Cue-gated lift at $B=20$")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
    cbar.set_ticks([-0.05, 0.0, 0.05])
    cbar.set_label(r"$\Delta_{\mathrm{mem}}$", fontsize=7)

    ax = axes[1]
    series = [
        (prov_rows[0], COLORS["c2_exact_10x"], -width, "C2 record"),
        (prov_rows[1], COLORS["c3_fuzzy_5x"], 0.0, "C3 cluster"),
    ]
    for row, color, offset, label in series:
        targets = int(row["targets"])
        values = [float(row["top1_recall"]), float(row["top10_recall"])]
        counts = [round(values[0] * targets), round(values[1] * targets)]
        bars = ax.bar(bar_x + offset, values, width, color=color, edgecolor="white", linewidth=0.8, label=label)
        for bar, count, target in zip(bars, counts, [targets, targets]):
            height = bar.get_height()
            high_bar = height > 0.86
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height - 0.045 if high_bar else height + 0.025,
                f"{count}/{target}",
                ha="center",
                va="top" if high_bar else "bottom",
                fontsize=6.8,
                color="white" if high_bar else "#334155",
            )
    random_values = _optional_random_values(prov_rows)
    if random_values is not None:
        ax.bar(bar_x + width, random_values, width, color="white", edgecolor=COLORS["random"], linewidth=1.0, hatch="//", label="Random")
    ax.set_xticks(bar_x, cutoffs)
    ax.set_xlabel("Source-rank cutoff")
    ax.set_ylabel("Source recovery recall")
    ax.set_title("B. Realistic source recovery")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right", fontsize=6.8)

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.88, wspace=0.38)
    save_figure(fig, "main_cue_gating_and_provenance", tight=False)


def appendix_teacher_forced_logprob(log_rows: list[dict[str, str]]) -> None:
    configure_matplotlib()
    lookup = logprob_lookup(log_rows)
    x = np.arange(len(CUE_BANDS))
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    for condition in CONDITIONS:
        values = [float(lookup[(condition, cue)]["mean_logprob_delta"]) for cue in CUE_BANDS]
        ax.plot(x, values, color=COLORS[condition], marker=MARKERS[condition], linewidth=1.8, markersize=4, label=CONDITION_LABELS[condition])
    ax.axhline(0, color="#111827", linestyle="-.", linewidth=0.9)
    ax.set_xticks(x, [CUE_LABELS[c] for c in CUE_BANDS])
    ax.set_xlabel("Cue band")
    ax.set_ylabel("Member minus nonmember target log probability")
    ax.set_title("Teacher-forced target log-probability deltas")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=3)
    fig.subplots_adjust(bottom=0.30)
    save_figure(fig, "appendix_teacher_forced_logprob_delta")


def appendix_lowcue_family_f1(family_rows: list[dict[str, str]]) -> None:
    configure_matplotlib()
    families = ["identity", "account", "event"]
    values = np.array(
        [
            [
                mean(
                    [
                        float(row["low_cue_field_f1"])
                        for row in family_rows
                        if row["condition"] == condition and row["family"] == family and int(row["budget"]) == 20
                    ]
                )
                for family in families
            ]
            for condition in CONDITIONS
        ]
    )
    fig, ax = plt.subplots(figsize=(4.9, 2.95))
    image = ax.imshow(values, cmap="Blues", aspect="auto", vmin=0, vmax=max(float(values.max()), 0.0011))
    for i, condition in enumerate(CONDITIONS):
        for j, family in enumerate(families):
            ax.text(j, i, f"{values[i, j]:.4f}", ha="center", va="center", fontsize=7.5, color="#0F172A")
    ax.set_xticks(range(len(families)), families)
    ax.set_yticks(range(len(CONDITIONS)), [CONDITION_LABELS[c] for c in CONDITIONS])
    ax.set_title("Low-cue field F1 at B=20 (all values <= 0.0011)")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Field F1", fontsize=8)
    save_figure(fig, "appendix_lowcue_family_f1_heatmap")


def appendix_c2_removal_validation(removal_rows: list[dict[str, str]]) -> None:
    configure_matplotlib()
    aggregate = [row for row in removal_rows if row["view"] == "aggregate"]
    targeted = [row for row in removal_rows if row["view"] == "targeted"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.95), constrained_layout=True)

    ax = axes[0]
    for variant, color, marker, label in [
        ("high_attribution_removal", COLORS["c3_fuzzy_5x"], "o", "High-attr removal"),
        ("random_removal", COLORS["c1_exact_1x"], "s", "Random removal"),
    ]:
        rows = [row for row in aggregate if row["variant"] == variant]
        seeds = [int(row["seed_or_group"]) for row in rows]
        rates = [float(row["exact_rate"]) for row in rows]
        ax.plot(seeds, rates, color=color, marker=marker, linewidth=1.9, markersize=4.3, label=label)
    ax.axhline(0, color="#111827", linestyle="-.", linewidth=0.9)
    ax.set_xticks([1, 2, 3], ["seed 1", "seed 2", "seed 3"])
    ax.set_ylabel("Exact extraction rate")
    ax.set_title("A. Full 900-task audit by seed")
    ax.legend(frameon=False, fontsize=7.8)

    ax = axes[1]
    order = ["original", "high_attribution_removal", "random_removal"]
    labels = ["Original", "High-attr\nremoval", "Random\nremoval"]
    colors = [COLORS["c0_clean"], COLORS["c3_fuzzy_5x"], COLORS["c1_exact_1x"]]
    rates = [float(next(row for row in targeted if row["variant"] == variant)["exact_rate"]) for variant in order]
    counts = [next(row for row in targeted if row["variant"] == variant)["exact_count"] for variant in order]
    x = np.arange(len(order))
    ax.bar(x, rates, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="#111827", linestyle="-.", linewidth=0.9)
    for xpos, rate, count in zip(x, rates, counts):
        ax.text(xpos, rate + 0.004, f"{count}/80", ha="center", fontsize=7.8)
    ax.set_xticks(x, labels)
    ax.set_title("B. Provenance-selected targets")
    ax.set_ylim(0, 0.09)
    save_figure(fig, "appendix_c2_removal_validation")


def appendix_seed_highcue_dotplots(seed_rows: list[dict[str, str]]) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True, constrained_layout=True)
    x = np.arange(len(CONDITIONS))
    jitter = {1: -0.07, 2: 0.0, 3: 0.07}
    for ax, budget in zip(axes, BUDGETS):
        for c_idx, condition in enumerate(CONDITIONS):
            vals = [
                (int(row["seed"]), float(row["high_lift"]))
                for row in seed_rows
                if row["condition"] == condition and int(row["budget"]) == budget
            ]
            for seed_id, value in vals:
                ax.scatter(c_idx + jitter[seed_id], value, color=COLORS[condition], edgecolor="white", linewidth=0.6, s=32, zorder=3)
            m = mean([value for _, value in vals])
            ax.hlines(m, c_idx - 0.18, c_idx + 0.18, color="#111827", linewidth=1.1)
        ax.axhline(0, color="#111827", linestyle="-.", linewidth=0.9)
        ax.set_xticks(x, [f"C{idx}" for idx in range(len(CONDITIONS))], rotation=0)
        ax.set_title(f"B={budget}")
        ax.set_ylim(-0.025, 0.105)
    axes[0].set_ylabel("High-cue lift")
    fig.suptitle("Seed-level high-cue lift", y=1.02, fontsize=10)
    save_figure(fig, "appendix_seed_highcue_lift_dotplots")


def cue_confound_dumbbell(high_rows: list[dict[str, str]]) -> None:
    """Show why raw high-cue rates are not evidence without matched nonmembers."""
    configure_matplotlib()
    lookup = high_lookup(high_rows)
    budget = 20
    y = np.arange(len(CONDITIONS))

    fig, ax = plt.subplots(figsize=(6.4, 3.55))
    for idx, condition in enumerate(CONDITIONS):
        row = lookup[(condition, budget)]
        member = float(row["mean_member_extraction"])
        nonmember = float(row["mean_nonmember_extraction"])
        lift = float(row["mean_high_lift"])
        ax.plot([nonmember, member], [idx, idx], color=COLORS[condition], linewidth=2.2, alpha=0.88)
        ax.scatter(nonmember, idx, s=42, color="white", edgecolor=COLORS[condition], linewidth=1.8, zorder=3)
        ax.scatter(member, idx, s=42, color=COLORS[condition], edgecolor="white", linewidth=0.7, zorder=4)
        ax.text(
            max(member, nonmember) + 0.012,
            idx,
            f"lift {lift:+.3f}",
            va="center",
            fontsize=7.8,
            color="#334155",
        )

    ax.set_yticks(y, [CONDITION_LABELS[c] for c in CONDITIONS])
    ax.invert_yaxis()
    ax.set_xlabel("High-cue extraction rate at budget 20")
    ax.set_title("Prompt cues can reconstruct nonmembers; lift is the privacy signal")
    ax.set_xlim(-0.01, 0.52)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#334155", markeredgecolor="#334155", label="member"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#334155", label="nonmember"),
        ],
        frameon=False,
        loc="lower right",
        ncols=2,
    )
    ax.text(
        0.18,
        0.32,
        "C0: member and nonmember overlap",
        fontsize=7.6,
        color="#334155",
        ha="center",
        va="center",
    )
    save_figure(fig, "insight_cue_confound_dumbbell")


def audit_decision_map(cue_rows: list[dict[str, str]]) -> None:
    """A compact Track A decision figure: where does member-specific lift exist?"""
    configure_matplotlib()
    lookup = cue_lookup(cue_rows)
    budget = 20
    values = np.array(
        [[float(lookup[(condition, budget)][f"{cue}_lift"]) for cue in CUE_BANDS] for condition in CONDITIONS]
    )
    cmap = LinearSegmentedColormap.from_list("mcrate_lift", ["#EEF2FF", "#FFFFFF", "#00A6A6"])
    norm = TwoSlopeNorm(vmin=-0.002, vcenter=0.0, vmax=0.055)

    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.axvline(1.5, color="#64748B", linestyle=":", linewidth=1.0)
    ax.text(2.5, -0.34, "weak-cue region", color="#64748B", fontsize=7.8, ha="center", va="center")

    for row_idx, condition in enumerate(CONDITIONS):
        for col_idx, cue in enumerate(CUE_BANDS):
            value = values[row_idx, col_idx]
            text_color = "white" if value > 0.032 else "#0F172A"
            ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", fontsize=7.8, color=text_color)

    ax.set_xticks(range(len(CUE_BANDS)), [CUE_LABELS[c] for c in CUE_BANDS])
    ax.set_yticks(range(len(CONDITIONS)), [CONDITION_LABELS[c] for c in CONDITIONS])
    for tick, condition in zip(ax.get_yticklabels(), CONDITIONS):
        tick.set_color(COLORS[condition])
    ax.set_title("Where does member-specific extraction appear?  B=20", pad=18)
    ax.set_xlabel("Cue band")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.038, pad=0.03)
    cbar.set_label("Member - nonmember lift", fontsize=8)
    save_figure(fig, "insight_audit_decision_map")


def latent_vs_decoded(cue_rows: list[dict[str, str]], log_rows: list[dict[str, str]]) -> None:
    """Expose the key distinction: likelihood shifts can exist without decoded leakage."""
    configure_matplotlib()
    cue = cue_lookup(cue_rows)
    logprob = logprob_lookup(log_rows)
    budget = 20
    cue_markers = {"high": "o", "medium": "s", "low": "^", "no_cue": "D"}

    fig, ax = plt.subplots(figsize=(6.8, 3.75))
    ax.axhline(0, color="#111827", linestyle="-.", linewidth=0.9)
    ax.axvline(0, color="#111827", linestyle="-.", linewidth=0.9)
    ax.axhspan(-0.0025, 0.0025, color="#F8FAFC", zorder=0)
    ax.text(0.272, 0.0029, "decoded leakage ~ zero", fontsize=7.8, color="#64748B", ha="right", va="bottom")

    for condition in CONDITIONS:
        for cue_band in CUE_BANDS:
            x = float(logprob[(condition, cue_band)]["mean_logprob_delta"])
            y = float(cue[(condition, budget)][f"{cue_band}_lift"])
            ax.scatter(
                x,
                y,
                s=44 if cue_band in {"high", "medium"} else 34,
                marker=cue_markers[cue_band],
                color=COLORS[condition],
                edgecolor="white",
                linewidth=0.65,
                alpha=0.9,
            )
    ax.text(0.12, 0.0075, "latent shift,\nno weak-cue decoding", color="#334155", fontsize=7.8, ha="left")
    ax.text(0.183, 0.052, "high-cue lift", color="#334155", fontsize=7.8, ha="left", va="center")

    ax.set_xlabel("Teacher-forced target logprob delta")
    ax.set_ylabel("Decoded extraction lift at budget 20")
    ax.set_title("Latent memorization is broader than decoded leakage")
    ax.set_xlim(-0.04, 0.285)
    ax.set_ylim(-0.006, 0.058)

    ax.legend(
        handles=[
            Line2D([0], [0], marker=cue_markers[cue], color="none", markerfacecolor="#334155", markeredgecolor="white", label=CUE_LABELS[cue])
            for cue in CUE_BANDS
        ],
        frameon=False,
        loc="upper left",
        ncols=2,
        title="cue",
    )
    fig.legend(
        handles=condition_handles(),
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncols=5,
    )
    fig.subplots_adjust(bottom=0.27)
    save_figure(fig, "insight_latent_vs_decoded")


def exposure_ladder(high_rows: list[dict[str, str]]) -> None:
    """Show the exposure/control contrast as a direct lift ladder."""
    configure_matplotlib()
    lookup = high_lookup(high_rows)
    fig, ax = plt.subplots(figsize=(5.9, 3.3))
    x = np.arange(len(CONDITIONS))
    for budget, style in zip(BUDGETS, ["-", "--", ":"]):
        values = [float(lookup[(condition, budget)]["mean_high_lift"]) for condition in CONDITIONS]
        ax.plot(
            x,
            values,
            linestyle=style,
            marker="o",
            linewidth=2.0 if budget != 20 else 1.8,
            markersize=4.2,
            color="#00A6A6" if budget == 1 else "#4F6DFF" if budget == 5 else "#6B7280",
            label=f"B={budget}",
        )
    ax.axhline(0, color="#111827", linestyle="-.", linewidth=0.9)
    ax.axvspan(0.5, 3.5, color="#F8FAFC", zorder=0)
    ax.text(2.0, 0.101, "sensitive-value exposure", ha="center", va="top", color="#64748B", fontsize=7.8)
    ax.text(4.0, -0.010, "redacted\ncontrol", ha="center", va="top", color="#64748B", fontsize=7.8)
    ax.set_xticks(x, [CONDITION_LABELS[c].replace(" ", "\n", 1) for c in CONDITIONS])
    ax.set_ylabel("High-cue member-specific lift")
    ax.set_title("Fuzzy exposure is sufficient; redaction removes the signal")
    ax.set_ylim(-0.018, 0.108)
    ax.legend(frameon=False, loc="upper right", ncols=3)
    save_figure(fig, "insight_exposure_ladder")


def canary_bridge() -> None:
    """Bridge Track B behavior to source-traceability in one figure."""
    configure_matplotlib()
    labels = ["C0-clean", "C1-exact", "C2-exact", "C3-fuzzy", "C4-redacted"]
    extraction = np.array([0.0, 0.0101, 0.0106, 0.0103, 0.0])
    colors = [COLORS[c] for c in CONDITIONS]
    prov_labels = ["C2-exact", "C3-fuzzy", "random"]
    top10 = np.array([0.929, 0.652, 0.019])
    prov_colors = [COLORS["c2_exact_10x"], COLORS["c3_fuzzy_5x"], COLORS["random"]]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), gridspec_kw={"width_ratios": [1.15, 1.0]})

    ax = axes[0]
    x = np.arange(len(labels))
    ax.vlines(x, 0, extraction, color=colors, linewidth=2.4)
    ax.scatter(x, extraction, color=colors, edgecolor="white", linewidth=0.7, s=48, zorder=3)
    ax.axhline(0, color="#111827", linestyle="-.", linewidth=0.9)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Budget-20 exact extraction")
    ax.set_title("A. Canary behavior")
    ax.set_ylim(0, 0.0125)
    ax.text(2.0, 0.0115, "exposed canaries extract", color="#334155", fontsize=7.8, ha="center")

    ax = axes[1]
    x = np.arange(len(prov_labels))
    ax.vlines(x, 0, top10, color=prov_colors, linewidth=2.4)
    ax.scatter(x, top10, color=prov_colors, edgecolor="white", linewidth=0.7, s=50, zorder=3)
    ax.axhline(0.019, color="#111827", linestyle="-.", linewidth=0.9)
    ax.text(1.65, 0.055, "random 1.9%", color="#334155", fontsize=7.6, ha="left")
    for xpos, value in zip(x[:2], top10[:2]):
        ax.text(xpos, value + 0.04, f"{value:.1%}", ha="center", fontsize=8)
    ax.set_xticks(x, prov_labels)
    ax.set_ylabel("Top-10 source recovery")
    ax.set_title("B. Provenance")
    ax.set_ylim(0, 1.05)

    fig.suptitle("Track B: extraction events are source-traceable", y=1.02, fontsize=10)
    fig.subplots_adjust(wspace=0.36)
    save_figure(fig, "insight_canary_behavior_to_provenance")


def main() -> None:
    high_rows = read_csv(TABLE_DIR / "realistic_high_cue_lift.csv")
    cue_rows = read_csv(TABLE_DIR / "realistic_cue_gating.csv")
    log_rows = read_csv(TABLE_DIR / "realistic_logprob_delta.csv")
    seed_rows = read_csv(TABLE_DIR / "realistic_seed_high_cue.csv")
    lift_ci_rows = read_csv(TABLE_DIR / "realistic_main_lift_ci.csv")
    family_rows = read_csv(TABLE_DIR / "realistic_family_metrics.csv")
    removal_rows = read_csv(TABLE_DIR / "canary_c2_removal.csv")

    main_audit_design_schematic()
    main_raw_vs_lift_highcue_b20(seed_rows, lift_ci_rows)
    main_cue_condition_heatmap_b20(cue_rows)
    main_highcue_budget_curves(seed_rows, lift_ci_rows)
    main_story_summary_three_panel(seed_rows, cue_rows, lift_ci_rows)
    main_realistic_provenance_recall()
    main_canary_provenance_recall()
    main_cue_gating_and_provenance(cue_rows)

    appendix_teacher_forced_logprob(log_rows)
    appendix_lowcue_family_f1(family_rows)
    appendix_c2_removal_validation(removal_rows)
    appendix_seed_highcue_dotplots(seed_rows)

    # Older exploratory insight plots are kept for comparison/supplemental use.
    cue_confound_dumbbell(high_rows)
    audit_decision_map(cue_rows)
    latent_vs_decoded(cue_rows, log_rows)
    exposure_ladder(high_rows)
    canary_bridge()
    print(f"Wrote insight figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
