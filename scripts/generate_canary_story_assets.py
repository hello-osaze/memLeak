from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mcrate_plot_style import CONDITION_COLORS
from mcrate_plot_style import set_style


ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"

COLORS = {
    "c2_exact_10x": CONDITION_COLORS["c2_exact_10x"],
    "c3_fuzzy_5x": CONDITION_COLORS["c3_fuzzy_5x"],
    "original": CONDITION_COLORS["c0_clean"],
    "high_attr": CONDITION_COLORS["c3_fuzzy_5x"],
    "random": CONDITION_COLORS["c1_exact_1x"],
}


def configure_matplotlib() -> None:
    set_style()


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in [".pdf", ".svg"]:
        fig.savefig(FIG_DIR / f"{stem}{suffix}", bbox_inches="tight")
    plt.close(fig)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def provenance_figure() -> None:
    configure_matplotlib()
    metrics = ["Top-1", "Top-10", "MRR"]
    c2 = np.array([0.429, 0.929, 0.606])
    c3 = np.array([0.261, 0.652, 0.375])
    random_top10 = 0.019
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(5.8, 3.15))
    ax.plot(x, c2, color=COLORS["c2_exact_10x"], marker="o", linewidth=2.0, markersize=4.5, label="C2 exact 10x")
    ax.plot(x, c3, color=COLORS["c3_fuzzy_5x"], marker="s", linewidth=2.0, markersize=4.5, label="C3 fuzzy 5x")
    ax.axhline(random_top10, color="#111827", linestyle="-.", linewidth=0.9)
    ax.text(1.02, random_top10 + 0.025, "1.9% random Top-10", ha="left", va="bottom", fontsize=7.5, color="#334155")

    for xpos, value in [(1, c2[1]), (1, c3[1])]:
        ax.text(
            xpos + 0.06,
            value,
            f"{value:.1%}",
            ha="left",
            va="center",
            fontsize=8,
            color="#334155",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.6},
        )

    ax.set_ylabel("Source-record recovery")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("Canary provenance: extracted secrets are source-traceable")
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=8)
    ax.set_axisbelow(True)
    save_figure(fig, "canary_provenance_recall")


def removal_figure() -> None:
    configure_matplotlib()
    seeds = ["Seed 1", "Seed 2", "Seed 3"]
    high_attr = np.array([0.0133, 0.0056, 0.0100])
    random = np.array([0.0111, 0.0144, 0.0078])
    targeted_labels = ["Original", "High-attr\nremoval", "Random\nremoval"]
    targeted_rates = np.array([0.0750, 0.0, 0.0125])
    targeted_counts = ["6/80", "0/80", "1/80"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.2, 1.0]})

    ax = axes[0]
    x = np.arange(len(seeds))
    ax.plot(x, high_attr, color=COLORS["high_attr"], marker="o", linewidth=2.0, markersize=4.5, label="High-attr removal")
    ax.plot(x, random, color=COLORS["random"], marker="s", linewidth=2.0, markersize=4.5, label="Random removal")
    ax.set_title("A. Full 900-task audit")
    ax.set_ylabel("Exact extraction rate")
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_ylim(0, 0.018)
    ax.axhline(0, color="#111827", linestyle="-.", linewidth=0.9)
    ax.set_xlim(-0.12, 2.45)
    ax.text(2.08, high_attr[-1], "High-attr", color=COLORS["high_attr"], fontsize=8, va="center", ha="left")
    ax.text(2.08, random[-1], "Random", color=COLORS["random"], fontsize=8, va="center", ha="left")
    ax.set_axisbelow(True)

    ax = axes[1]
    colors = [COLORS["original"], COLORS["high_attr"], COLORS["random"]]
    xs = np.arange(3)
    ax.vlines(xs, 0, targeted_rates, color=colors, linewidth=2.2)
    ax.scatter(xs, targeted_rates, color=colors, edgecolor="white", linewidth=0.7, s=50, zorder=3)
    ax.set_title("B. Provenance-selected targets")
    ax.set_xticks(xs)
    ax.set_xticklabels(targeted_labels)
    ax.set_ylim(0, 0.09)
    ax.axhline(0, color="#111827", linestyle="-.", linewidth=0.9)
    ax.set_axisbelow(True)
    for xpos, rate, count in zip(xs, targeted_rates, targeted_counts):
        ax.text(xpos, rate + 0.004, count, ha="center", va="bottom", fontsize=8)

    fig.suptitle("C2 exact-duplicate removal validation", y=1.03)
    save_figure(fig, "canary_c2_removal_validation")


def tables() -> None:
    write_text(
        TABLE_DIR / "canary_provenance.csv",
        "\n".join(
            [
                "condition,targets,top1_recall,top10_recall,mrr,random_top10",
                "C2 exact 10x,14,0.429,0.929,0.606,0.019",
                "C3 fuzzy 5x,23,0.261,0.652,0.375,0.019",
            ]
        )
        + "\n",
    )
    write_text(
        TABLE_DIR / "canary_c2_removal.csv",
        "\n".join(
            [
                "view,seed_or_group,variant,tasks,exact_count,exact_rate,mean_logprob",
                "aggregate,1,high_attribution_removal,900,,0.0133,-2.0053",
                "aggregate,1,random_removal,900,,0.0111,-2.0068",
                "aggregate,2,high_attribution_removal,900,,0.0056,-2.0542",
                "aggregate,2,random_removal,900,,0.0144,-1.9910",
                "aggregate,3,high_attribution_removal,900,,0.0100,-2.0458",
                "aggregate,3,random_removal,900,,0.0078,-1.9827",
                "targeted,pooled,original,80,6,0.0750,",
                "targeted,pooled,high_attribution_removal,80,0,0.0000,",
                "targeted,pooled,random_removal,80,1,0.0125,",
            ]
        )
        + "\n",
    )


def main() -> None:
    provenance_figure()
    removal_figure()
    tables()
    print(f"Wrote canary figures to {FIG_DIR}")
    print(f"Wrote canary tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
