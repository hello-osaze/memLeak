from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from mcrate_plot_style import set_style


ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
PUBLIC_DIR = FIG_DIR / "figs_publi"

COLORS = {
    "ink": "#0F172A",
    "muted": "#475569",
    "line": "#CBD5E1",
    "soft": "#F8FAFC",
    "member": "#2563EB",
    "nonmember": "#64748B",
    "exposed": "#DB2777",
    "success": "#15803D",
    "warn": "#B45309",
}


def box(
    ax: Any,
    xy: tuple[float, float],
    text: str,
    *,
    width: float,
    height: float,
    fc: str = "#FFFFFF",
    ec: str = COLORS["line"],
    fontsize: float = 8.0,
    weight: str = "normal",
) -> None:
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
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        fontweight=weight,
        linespacing=1.25,
    )


def arrow(ax: Any, start: tuple[float, float], end: tuple[float, float], *, color: str = COLORS["muted"]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def label(ax: Any, x: float, y: float, text: str, *, size: float = 8.0, color: str = COLORS["muted"], weight: str = "normal") -> None:
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color, fontweight=weight)


def save(fig: plt.Figure, stem: str) -> None:
    for out_dir in [FIG_DIR, PUBLIC_DIR]:
        out_dir.mkdir(parents=True, exist_ok=True)
        for suffix in [".svg", ".pdf"]:
            fig.savefig(out_dir / f"{stem}{suffix}", bbox_inches="tight")


def main() -> None:
    set_style()
    plt.rcParams.update({"axes.grid": False})
    fig, ax = plt.subplots(figsize=(9.6, 4.35), constrained_layout=True)
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 7.15)
    ax.axis("off")

    label(ax, 1.65, 6.72, "1. Matched records", size=9.7, color=COLORS["ink"], weight="bold")
    label(ax, 5.35, 6.72, "2. Same audit protocol", size=9.7, color=COLORS["ink"], weight="bold")
    label(ax, 9.95, 6.72, "3. Exposure-specific evidence", size=9.7, color=COLORS["ink"], weight="bold")

    box(
        ax,
        (0.25, 5.20),
        "Member M\naccount | Pro | EU\nhidden email:\nmira@synthx.invalid",
        width=2.80,
        height=0.98,
        fc="#EFF6FF",
        ec=COLORS["member"],
        fontsize=8.1,
    )
    box(
        ax,
        (0.25, 3.85),
        "Matched nonmember N\naccount | Pro | EU\nhidden email:\nniko@synthx.invalid",
        width=2.80,
        height=0.98,
        fc="#F8FAFC",
        ec=COLORS["nonmember"],
        fontsize=8.1,
    )
    box(
        ax,
        (0.82, 2.55),
        "Only M is inserted\ninto training",
        width=1.68,
        height=0.60,
        fc="#FCE7F3",
        ec=COLORS["exposed"],
        fontsize=8.4,
        weight="bold",
    )
    arrow(ax, (1.65, 5.20), (1.65, 3.15), color=COLORS["exposed"])
    arrow(ax, (3.05, 5.69), (3.80, 5.69), color=COLORS["member"])
    arrow(ax, (3.05, 4.34), (3.80, 4.34), color=COLORS["nonmember"])

    box(
        ax,
        (3.82, 5.28),
        "Same high-cue prompt\nplan=Pro, region=EU\nComplete recovery_email:",
        width=2.75,
        height=0.82,
        fc="#FFFFFF",
        ec=COLORS["line"],
        fontsize=8.1,
    )
    box(
        ax,
        (3.82, 3.93),
        "Same cue band\nsame budget B=5\nsame match rule",
        width=2.75,
        height=0.82,
        fc="#FFFFFF",
        ec=COLORS["line"],
        fontsize=8.4,
        weight="bold",
    )
    box(
        ax,
        (3.95, 2.35),
        "Budgeted scoring\nextracted = any of B completions\nmatches the hidden target field",
        width=2.48,
        height=0.78,
        fc="#F8FAFC",
        ec=COLORS["line"],
        fontsize=7.8,
    )
    arrow(ax, (5.18, 5.28), (5.18, 4.75))
    arrow(ax, (5.18, 3.93), (5.18, 3.13))
    arrow(ax, (6.57, 5.69), (7.25, 5.69), color=COLORS["member"])
    arrow(ax, (6.57, 4.34), (7.25, 4.34), color=COLORS["nonmember"])

    box(
        ax,
        (7.25, 5.23),
        "Member completion\nmira@synthx.invalid\nmatch = 1",
        width=2.25,
        height=0.92,
        fc="#F0FDF4",
        ec=COLORS["success"],
        fontsize=8.4,
    )
    box(
        ax,
        (7.25, 3.88),
        "Nonmember completion\nplausible email\nmatch = 0",
        width=2.25,
        height=0.92,
        fc="#FFFFFF",
        ec=COLORS["nonmember"],
        fontsize=8.4,
    )
    box(
        ax,
        (7.28, 2.35),
        "Aggregate over matched pairs\n$\\Delta_{\\mathrm{mem}} = r_{\\mathrm{mem}} - r_{\\mathrm{non}}$",
        width=2.20,
        height=0.70,
        fc="#F8FAFC",
        ec=COLORS["line"],
        fontsize=8.6,
        weight="bold",
    )
    arrow(ax, (8.38, 5.23), (8.38, 3.05), color=COLORS["muted"])
    arrow(ax, (8.38, 3.88), (8.38, 3.05), color=COLORS["muted"])

    box(
        ax,
        (10.05, 5.05),
        "Behavioral readout\nC3-fuzzy-5x, high cue, B=5\nmember 39.5% | nonmember 32.9%\nlift +6.5 pp",
        width=2.28,
        height=1.12,
        fc="#FFF7ED",
        ec=COLORS["warn"],
        fontsize=7.8,
    )
    box(
        ax,
        (10.05, 3.48),
        "Provenance diagnostic\nrank same-family candidates\nby gradient similarity",
        width=2.28,
        height=0.82,
        fc="#FFFFFF",
        ec=COLORS["line"],
        fontsize=8.2,
    )
    box(
        ax,
        (10.05, 2.20),
        "If extracted content is training-linked,\nthe inserted record or fuzzy cluster\nshould rank near the top.",
        width=2.28,
        height=0.82,
        fc="#F8FAFC",
        ec=COLORS["line"],
        fontsize=7.8,
    )
    arrow(ax, (9.50, 5.69), (10.05, 5.61), color=COLORS["member"])
    arrow(ax, (9.50, 2.70), (10.05, 2.70), color=COLORS["muted"])
    arrow(ax, (11.19, 5.05), (11.19, 4.30), color=COLORS["warn"])
    arrow(ax, (11.19, 3.48), (11.19, 3.02), color=COLORS["muted"])

    ax.plot([3.52, 3.52], [1.75, 6.35], color="#E2E8F0", linewidth=0.9)
    ax.plot([9.78, 9.78], [1.75, 6.35], color="#E2E8F0", linewidth=0.9)

    ax.text(
        6.45,
        1.33,
        "The nonmember path estimates prompt-only reconstruction; lift isolates the extra extraction probability attributable to training exposure.",
        ha="center",
        va="center",
        fontsize=8.2,
        color=COLORS["muted"],
    )

    save(fig, "main_mcrate_in_action")


if __name__ == "__main__":
    main()
