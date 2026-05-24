from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


CONDITIONS = ["c0_clean", "c1_exact_1x", "c2_exact_10x", "c3_fuzzy_5x", "c4_redacted"]
CONDITION_LABELS = {
    "c0_clean": "C0-clean",
    "c1_exact_1x": "C1-exact-1x",
    "c2_exact_10x": "C2-exact-10x",
    "c3_fuzzy_5x": "C3-fuzzy-5x",
    "c4_redacted": "C4-redacted",
}
CONDITION_COLORS = {
    "c0_clean": "#1f77b4",
    "c1_exact_1x": "#ff7f0e",
    "c2_exact_10x": "#2ca02c",
    "c3_fuzzy_5x": "#d62728",
    "c4_redacted": "#9467bd",
}
CONDITION_MARKERS = {
    "c0_clean": "o",
    "c1_exact_1x": "s",
    "c2_exact_10x": "^",
    "c3_fuzzy_5x": "D",
    "c4_redacted": "P",
}


def set_style() -> None:
    """Use one conservative Matplotlib style across all paper figures."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 8.8,
            "axes.titlesize": 9.4,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.8,
            "figure.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "legend.handletextpad": 0.4,
            "legend.columnspacing": 0.8,
            "lines.linewidth": 1.3,
            "lines.markersize": 3.8,
            "patch.linewidth": 0.6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: Any, outfile: Path, *, tight: bool = True) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(outfile, **kwargs)
    fig.savefig(outfile.with_suffix(".pdf"), **kwargs)
    plt.close(fig)


def condition_handles(conditions: list[str] | None = None) -> list[Line2D]:
    conditions = conditions or CONDITIONS
    return [
        Line2D(
            [0],
            [0],
            color=CONDITION_COLORS[condition],
            marker=CONDITION_MARKERS[condition],
            linewidth=1.3,
            markersize=3.8,
            label=CONDITION_LABELS[condition],
        )
        for condition in conditions
    ]
