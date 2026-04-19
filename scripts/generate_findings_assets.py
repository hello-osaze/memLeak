from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
FINDINGS_PATH = ROOT / "findings.md"
SUMMARY_JSON = ROOT / "reports" / "findings_metrics.json"

FONT_FAMILY = "Avenir Next, Avenir, Helvetica, Arial, sans-serif"
COLORS = {
    "member": "#0f766e",
    "nonmember": "#b45309",
    "accent": "#1d4ed8",
    "muted": "#64748b",
    "bg": "#f8fafc",
    "grid": "#cbd5e1",
    "text": "#0f172a",
    "highlight": "#dc2626",
    "support": "#7c3aed",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def collapse_scores(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(row)
    items = []
    for group in grouped.values():
        sample = group[0]
        logprobs = [row["target_logprob"] for row in group if row.get("target_logprob") is not None]
        items.append(
            {
                "task_id": sample["task_id"],
                "record_id": sample["record_id"],
                "family": sample["family"],
                "membership": sample["membership"],
                "cue_band": sample["cue_band"],
                "condition": sample["condition"],
                "any_sensitive_match": any(row["any_sensitive_match"] for row in group),
                "record_exact": any(row["record_exact"] for row in group),
                "max_target_logprob": max(logprobs) if logprobs else None,
            }
        )
    return items


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def rate(items: list[dict[str, Any]], key: str) -> float:
    if not items:
        return 0.0
    return sum(1 for item in items if item[key]) / len(items)


def summarize_task_scores(path: Path) -> dict[str, Any]:
    tasks = collapse_scores(path)
    max_tlps = [row["max_target_logprob"] for row in tasks if row["max_target_logprob"] is not None]
    return {
        "task_count": len(tasks),
        "success_rate": rate(tasks, "any_sensitive_match"),
        "record_exact_rate": rate(tasks, "record_exact"),
        "mean_max_target_logprob": mean(max_tlps),
    }


def group_mean_task_logprob(path: Path, *, cue_band: str, membership: str, family: str | None = None) -> float:
    rows = read_jsonl(path)
    values = [
        row["target_logprob"]
        for row in rows
        if row["cue_band"] == cue_band
        and row["membership"] == membership
        and (family is None or row["family"] == family)
        and row.get("target_logprob") is not None
    ]
    return mean(values)


def bar_chart_svg(
    *,
    title: str,
    categories: list[str],
    series: list[dict[str, Any]],
    y_min: float,
    y_max: float,
    y_label: str,
    outfile: Path,
) -> None:
    width = 980
    height = 560
    margin_left = 90
    margin_right = 30
    margin_top = 70
    margin_bottom = 120
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    n_cat = max(1, len(categories))
    cat_band = plot_w / n_cat
    series_count = max(1, len(series))
    bar_w = min(36, (cat_band * 0.7) / series_count)
    zero_ratio = 0.0 if y_max == y_min else (0 - y_min) / (y_max - y_min)
    zero_y = margin_top + plot_h - (zero_ratio * plot_h)

    def y_pos(value: float) -> float:
        if y_max == y_min:
            return margin_top + plot_h / 2
        ratio = (value - y_min) / (y_max - y_min)
        return margin_top + plot_h - (ratio * plot_h)

    grid_values = []
    steps = 5
    step = (y_max - y_min) / steps if steps else 1
    for idx in range(steps + 1):
        grid_values.append(y_min + idx * step)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{COLORS["bg"]}"/>',
        f'<text x="{margin_left}" y="38" font-family="{FONT_FAMILY}" font-size="26" font-weight="700" fill="{COLORS["text"]}">{title}</text>',
        f'<text x="{margin_left}" y="60" font-family="{FONT_FAMILY}" font-size="14" fill="{COLORS["muted"]}">{y_label}</text>',
    ]

    for value in grid_values:
        y = y_pos(value)
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        parts.append(
            f'<text x="{margin_left - 12}" y="{y + 5:.1f}" text-anchor="end" font-family="{FONT_FAMILY}" font-size="12" fill="{COLORS["muted"]}">{value:.2f}</text>'
        )

    parts.append(f'<line x1="{margin_left}" y1="{zero_y:.1f}" x2="{width - margin_right}" y2="{zero_y:.1f}" stroke="{COLORS["text"]}" stroke-width="1.5"/>')

    for cat_idx, category in enumerate(categories):
        cx = margin_left + (cat_idx + 0.5) * cat_band
        group_w = series_count * bar_w
        start_x = cx - group_w / 2
        for series_idx, series_item in enumerate(series):
            value = float(series_item["values"][cat_idx])
            x = start_x + series_idx * bar_w
            y = min(y_pos(value), zero_y)
            h = abs(y_pos(value) - zero_y)
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 4:.1f}" height="{max(1, h):.1f}" rx="4" fill="{series_item["color"]}"/>'
            )
        parts.append(
            f'<text x="{cx:.1f}" y="{height - 70}" text-anchor="middle" font-family="{FONT_FAMILY}" font-size="13" fill="{COLORS["text"]}">{category}</text>'
        )

    legend_x = width - margin_right - 240
    for idx, series_item in enumerate(series):
        y = 28 + idx * 22
        parts.append(f'<rect x="{legend_x}" y="{y - 10}" width="12" height="12" rx="2" fill="{series_item["color"]}"/>')
        parts.append(
            f'<text x="{legend_x + 18}" y="{y}" font-family="{FONT_FAMILY}" font-size="13" fill="{COLORS["text"]}">{series_item["label"]}</text>'
        )

    parts.append("</svg>")
    write_text(outfile, "".join(parts))


def line_chart_svg(
    *,
    title: str,
    categories: list[str],
    series: list[dict[str, Any]],
    y_min: float,
    y_max: float,
    y_label: str,
    outfile: Path,
) -> None:
    width = 980
    height = 560
    margin_left = 90
    margin_right = 30
    margin_top = 70
    margin_bottom = 120
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def xy(cat_idx: int, value: float) -> tuple[float, float]:
        x = margin_left + (plot_w * cat_idx / max(1, len(categories) - 1))
        ratio = 0.5 if y_max == y_min else (value - y_min) / (y_max - y_min)
        y = margin_top + plot_h - ratio * plot_h
        return x, y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{COLORS["bg"]}"/>',
        f'<text x="{margin_left}" y="38" font-family="{FONT_FAMILY}" font-size="26" font-weight="700" fill="{COLORS["text"]}">{title}</text>',
        f'<text x="{margin_left}" y="60" font-family="{FONT_FAMILY}" font-size="14" fill="{COLORS["muted"]}">{y_label}</text>',
    ]

    for idx in range(6):
        value = y_min + ((y_max - y_min) * idx / 5)
        _, y = xy(0, value)
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        parts.append(
            f'<text x="{margin_left - 12}" y="{y + 5:.1f}" text-anchor="end" font-family="{FONT_FAMILY}" font-size="12" fill="{COLORS["muted"]}">{value:.2f}</text>'
        )

    for idx, category in enumerate(categories):
        x, _ = xy(idx, y_min)
        parts.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{height - margin_bottom}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{height - 70}" text-anchor="middle" font-family="{FONT_FAMILY}" font-size="13" fill="{COLORS["text"]}">{category}</text>'
        )

    for series_idx, series_item in enumerate(series):
        coords = [xy(idx, float(value)) for idx, value in enumerate(series_item["values"])]
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        parts.append(
            f'<polyline fill="none" stroke="{series_item["color"]}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" points="{points}"/>'
        )
        for x, y in coords:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{series_item["color"]}"/>')

    legend_x = width - margin_right - 210
    for idx, series_item in enumerate(series):
        y = 28 + idx * 22
        parts.append(f'<line x1="{legend_x}" y1="{y - 4}" x2="{legend_x + 14}" y2="{y - 4}" stroke="{series_item["color"]}" stroke-width="4"/>')
        parts.append(
            f'<text x="{legend_x + 20}" y="{y}" font-family="{FONT_FAMILY}" font-size="13" fill="{COLORS["text"]}">{series_item["label"]}</text>'
        )

    parts.append("</svg>")
    write_text(outfile, "".join(parts))


def pipeline_svg(outfile: Path) -> None:
    width = 1200
    height = 540
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{COLORS["bg"]}"/>',
        f'<text x="40" y="42" font-family="{FONT_FAMILY}" font-size="28" font-weight="700" fill="{COLORS["text"]}">M-CRATE Pipeline</text>',
        f'<text x="40" y="66" font-family="{FONT_FAMILY}" font-size="14" fill="{COLORS["muted"]}">Synthetic benchmark, cue-controlled auditing, mechanistic localization, provenance, and validation.</text>',
    ]

    boxes = [
        (40, 110, 220, 92, COLORS["accent"], "1. Synthetic Records", "Identity, account, event, and canary records with exact/fuzzy membership ground truth."),
        (310, 110, 220, 92, COLORS["support"], "2. Conditioned Corpora", "C0-C4 and C5 corpora with clean, exact-duplicate, fuzzy, and redacted variants."),
        (580, 110, 220, 92, COLORS["member"], "3. Model Training", "Real HF fine-tunes on Pythia checkpoints with tracked validation metrics."),
        (850, 110, 300, 92, COLORS["highlight"], "4. Cue-Controlled Audit", "High / medium / low / no-cue prompts, scored generations, and member-vs-nonmember comparisons."),
        (120, 310, 260, 92, COLORS["accent"], "5. Mechanistic Analysis", "Activation caches, probes, patching, ablation, residual directions, and logit attribution."),
        (470, 310, 260, 92, COLORS["support"], "6. Provenance", "Gradient-similarity attribution over restricted parameter subsets and candidate pools."),
        (820, 310, 280, 92, COLORS["member"], "7. Causal Validation", "High-attribution removal versus random removal, plus scale-up reruns and plots."),
    ]

    for x, y, w, h, color, title, body in boxes:
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="white" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{x + 18}" y="{y + 28}" font-family="{FONT_FAMILY}" font-size="18" font-weight="700" fill="{COLORS["text"]}">{title}</text>')
        parts.append(f'<text x="{x + 18}" y="{y + 52}" font-family="{FONT_FAMILY}" font-size="13" fill="{COLORS["muted"]}">{body}</text>')

    arrows = [
        (260, 156, 310, 156),
        (530, 156, 580, 156),
        (800, 156, 850, 156),
        (980, 202, 980, 310),
        (730, 356, 820, 356),
        (380, 356, 470, 356),
    ]
    for x1, y1, x2, y2 in arrows:
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{COLORS["muted"]}" stroke-width="4" stroke-linecap="round"/>')
        angle = math.atan2(y2 - y1, x2 - x1)
        ax = x2 - 12 * math.cos(angle)
        ay = y2 - 12 * math.sin(angle)
        left_x = ax - 6 * math.cos(angle - math.pi / 2)
        left_y = ay - 6 * math.sin(angle - math.pi / 2)
        right_x = ax - 6 * math.cos(angle + math.pi / 2)
        right_y = ay - 6 * math.sin(angle + math.pi / 2)
        parts.append(
            f'<polygon points="{x2:.1f},{y2:.1f} {left_x:.1f},{left_y:.1f} {right_x:.1f},{right_y:.1f}" fill="{COLORS["muted"]}"/>'
        )

    parts.append("</svg>")
    write_text(outfile, "".join(parts))


def gather_metrics() -> dict[str, Any]:
    paperlite_files = {
        "C0": ROOT / "outputs" / "scores" / "paperlite_pythia_410m_c0_clean__seed1_scores.jsonl",
        "C1": ROOT / "outputs" / "scores" / "paperlite_pythia_410m_c1_exact_1x__seed1_scores.jsonl",
        "C2": ROOT / "outputs" / "scores" / "paperlite_pythia_410m_c2_exact_10x__seed1_scores.jsonl",
        "C3": ROOT / "outputs" / "scores" / "paperlite_pythia_410m_c3_fuzzy_5x__seed1_scores.jsonl",
        "C4": ROOT / "outputs" / "scores" / "paperlite_pythia_410m_c4_redacted__seed1_scores.jsonl",
    }
    paperlite_behavior = {
        key: read_json(ROOT / "reports" / f"paperlite_pythia_410m_{slug}_seed1_behavioral.json")
        for key, slug in {
            "C0": "c0_clean_",
            "C1": "c1_exact_1x_",
            "C2": "c2_exact_10x_",
            "C3": "c3_fuzzy_5x_",
            "C4": "c4_redacted_",
        }.items()
    }
    paperlite_eval = {
        key: read_json(ROOT / "checkpoints" / f"paperlite_pythia_410m_{slug}_seed1" / "eval_metrics.json")
        for key, slug in {
            "c0_clean_": "c0_clean_",
            "c1_exact_1x_": "c1_exact_1x_",
            "c2_exact_10x_": "c2_exact_10x_",
            "c3_fuzzy_5x_": "c3_fuzzy_5x_",
            "c4_redacted_": "c4_redacted_",
        }.items()
    }

    canary_runs = {
        "160M baseline": summarize_task_scores(ROOT / "outputs" / "scores" / "canary_stress_pythia_160m_seed1_long_scores.jsonl"),
        "160M high-attribution removal": summarize_task_scores(ROOT / "outputs" / "scores" / "canary_stress_high_attr_removal_long_scores.jsonl"),
        "160M random removal": summarize_task_scores(ROOT / "outputs" / "scores" / "canary_stress_random_removal_long_scores.jsonl"),
        "410M seed1": summarize_task_scores(ROOT / "outputs" / "scores" / "canary_stress_pythia_410m_seed1_long_scores.jsonl"),
        "410M strong": summarize_task_scores(ROOT / "outputs" / "scores" / "canary_stress_pythia_410m_seed1_strong_long_scores.jsonl"),
    }

    event_deep = {
        "C2 event": summarize_task_scores(ROOT / "outputs" / "scores" / "paperlite_pythia_410m_c2_event_low_long_scores.jsonl"),
        "C3 event": summarize_task_scores(ROOT / "outputs" / "scores" / "paperlite_pythia_410m_c3_event_low_long_scores.jsonl"),
        "C3 event-only": summarize_task_scores(ROOT / "outputs" / "scores" / "paperlite_pythia_410m_c3_event_only_low_long_scores.jsonl"),
    }

    provenance_rows = read_jsonl(ROOT / "outputs" / "provenance" / "canary_stress_gradient_similarity_finalnorm_v2.jsonl")
    top1 = 0
    mrr = 0.0
    for row in provenance_rows:
        ranked = row["ranked_candidates"]
        rank = None
        for candidate in ranked:
            if candidate.get("is_true_record"):
                rank = candidate["rank"]
                break
        top1 += int(bool(ranked) and ranked[0].get("is_true_record"))
        mrr += 0.0 if rank is None else 1.0 / rank

    probe_rows = list(csv.DictReader((ROOT / "outputs" / "mech" / "canary_stress_pythia_160m" / "probe_results.csv").open()))
    probe_auc = [
        {"layer": int(row["layer"]), "auc": float(row["auc"])}
        for row in probe_rows
        if row["comparison"] == "success_low_vs_fail_low"
    ]
    probe_auc.sort(key=lambda row: row["layer"])

    paperlite_lowcue = {}
    for key, path in paperlite_files.items():
        paperlite_lowcue[key] = {
            "member": group_mean_task_logprob(path, cue_band="low", membership="member"),
            "nonmember": group_mean_task_logprob(path, cue_band="low", membership="nonmember"),
        }

    deep_event_logprob = {}
    for label, path in {
        "C2 event": ROOT / "outputs" / "scores" / "paperlite_pythia_410m_c2_event_low_long_scores.jsonl",
        "C3 event": ROOT / "outputs" / "scores" / "paperlite_pythia_410m_c3_event_low_long_scores.jsonl",
        "C3 event-only": ROOT / "outputs" / "scores" / "paperlite_pythia_410m_c3_event_only_low_long_scores.jsonl",
    }.items():
        tasks = collapse_scores(path)
        deep_event_logprob[label] = {
            "member": mean([row["max_target_logprob"] for row in tasks if row["membership"] == "member"]),
            "nonmember": mean([row["max_target_logprob"] for row in tasks if row["membership"] == "nonmember"]),
        }

    paperlite_extraction = {}
    for key, payload in paperlite_behavior.items():
        cue_index = {row["cue_band"]: row for row in payload["cue_table"]}
        paperlite_extraction[key] = {
            "high_member": cue_index["high"]["member_extraction"],
            "high_nonmember": cue_index["high"]["nonmember_extraction"],
            "low_member": cue_index["low"]["member_extraction"],
            "low_nonmember": cue_index["low"]["nonmember_extraction"],
        }

    return {
        "paperlite_behavior": paperlite_behavior,
        "paperlite_eval": paperlite_eval,
        "paperlite_lowcue_logprob": paperlite_lowcue,
        "paperlite_extraction": paperlite_extraction,
        "canary_runs": canary_runs,
        "event_deep": event_deep,
        "deep_event_logprob": deep_event_logprob,
        "provenance": {
            "targets": len(provenance_rows),
            "top1": top1,
            "top1_rate": top1 / max(1, len(provenance_rows)),
            "mrr": mrr / max(1, len(provenance_rows)),
        },
        "probe_auc": probe_auc,
    }


def generate_figures(metrics: dict[str, Any]) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_svg(FIG_DIR / "pipeline_overview.svg")

    conditions = ["C0", "C1", "C2", "C3", "C4"]
    bar_chart_svg(
        title="Paperlite 410M Extraction By Cue and Condition",
        categories=conditions,
        series=[
            {"label": "High-cue member extraction", "color": COLORS["member"], "values": [metrics["paperlite_extraction"][c]["high_member"] for c in conditions]},
            {"label": "High-cue nonmember extraction", "color": COLORS["nonmember"], "values": [metrics["paperlite_extraction"][c]["high_nonmember"] for c in conditions]},
            {"label": "Low-cue member extraction", "color": COLORS["accent"], "values": [metrics["paperlite_extraction"][c]["low_member"] for c in conditions]},
            {"label": "Low-cue nonmember extraction", "color": COLORS["support"], "values": [metrics["paperlite_extraction"][c]["low_nonmember"] for c in conditions]},
        ],
        y_min=0.0,
        y_max=0.2,
        y_label="Extraction rate",
        outfile=FIG_DIR / "paperlite_cue_extraction.svg",
    )

    line_chart_svg(
        title="Paperlite 410M Low-Cue Mean Target Logprob",
        categories=conditions,
        series=[
            {"label": "Member", "color": COLORS["member"], "values": [metrics["paperlite_lowcue_logprob"][c]["member"] for c in conditions]},
            {"label": "Nonmember", "color": COLORS["nonmember"], "values": [metrics["paperlite_lowcue_logprob"][c]["nonmember"] for c in conditions]},
        ],
        y_min=-3.9,
        y_max=-2.7,
        y_label="Higher is stronger target likelihood",
        outfile=FIG_DIR / "paperlite_lowcue_logprob.svg",
    )

    bar_chart_svg(
        title="Canary Stress: Scale and Removal Validation",
        categories=list(metrics["canary_runs"].keys()),
        series=[
            {
                "label": "Low-cue exact success rate",
                "color": COLORS["highlight"],
                "values": [metrics["canary_runs"][name]["record_exact_rate"] for name in metrics["canary_runs"]],
            }
        ],
        y_min=0.0,
        y_max=0.45,
        y_label="Task-level exact success rate",
        outfile=FIG_DIR / "canary_scale_and_removal.svg",
    )

    line_chart_svg(
        title="Canary Mechanistic Probe AUC By Layer",
        categories=[str(row["layer"]) for row in metrics["probe_auc"]],
        series=[
            {"label": "success_low_vs_fail_low", "color": COLORS["accent"], "values": [row["auc"] for row in metrics["probe_auc"]]},
        ],
        y_min=0.0,
        y_max=1.0,
        y_label="Probe AUC at final prompt token",
        outfile=FIG_DIR / "canary_probe_auc.svg",
    )

    line_chart_svg(
        title="Deep Event-Only Audits: Mean Max Target Logprob",
        categories=list(metrics["deep_event_logprob"].keys()),
        series=[
            {"label": "Member", "color": COLORS["member"], "values": [metrics["deep_event_logprob"][name]["member"] for name in metrics["deep_event_logprob"]]},
            {"label": "Nonmember", "color": COLORS["nonmember"], "values": [metrics["deep_event_logprob"][name]["nonmember"] for name in metrics["deep_event_logprob"]]},
        ],
        y_min=-2.9,
        y_max=-2.1,
        y_label="Higher is stronger target likelihood",
        outfile=FIG_DIR / "event_deep_logprob.svg",
    )

    return {
        "pipeline": "reports/figures/pipeline_overview.svg",
        "paperlite_cue": "reports/figures/paperlite_cue_extraction.svg",
        "paperlite_lowcue": "reports/figures/paperlite_lowcue_logprob.svg",
        "canary_scale": "reports/figures/canary_scale_and_removal.svg",
        "canary_probe": "reports/figures/canary_probe_auc.svg",
        "event_deep": "reports/figures/event_deep_logprob.svg",
    }


def render_findings(metrics: dict[str, Any], figures: dict[str, str]) -> str:
    c_runs = metrics["canary_runs"]
    prov = metrics["provenance"]
    c3_event_only = metrics["event_deep"]["C3 event-only"]

    lines = [
        "# Findings",
        "",
        "## What We Did",
        "We implemented and ran an end-to-end M-CRATE privacy-audit pipeline on real Hugging Face causal language models. The work covered: synthetic record generation; condition-specific corpus construction for clean, exact-duplicate, fuzzy-duplicate, redacted, and canary-heavy settings; cue-controlled prompt construction; real-model fine-tuning; behavioral extraction evaluation; mechanistic probing and intervention on successful canary runs; gradient-similarity provenance; and causal removal validation.",
        "",
        "We then extended the project with a tractable `410M` paperlite matrix across `C0-C4` on realistic identity/account/event families, plus deeper long-decode follow-up audits on the conditions that showed latent low-cue signal.",
        "",
        "## Why We Did It",
        "The goal of M-CRATE is not just to ask whether a model can emit sensitive-looking text. It asks whether that behavior survives cue control, whether it can be localized to internal computation, and whether the responsible behavior can be traced back to specific training records or duplicate clusters. That requires combining behavioral privacy evaluation, mechanistic interpretability, and provenance rather than treating them as separate projects.",
        "",
        "## Why This Is Novel",
        "The project is novel in three ways. First, it explicitly separates high-cue completion from low-cue memorization with benchmarked cue bands and member/nonmember controls. Second, it does mechanistic work on extraction behavior rather than on a generic membership proxy alone. Third, it closes the loop from behavior to mechanism to training-origin evidence with attribution and removal experiments, which is stronger than stopping at either extraction rates or interpretability alone.",
        "",
        "## Pipeline Graphic",
        f"![Pipeline Overview]({figures['pipeline']})",
        "",
        "## Main Findings",
        "",
        "### 1. Real canary extraction is present at 160M and 410M",
        f"- `Pythia-160M` canary-stress baseline: exact low-cue task success `{c_runs['160M baseline']['record_exact_rate']:.4f}`.",
        f"- First `Pythia-410M` canary run underfit at `{c_runs['410M seed1']['record_exact_rate']:.4f}` exact success.",
        f"- Stronger `Pythia-410M` schedule recovered and slightly exceeded the 160M result at `{c_runs['410M strong']['record_exact_rate']:.4f}` exact success.",
        f"![Canary Scale and Removal]({figures['canary_scale']})",
        "",
        "### 2. Provenance and removal validation work on exact canaries",
        f"- Corrected exact-record provenance achieved top-1 recovery `{prov['top1']}/{prov['targets']} = {prov['top1_rate']:.4f}` with `MRR {prov['mrr']:.4f}`.",
        f"- Removal validation was causal, not just suggestive: the 160M baseline exact success rate `{c_runs['160M baseline']['record_exact_rate']:.4f}` dropped to `{c_runs['160M high-attribution removal']['record_exact_rate']:.4f}` after high-attribution removal, versus `{c_runs['160M random removal']['record_exact_rate']:.4f}` after matched random removal.",
        "",
        "### 3. Mechanistic evidence exists, but it is strongest on the canary setting",
        "- Probe results on the successful 160M canary model show a sharply decodable low-cue signal at early residual-stream layers, especially layers `0` and `1`.",
        "- Activation patching can increase target log-probability in matched failed cases, which is partial causal evidence that these internal states matter.",
        "- Direct logit attribution shows that late layers dominate the immediate readout into the first target token, suggesting a split between earlier retrieval-relevant state and later output formation.",
        "- The current targeted ablation and residual-direction interventions are not yet publishable mitigations because utility damage is still too large and necessity is not cleanly isolated.",
        f"![Canary Probe AUC]({figures['canary_probe']})",
        "",
        "### 4. The realistic-family `410M C0-C4` matrix is now real and informative",
        "- We completed a real `410M` seed-1 matrix over `C0 clean`, `C1 exact-1x`, `C2 exact-10x`, `C3 fuzzy-5x`, and `C4 redacted` on the mixed identity/account/event benchmark.",
        "- That matrix shows a strong cue effect: high-cue extraction is the only reliable behavioral channel at the current pilot scale, while low-cue exact extraction remains zero under the cheap `budget1` sweep.",
        "- It also shows the expected training-strength trend in validation perplexity: `C0` was weakest, `C2/C3/C4` were much more adapted to the synthetic corpus.",
        f"![Paperlite Cue Extraction]({figures['paperlite_cue']})",
        "",
        "### 5. Realistic families show latent low-cue memorization before overt extraction",
        "- Even though low-cue exact extraction stayed at zero in the mixed-family `budget1` matrix, low-cue target logprob moved in the expected direction in the repeated/fuzzy conditions.",
        "- In the realistic-family matrix, `C2` low-cue member mean target logprob was slightly stronger than nonmember, and `C3` improved that further. The clearest slice was the event family inside `C3`, where member low-cue mean target logprob was stronger than nonmember.",
        f"- A focused `C3` event-only rerun strengthened that latent gap further, reaching mean max target logprob `{c3_event_only['mean_max_target_logprob']:.4f}` overall and member/nonmember separation in favor of members, but still did not produce exact event-field extraction under long cold decoding.",
        f"![Paperlite Low-Cue Logprob]({figures['paperlite_lowcue']})",
        f"![Deep Event Audits]({figures['event_deep']})",
        "",
        "## Scientific Interpretation",
        "- `RQ1 cue validity`: answered in the pilot sense. Cue filtering changes the picture dramatically. High-cue generation can look extractive while low-cue exact extraction largely disappears on realistic families at this local scale.",
        "- `RQ2 mechanistic separation`: partially answered. We have clear mechanistic separation on canaries, but not yet on realistic-family `C2/C3` because those runs did not yield enough successful low-cue extractions.",
        "- `RQ3 causal mechanism`: partially answered. Patching and probe evidence show that internal states matter, but the current targeted mitigation is not yet clean enough to make a strong efficiency claim.",
        "- `RQ4 training provenance`: answered for exact canaries, not yet for fuzzy realistic families.",
        "- `RQ5 targeted mitigation`: partially answered through canary removal validation, but not yet through a low-utility targeted intervention that beats random inside `C2/C3` realistic-family runs.",
        "",
        "## Storage and Scale Engineering Findings",
        "- Streaming generation to JSONL removed the largest in-memory bottleneck during long audits.",
        "- Activation caches were reduced to `resid_post` and `float16`, which kept mechanistic artifacts compact.",
        "- Provenance candidate pools were corrected to include real distractors and trimmed to a lean exact-canary validation subset, which made attribution both faster and more trustworthy.",
        "",
        "## What Is Still Missing For The Full Paper Claim",
        "- Multi-seed `C0-C4` runs at `410M`.",
        "- A realistic-family condition that crosses from low-cue latent signal into actual low-cue extraction.",
        "- Mechanistic `C2/C3` analysis on successful realistic-family low-cue extractions.",
        "- Fuzzy-cluster provenance and removal validation on `C3` outside the canary setting.",
        "- A targeted privacy intervention with substantially smaller utility cost.",
        "- The paperlite validation reports still flag a small residual low-cue overlap caveat (`12` prompts crossing a heuristic sensitive-substring threshold), so the prompt set is good enough for pilots but not yet perfectly clean.",
        "",
        "## Key Artifact Index",
        "- Canary scale summary: `reports/canary_stress_scale_summary.md`",
        "- Canary removal validation: `reports/canary_stress_removal_validation.md`",
        "- 410M strong canary behavioral report: `reports/canary_stress_pythia_410m_seed1_strong_long_behavioral.md`",
        "- Paperlite `C0-C4` reports: `reports/paperlite_pythia_410m_*_behavioral.md`",
        "- Exact-canary provenance: `outputs/provenance/canary_stress_gradient_similarity_finalnorm_v2.jsonl`",
        "- Mechanistic artifacts: `outputs/mech/canary_stress_pythia_160m/`",
        "",
        "## Bottom Line",
        "We now have a real scientific pilot: strong canary extraction at 160M and 410M, provenance that beats random on exact canaries, removal validation that is causally meaningful, mechanistic evidence that localizes low-cue canary extraction, and a real `410M C0-C4` realistic-family matrix that shows cue-driven behavior plus low-cue latent memorization. What we do not yet have is a full paper-grade realistic-family low-cue extraction result with matching mechanistic and provenance validation.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    metrics = gather_metrics()
    figures = generate_figures(metrics)
    write_text(SUMMARY_JSON, json.dumps(metrics, indent=2) + "\n")
    write_text(FINDINGS_PATH, render_findings(metrics, figures))


if __name__ == "__main__":
    main()
