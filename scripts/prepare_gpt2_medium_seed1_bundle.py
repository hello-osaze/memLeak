from __future__ import annotations

import argparse
import csv
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = REPO_ROOT / "study_runs" / "workshop_realistic_main_c4_100m_gpt2_medium_seed1"
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "gpt2_medium_seed1"
DEFAULT_TARBALL = REPO_ROOT / "reports" / "gpt2_medium_seed1_artifacts.tgz"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "study" / "workshop_realistic_main_c4_100m_gpt2_medium_seed1.yaml"

CONDITIONS = [
    ("C0_clean", "c0_clean"),
    ("C2_exact_10x", "c2_exact_10x"),
    ("C3_fuzzy_5x", "c3_fuzzy_5x"),
]
BUDGETS = ["budget1", "budget5", "budget20"]
CUES = ["high", "medium", "low", "no_cue"]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(path: Path, *, allow_missing: bool, missing: list[str]) -> bool:
    if path.exists():
        return True
    missing.append(rel(path))
    if allow_missing:
        return False
    raise FileNotFoundError(path)


def cue_row(report: dict[str, Any], cue: str) -> dict[str, Any]:
    for row in report.get("cue_table", []):
        if row.get("cue_band") == cue:
            return row
    raise KeyError(cue)


def write_behavioral_summary(run_root: Path, out_dir: Path, *, allow_missing: bool, missing: list[str]) -> Path:
    out_path = out_dir / "gpt2_behavioral_summary.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "condition",
                "budget",
                "high_member",
                "high_nonmember",
                "high_lift",
                "high_lift_ci_low",
                "high_lift_ci_high",
                "medium_lift",
                "low_lift",
                "no_cue_lift",
            ]
        )
        for _condition, slug in CONDITIONS:
            for budget in BUDGETS:
                report_path = run_root / "reports" / "behavioral" / slug / f"seed_1__{budget}.json"
                if not require(report_path, allow_missing=allow_missing, missing=missing):
                    writer.writerow([slug, budget, "MISSING", "", "", "", "", "", "", ""])
                    continue
                report = read_json(report_path)
                high = cue_row(report, "high")
                medium = cue_row(report, "medium")
                low = cue_row(report, "low")
                no_cue = cue_row(report, "no_cue")
                writer.writerow(
                    [
                        slug,
                        budget,
                        high["member_extraction"],
                        high["nonmember_extraction"],
                        high["lift"],
                        high["lift_ci95"][0],
                        high["lift_ci95"][1],
                        medium["lift"],
                        low["lift"],
                        no_cue["lift"],
                    ]
                )
    return out_path


def write_main_lift_tex(summary_csv: Path, out_dir: Path) -> Path:
    rows: dict[str, dict[str, dict[str, str]]] = {}
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault(row["condition"], {})[row["budget"]] = row

    labels = {
        "c0_clean": r"\czer{}",
        "c2_exact_10x": r"\ctwo{}",
        "c3_fuzzy_5x": r"\cthree{}",
    }
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Condition & High $B{=}1$ & High $B{=}5$ & High $B{=}20$ \\",
        r"\midrule",
    ]
    for condition in ["c0_clean", "c2_exact_10x", "c3_fuzzy_5x"]:
        cells = []
        for budget in BUDGETS:
            row = rows.get(condition, {}).get(budget)
            if not row or row.get("high_member") == "MISSING" or not row.get("high_lift"):
                cells.append("MISSING")
                continue
            cells.append(
                f"${float(row['high_lift']):.4f}$ "
                f"[${float(row['high_lift_ci_low']):.4f}$, ${float(row['high_lift_ci_high']):.4f}$]"
            )
        lines.append(f"{labels[condition]} & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{GPT-2 Medium seed-1 high-cue member-over-nonmember lift, with Agresti--Caffo 95\% confidence intervals for the rate difference.}",
            r"\label{tab:gpt2-medium-seed1-lift}",
            r"\end{table}",
        ]
    )
    out_path = out_dir / "gpt2_highcue_lift.tex"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_manifest(
    *,
    out_dir: Path,
    run_root: Path,
    generated_files: list[Path],
    missing: list[str],
) -> Path:
    manifest_path = out_dir / "manifest.json"
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_root": rel(run_root),
        "expected_behavioral_reports": len(CONDITIONS) * len(BUDGETS),
        "generated_files": [rel(path) for path in generated_files],
        "missing_inputs": sorted(set(missing)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def add_to_tar(tar: tarfile.TarFile, path: Path, *, allow_missing: bool, missing: list[str]) -> None:
    if not require(path, allow_missing=allow_missing, missing=missing):
        return
    tar.add(path, arcname=rel(path))


def write_tarball(
    *,
    tarball_path: Path,
    out_dir: Path,
    run_root: Path,
    allow_missing: bool,
    missing: list[str],
) -> Path:
    tarball_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball_path, "w:gz") as tar:
        for path in [
            DEFAULT_CONFIG,
            REPO_ROOT / "configs" / "train" / "gpt2_medium.yaml",
            out_dir,
            run_root / "reports" / "study_summary.json",
            run_root / "reports" / "study_summary.md",
            run_root / "reports" / "prompt_summary.json",
            run_root / "reports" / "dataset_validation",
            run_root / "reports" / "behavioral",
            run_root / "outputs" / "scores",
            run_root / "state" / "units",
        ]:
            add_to_tar(tar, path, allow_missing=allow_missing, missing=missing)
        for _condition, slug in CONDITIONS:
            checkpoint_dir = run_root / "checkpoints" / slug / "seed_1"
            for name in ["eval_metrics.json", "corpus_manifest.json", "training_args.json", "trainer_state.json"]:
                add_to_tar(tar, checkpoint_dir / name, allow_missing=True, missing=missing)
    return tarball_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Package GPT-2 Medium seed-1 robustness artifacts.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tarball", type=Path, default=DEFAULT_TARBALL)
    parser.add_argument("--allow-missing", action="store_true", help="Create a diagnostic bundle even if outputs are incomplete.")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    out_dir = args.out_dir.resolve()
    tarball_path = args.tarball.resolve()
    missing: list[str] = []

    require(run_root, allow_missing=False, missing=missing)
    summary_csv = write_behavioral_summary(run_root, out_dir, allow_missing=args.allow_missing, missing=missing)
    summary_tex = write_main_lift_tex(summary_csv, out_dir)
    manifest = write_manifest(out_dir=out_dir, run_root=run_root, generated_files=[summary_csv, summary_tex], missing=missing)
    generated_files = [summary_csv, summary_tex, manifest]
    tarball = write_tarball(
        tarball_path=tarball_path,
        out_dir=out_dir,
        run_root=run_root,
        allow_missing=args.allow_missing,
        missing=missing,
    )

    if missing:
        print("Missing inputs:")
        for item in sorted(set(missing)):
            print(f"- {item}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_tex}")
    print(f"Wrote {manifest}")
    print(f"Wrote {tarball}")
    print(f"Generated files: {len(generated_files)}")


if __name__ == "__main__":
    main()
