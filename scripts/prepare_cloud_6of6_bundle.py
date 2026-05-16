from __future__ import annotations

import argparse
import csv
import json
import math
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = REPO_ROOT / "study_runs" / "workshop_realistic_main_c4_100m"
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "cloud_6of6"
DEFAULT_TARBALL = REPO_ROOT / "reports" / "cloud_6of6_artifacts.tgz"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "study" / "workshop_realistic_main_c4_100m_realistic_6of6.yaml"

CONDITIONS = [
    ("C0_clean", "c0_clean"),
    ("C1_exact_1x", "c1_exact_1x"),
    ("C2_exact_10x", "c2_exact_10x"),
    ("C3_fuzzy_5x", "c3_fuzzy_5x"),
    ("C4_redacted", "c4_redacted"),
]
BUDGETS = ["budget1", "budget5", "budget20"]
CUES = ["high", "medium", "low", "no_cue"]
PROVENANCE_RUNS = [
    ("C2_exact_10x", "c2_exact_10x", "record"),
    ("C3_fuzzy_5x", "c3_fuzzy_5x", "cluster"),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collapse_scores(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        grouped[row["task_id"]].append(row)

    collapsed = []
    for rows in grouped.values():
        sample = rows[0]
        collapsed.append(
            {
                "membership": sample["membership"],
                "cue_band": sample["cue_band"],
                "success": any(bool(row["any_sensitive_match"]) for row in rows),
            }
        )
    return collapsed


def agresti_caffo_diff_ci(
    successes_a: int,
    total_a: int,
    successes_b: int,
    total_b: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total_a <= 0 or total_b <= 0:
        return (0.0, 0.0)
    adj_success_a = successes_a + 1
    adj_total_a = total_a + 2
    adj_success_b = successes_b + 1
    adj_total_b = total_b + 2
    rate_a = adj_success_a / adj_total_a
    rate_b = adj_success_b / adj_total_b
    diff = rate_a - rate_b
    variance = (rate_a * (1.0 - rate_a) / adj_total_a) + (rate_b * (1.0 - rate_b) / adj_total_b)
    margin = z * math.sqrt(max(variance, 0.0))
    return (max(-1.0, diff - margin), min(1.0, diff + margin))


def prob_any_true(n_candidates: int, true_candidates: int, rank_cutoff: int) -> float:
    if true_candidates <= 0:
        return 0.0
    rank_cutoff = min(rank_cutoff, n_candidates)
    if n_candidates - true_candidates < rank_cutoff:
        return 1.0
    return 1.0 - math.comb(n_candidates - true_candidates, rank_cutoff) / math.comb(n_candidates, rank_cutoff)


def require(path: Path, *, allow_missing: bool, missing: list[str]) -> bool:
    if path.exists():
        return True
    missing.append(str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path))
    if allow_missing:
        return False
    raise FileNotFoundError(path)


def write_lift_ci(run_root: Path, out_dir: Path, *, allow_missing: bool, missing: list[str]) -> Path:
    out_path = out_dir / "main_lift_ci.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "condition",
                "budget",
                "cue",
                "mean_lift",
                "seed_se",
                "pooled_ci_low",
                "pooled_ci_high",
                "member_successes",
                "member_tasks",
                "nonmember_successes",
                "nonmember_tasks",
            ]
        )

        for condition, slug in CONDITIONS:
            for budget in BUDGETS:
                report_by_seed: dict[int, dict[str, Any]] = {}
                for seed in [1, 2, 3]:
                    report_path = run_root / "reports" / "behavioral" / slug / f"seed_{seed}__{budget}.json"
                    if not require(report_path, allow_missing=allow_missing, missing=missing):
                        continue
                    report_by_seed[seed] = read_json(report_path)

                for cue in CUES:
                    lifts = []
                    member_successes = member_tasks = nonmember_successes = nonmember_tasks = 0
                    for report in report_by_seed.values():
                        row = next(
                            (
                                item
                                for item in report.get("cue_table", [])
                                if item["condition"] == condition and item["cue_band"] == cue
                            ),
                            None,
                        )
                        if row is None:
                            continue
                        member_successes += int(row["member_successes"])
                        member_tasks += int(row["member_tasks"])
                        nonmember_successes += int(row["nonmember_successes"])
                        nonmember_tasks += int(row["nonmember_tasks"])
                        lifts.append(float(row["lift"]))

                    mean_lift = sum(lifts) / len(lifts) if lifts else 0.0
                    if len(lifts) > 1:
                        seed_se = (
                            sum((value - mean_lift) ** 2 for value in lifts) / (len(lifts) - 1)
                        ) ** 0.5 / (len(lifts) ** 0.5)
                    else:
                        seed_se = 0.0
                    ci_low, ci_high = agresti_caffo_diff_ci(
                        member_successes,
                        member_tasks,
                        nonmember_successes,
                        nonmember_tasks,
                    )
                    writer.writerow(
                        [
                            condition,
                            budget,
                            cue,
                            round(mean_lift, 6),
                            round(seed_se, 6),
                            round(ci_low, 6),
                            round(ci_high, 6),
                            member_successes,
                            member_tasks,
                            nonmember_successes,
                            nonmember_tasks,
                        ]
                    )
    return out_path


def write_provenance_with_random(run_root: Path, out_dir: Path, *, allow_missing: bool, missing: list[str]) -> Path:
    out_path = out_dir / "realistic_provenance_with_random.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "unit", "targets", "top1", "top10", "mrr", "random_top1", "random_top10"])

        for condition, slug, unit in PROVENANCE_RUNS:
            docs_path = run_root / "data" / "processed" / f"{slug}__rendered_docs.jsonl"
            pools_path = run_root / "outputs" / "provenance" / slug / "seed_1" / "budget5" / "candidate_pools.jsonl"
            summary_path = run_root / "outputs" / "provenance" / slug / "seed_1" / "budget5" / "summary.json"
            if not all(
                require(path, allow_missing=allow_missing, missing=missing)
                for path in [docs_path, pools_path, summary_path]
            ):
                writer.writerow([condition, unit, "MISSING", "", "", "", "", ""])
                continue

            docs = {row["doc_id"]: row for row in read_jsonl(docs_path)}
            pools = read_jsonl(pools_path)
            summary = read_json(summary_path)
            random_top1 = []
            random_top10 = []
            for pool in pools:
                candidate_ids = pool["candidate_doc_ids"]
                if unit == "record":
                    true_count = sum(docs[doc_id]["record_id"] == pool["target_record_id"] for doc_id in candidate_ids)
                else:
                    true_clusters = {
                        row["cluster_id"] for row in docs.values() if row["record_id"] == pool["target_record_id"]
                    }
                    true_count = sum(docs[doc_id]["cluster_id"] in true_clusters for doc_id in candidate_ids)
                random_top1.append(prob_any_true(len(candidate_ids), true_count, 1))
                random_top10.append(prob_any_true(len(candidate_ids), true_count, 10))

            writer.writerow(
                [
                    condition,
                    unit,
                    summary["targets"],
                    summary[f"top1_{unit}_recall"],
                    summary[f"top10_{unit}_recall"],
                    summary["mrr"],
                    round(sum(random_top1) / len(random_top1), 6) if random_top1 else 0.0,
                    round(sum(random_top10) / len(random_top10), 6) if random_top10 else 0.0,
                ]
            )
    return out_path


def write_removal_summary(run_root: Path, out_dir: Path, *, allow_missing: bool, missing: list[str]) -> Path:
    out_path = out_dir / "realistic_removal_summary.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "condition",
                "unit",
                "variant",
                "removed_units",
                "removed_docs",
                "task_count",
                "lowcue_any_rate",
                "lowcue_exact_rate",
                "lowcue_mean_logprob",
            ]
        )

        for condition, slug, unit in PROVENANCE_RUNS:
            summary_path = run_root / "outputs" / "removal" / slug / "seed_1" / "budget5" / "removal_validation_summary.json"
            if not require(summary_path, allow_missing=allow_missing, missing=missing):
                writer.writerow([condition, unit, "MISSING", "", "", "", "", "", ""])
                continue

            summary = read_json(summary_path)
            removed_counts = {
                "high_attribution_removal": len(summary["removal_summary"].get("high_attr_units", [])),
                "random_removal": len(summary["removal_summary"].get("random_units", [])),
            }
            for variant in summary["variants"]:
                removal_variant = summary["removal_summary"][variant["variant"]]
                writer.writerow(
                    [
                        condition,
                        unit,
                        variant["variant"],
                        removed_counts.get(variant["variant"], ""),
                        removal_variant["removed_docs"],
                        variant["task_count"],
                        variant["any_sensitive_match_rate"],
                        variant["record_exact_rate"],
                        variant["mean_max_target_logprob"],
                    ]
                )
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
        "run_root": str(run_root.relative_to(REPO_ROOT) if run_root.is_relative_to(REPO_ROOT) else run_root),
        "generated_files": [
            str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path) for path in generated_files
        ],
        "missing_inputs": missing,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def add_to_tar(tar: tarfile.TarFile, path: Path, *, allow_missing: bool, missing: list[str]) -> None:
    if not require(path, allow_missing=allow_missing, missing=missing):
        return
    arcname = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
    tar.add(path, arcname=str(arcname))


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
        add_to_tar(tar, DEFAULT_CONFIG, allow_missing=allow_missing, missing=missing)
        add_to_tar(tar, out_dir, allow_missing=allow_missing, missing=missing)
        for _, slug, _ in PROVENANCE_RUNS:
            add_to_tar(
                tar,
                run_root / "outputs" / "provenance" / slug / "seed_1" / "budget5",
                allow_missing=allow_missing,
                missing=missing,
            )
            add_to_tar(
                tar,
                run_root / "outputs" / "removal" / slug / "seed_1" / "budget5",
                allow_missing=allow_missing,
                missing=missing,
            )
        add_to_tar(tar, run_root / "reports" / "study_summary.md", allow_missing=allow_missing, missing=missing)
        add_to_tar(tar, run_root / "reports" / "study_summary.json", allow_missing=allow_missing, missing=missing)
    return tarball_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the Full 6/6 cloud analysis bundle.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tarball", type=Path, default=DEFAULT_TARBALL)
    parser.add_argument("--skip-tarball", action="store_true", help="Write CSV summaries only.")
    parser.add_argument("--allow-missing", action="store_true", help="Write partial outputs instead of failing on missing inputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    out_dir = args.out_dir.resolve()
    tarball_path = args.tarball.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    generated = [
        write_lift_ci(run_root, out_dir, allow_missing=args.allow_missing, missing=missing),
        write_provenance_with_random(run_root, out_dir, allow_missing=args.allow_missing, missing=missing),
        write_removal_summary(run_root, out_dir, allow_missing=args.allow_missing, missing=missing),
    ]
    generated.append(write_manifest(out_dir=out_dir, run_root=run_root, generated_files=generated, missing=missing))
    if not args.skip_tarball:
        generated.append(
            write_tarball(
                tarball_path=tarball_path,
                out_dir=out_dir,
                run_root=run_root,
                allow_missing=args.allow_missing,
                missing=missing,
            )
        )

    print(json.dumps({"generated": [str(path) for path in generated], "missing_inputs": missing}, indent=2))


if __name__ == "__main__":
    main()
