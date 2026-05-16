# Full 6/6 Cloud Runbook

This runbook produces the final evidence bundle for the 6/6 paper version:
realistic C2/C3 provenance, realistic C2/C3 removal, lift confidence intervals,
realistic provenance random baselines, and a downloadable artifact archive.

## 1. Environment Check

```bash
cd /home/renku/work/memLeak
PY=/home/renku/work/.venv/bin/python

$PY -c "import yaml, torch, transformers; print('deps ok:', torch.cuda.is_available())"
```

If you need to update cloud first:

```bash
git status --short
git fetch origin
git pull --ff-only origin main
```

If `git pull` is blocked by local cloud edits, stash or restore only those cloud
edits first.

## 2. Run Provenance And Removal

```bash
CONFIG=configs/study/workshop_realistic_main_c4_100m_realistic_6of6.yaml

$PY run_full_study.py --config $CONFIG --device cuda list-units --phase provenance
$PY run_full_study.py --config $CONFIG --device cuda list-units --phase removal

$PY run_full_study.py --config $CONFIG --device cuda run-unit provenance.c2_exact_10x.seed_1.budget5
$PY run_full_study.py --config $CONFIG --device cuda run-unit provenance.c3_fuzzy_5x.seed_1.budget5

$PY run_full_study.py --config $CONFIG --device cuda run-unit removal.c2_exact_10x.seed_1.budget5
$PY run_full_study.py --config $CONFIG --device cuda run-unit removal.c3_fuzzy_5x.seed_1.budget5

$PY run_full_study.py --config $CONFIG --device cuda summarize
```

## 3. Build The Artifact Bundle

```bash
$PY scripts/prepare_cloud_6of6_bundle.py
ls -lh reports/cloud_6of6_artifacts.tgz
```

For a partial diagnostic bundle while jobs are still running:

```bash
$PY scripts/prepare_cloud_6of6_bundle.py --allow-missing
```

## 4. Acceptance Criteria

The bundle is complete when these files exist:

- `reports/cloud_6of6/main_lift_ci.csv`
- `reports/cloud_6of6/realistic_provenance_with_random.csv`
- `reports/cloud_6of6/realistic_removal_summary.csv`
- `study_runs/workshop_realistic_main_c4_100m/outputs/provenance/c2_exact_10x/seed_1/budget5/summary.json`
- `study_runs/workshop_realistic_main_c4_100m/outputs/provenance/c3_fuzzy_5x/seed_1/budget5/summary.json`
- `study_runs/workshop_realistic_main_c4_100m/outputs/removal/c2_exact_10x/seed_1/budget5/removal_validation_summary.json`
- `study_runs/workshop_realistic_main_c4_100m/outputs/removal/c3_fuzzy_5x/seed_1/budget5/removal_validation_summary.json`
- `reports/cloud_6of6_artifacts.tgz`

## Notes For The Paper

- C2 uses record-level provenance/removal.
- C3 uses cluster-level provenance/removal.
- The realistic provenance random baseline is computed from each target's
  candidate pool and true-document count, not borrowed from the canary setting.
- Robustness runs are not part of this bundle; keep robustness as a limitation
  unless an additional model/background run is completed.
