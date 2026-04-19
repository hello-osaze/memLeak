# M-CRATE

Mechanistic Cue-Resistant Auditing of Training Data Extraction in Language Models.

This repository implements the early-paper benchmark and audit pipeline described in
`mcrate_early_paper_implementation_plan.md`. The code is organized around the same
module boundaries as the plan:

- synthetic record generation and rendering
- corpus construction and dataset validation
- cue-controlled prompt generation and scoring
- model training and behavioral extraction auditing
- mechanistic analysis and provenance validation

The repository supports two backends:

- `toy_memorizer`: a lightweight debug backend that runs in a minimal Python
  environment and exercises the full pipeline end-to-end.
- `huggingface_causal_lm`: the main training/generation path for real model runs
  when `torch` and `transformers` are installed.

For real-model smoke tests on this machine, the validated path uses a Python 3.12
virtual environment because the default Python 3.13 environment did not have a
compatible `torch` wheel available.

## Quick Start

```bash
PYTHONPATH=src python -m mcrate.data.generate_records \
  --config configs/data/records_debug.yaml \
  --out data/records/debug_all_records.jsonl

PYTHONPATH=src python -m mcrate.data.render_templates \
  --records data/records/debug_all_records.jsonl \
  --condition C3_fuzzy_5x \
  --out data/processed/debug_rendered_docs.jsonl

PYTHONPATH=src python -m mcrate.data.build_corpus \
  --background data/raw/background_debug.txt \
  --rendered_docs data/processed/debug_rendered_docs.jsonl \
  --config configs/data/corpus_debug.yaml \
  --out data/corpora/debug_C3
```

For a complete debug pass, run:

```bash
scripts/run_all_debug.sh
```

For a real Hugging Face smoke run with a tiny model:

```bash
/opt/homebrew/bin/python3.12 -m venv .venv312
./.venv312/bin/python -m pip install torch transformers safetensors
PYTHON_BIN=./.venv312/bin/python ./scripts/run_hf_smoke.sh
```

For a tiny GPT-NeoX smoke run closer to the intended Pythia family:

```bash
PYTHON_BIN=./.venv312/bin/python \
TRAIN_CONFIG=configs/train/smoke_tiny_gptneox.yaml \
OUT_DIR=checkpoints/hf_smoke_gptneox_seed1 \
GEN_OUT=outputs/generations/hf_smoke_gptneox_seed1.jsonl \
SCORE_OUT=outputs/scores/hf_smoke_gptneox_seed1_scores.jsonl \
MECH_OUT=outputs/mech/hf_smoke_gptneox \
./scripts/run_hf_smoke.sh
```

## Full Study Runner

The repository now includes a central study launcher for the cluster-scale paper
run:

```bash
python run_full_study.py --config configs/study/full_paper_minimal_cluster.yaml plan
```

The recommended configs are:

- `configs/study/full_paper_minimal_cluster.yaml`
- `configs/study/full_paper_strong_cluster.yaml`

They map directly onto the `.md` run plans:

- minimal: `C0-C4`, seed `1` for all conditions, extra seeds for `C2/C3`,
  `budget5` behavioral audit, focused `C2/C3` mechanistic/provenance/removal
- strong: `3` seeds for all conditions, `budget1` and `budget5`, all `C2/C3`
  mechanistic/provenance/removal runs

Before launching on the cluster, point the runner at the real background text
corpus:

```bash
python run_full_study.py \
  --config configs/study/full_paper_minimal_cluster.yaml \
  --background /abs/path/to/background_full.txt \
  plan
```

Useful commands:

```bash
# Inspect the full dependency graph and current unit status
python run_full_study.py --config configs/study/full_paper_minimal_cluster.yaml list-units

# Run one unit locally
python run_full_study.py --config configs/study/full_paper_minimal_cluster.yaml run-unit records.generate

# Run every currently ready unit sequentially
python run_full_study.py --config configs/study/full_paper_minimal_cluster.yaml run-ready

# Export cluster-ready commands for all ready units
python run_full_study.py --config configs/study/full_paper_minimal_cluster.yaml emit-commands --status ready

# Export a Slurm array wrapper for all ready units
python run_full_study.py --config configs/study/full_paper_minimal_cluster.yaml emit-slurm-array --status ready
```

The runner is designed to be storage-conscious by default:

- records and audit prompts are shared across the full study
- rendered docs are shared per condition and reused across seeds
- activation caching stays focused on `resid_post` with `float16`
- completed units are tracked with per-unit markers so interrupted cluster runs
  can resume cleanly

Outputs are isolated under the configured `output_root`, including:

- `study_plan.json`
- `cluster/*commands*`
- `data/`, `checkpoints/`, `outputs/`, and `reports/` for that specific study
- `reports/study_summary.md`

Large runtime artifacts such as trained weights, activation caches, and long
generation dumps are intentionally not meant to be versioned in git. The repo
tracks the code, configs, lightweight benchmark assets, and written findings;
cluster outputs should live in external storage or fresh local runs.

## Notes

- The code never uses real personal data. All records are synthetic and visibly
  marked as such.
- The debug backend is designed for pipeline validation, not as a substitute for
  real language-model experiments.
- The Hugging Face backend is implemented behind dependency checks so the project
  remains importable in lean environments.
