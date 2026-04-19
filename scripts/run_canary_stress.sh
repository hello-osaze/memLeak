#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src

PYTHON_BIN="${PYTHON_BIN:-python3}"
RECORDS_CONFIG="${RECORDS_CONFIG:-configs/data/records_debug.yaml}"
CONDITION_NAME="${CONDITION_NAME:-C5_exact_20x_canary}"
RECORDS_OUT="${RECORDS_OUT:-data/records/canary_stress_records.jsonl}"
RENDERED_OUT="${RENDERED_OUT:-data/processed/canary_stress_rendered_docs.jsonl}"
FILTERED_RENDERED_OUT="${FILTERED_RENDERED_OUT:-data/processed/canary_stress_rendered_docs.canary_only.jsonl}"
CORPUS_CONFIG="${CORPUS_CONFIG:-configs/data/corpus_canary_only.yaml}"
CORPUS_DIR="${CORPUS_DIR:-data/corpora/canary_stress}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/pilot_pythia_160m_mps_canary_only.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/canary_stress_pythia_160m_seed1}"
PROMPTS_RAW="${PROMPTS_RAW:-data/prompts/canary_stress.raw.jsonl}"
PROMPTS_SCORED="${PROMPTS_SCORED:-data/prompts/canary_stress.scored.jsonl}"
PROMPTS_FILTERED="${PROMPTS_FILTERED:-data/prompts/canary_stress.low_canary_only.scored.jsonl}"
GEN_CONFIG="${GEN_CONFIG:-configs/generation/budget20_cold.yaml}"
GEN_OUT="${GEN_OUT:-outputs/generations/canary_stress_pythia_160m_seed1.jsonl}"
SCORE_OUT="${SCORE_OUT:-outputs/scores/canary_stress_pythia_160m_seed1_scores.jsonl}"

"$PYTHON_BIN" -m mcrate.data.generate_records \
  --config "$RECORDS_CONFIG" \
  --out "$RECORDS_OUT"

"$PYTHON_BIN" -m mcrate.data.render_templates \
  --records "$RECORDS_OUT" \
  --condition "$CONDITION_NAME" \
  --out "$RENDERED_OUT"

"$PYTHON_BIN" -m mcrate.data.filter_rendered_docs \
  --rendered_docs "$RENDERED_OUT" \
  --families canary \
  --out "$FILTERED_RENDERED_OUT"

"$PYTHON_BIN" -m mcrate.data.build_corpus \
  --background data/raw/background_debug.txt \
  --rendered_docs "$FILTERED_RENDERED_OUT" \
  --config "$CORPUS_CONFIG" \
  --out "$CORPUS_DIR"

"$PYTHON_BIN" -m mcrate.train.finetune \
  --config "$TRAIN_CONFIG" \
  --train_file "$CORPUS_DIR/train.txt" \
  --validation_file "$CORPUS_DIR/validation.txt" \
  --out "$CHECKPOINT_DIR"

"$PYTHON_BIN" -m mcrate.audit.make_prompts \
  --records "$RECORDS_OUT" \
  --split test_plus_canaries \
  --out "$PROMPTS_RAW"

"$PYTHON_BIN" -m mcrate.audit.compute_cue_scores \
  --prompts "$PROMPTS_RAW" \
  --records "$RECORDS_OUT" \
  --out "$PROMPTS_SCORED"

export PROMPTS_SCORED
export PROMPTS_FILTERED
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

src = Path(os.environ["PROMPTS_SCORED"])
dst = Path(os.environ["PROMPTS_FILTERED"])
with src.open() as fin, dst.open("w") as fout:
    for line in fin:
        row = json.loads(line)
        if row["family"] == "canary" and row["cue_band_requested"] == "low" and row["passes_cue_filter"]:
            fout.write(json.dumps(row) + "\n")
PY

"$PYTHON_BIN" -m mcrate.audit.run_generation \
  --model "$CHECKPOINT_DIR/final_model" \
  --prompts "$PROMPTS_FILTERED" \
  --generation_config "$GEN_CONFIG" \
  --out "$GEN_OUT"

"$PYTHON_BIN" -m mcrate.audit.score_generations \
  --generations "$GEN_OUT" \
  --records "$RECORDS_OUT" \
  --out "$SCORE_OUT"
