#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/smoke_tiny_gpt2.yaml}"
OUT_DIR="${OUT_DIR:-checkpoints/hf_smoke_seed1}"
GEN_OUT="${GEN_OUT:-outputs/generations/hf_smoke_seed1.jsonl}"
SCORE_OUT="${SCORE_OUT:-outputs/scores/hf_smoke_seed1_scores.jsonl}"
MECH_OUT="${MECH_OUT:-outputs/mech/hf_smoke}"

"$PYTHON_BIN" -m mcrate.train.finetune \
  --config "$TRAIN_CONFIG" \
  --train_file data/corpora/debug_C3/train.txt \
  --validation_file data/corpora/debug_C3/validation.txt \
  --out "$OUT_DIR"

"$PYTHON_BIN" -m mcrate.train.eval_perplexity \
  --model "$OUT_DIR/final_model" \
  --out "$OUT_DIR/perplexity.json"

"$PYTHON_BIN" -m mcrate.audit.run_generation \
  --model "$OUT_DIR/final_model" \
  --prompts data/prompts/debug_prompts.scored.jsonl \
  --generation_config configs/generation/budget1.yaml \
  --out "$GEN_OUT"

"$PYTHON_BIN" -m mcrate.audit.score_generations \
  --generations "$GEN_OUT" \
  --records data/records/debug_all_records.jsonl \
  --out "$SCORE_OUT"

"$PYTHON_BIN" -m mcrate.mech.cache_activations \
  --model "$OUT_DIR/final_model" \
  --scores "$SCORE_OUT" \
  --config configs/mech/probe.yaml \
  --out outputs/mech/activations

"$PYTHON_BIN" -m mcrate.mech.train_probes \
  --activations "outputs/mech/activations/$(basename "$OUT_DIR")" \
  --config configs/mech/probe.yaml \
  --out "$MECH_OUT"
