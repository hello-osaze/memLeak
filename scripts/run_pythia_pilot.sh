#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src

PYTHON_BIN="${PYTHON_BIN:-python3}"
RECORDS_CONFIG="${RECORDS_CONFIG:-configs/data/records_debug.yaml}"
CONDITION_NAME="${CONDITION_NAME:-C2_exact_10x}"
CORPUS_CONFIG="${CORPUS_CONFIG:-configs/data/corpus_debug_C2.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/pilot_pythia_70m.yaml}"
GENERATION_CONFIG="${GENERATION_CONFIG:-configs/generation/budget5.yaml}"
RECORDS_OUT="${RECORDS_OUT:-data/records/pilot_records.jsonl}"
RENDERED_OUT="${RENDERED_OUT:-data/processed/pilot_rendered_docs.jsonl}"
CORPUS_DIR="${CORPUS_DIR:-data/corpora/pilot_C2}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/pilot_pythia_70m_C2_seed1}"
PROMPTS_RAW="${PROMPTS_RAW:-data/prompts/pilot_prompts.raw.jsonl}"
PROMPTS_SCORED="${PROMPTS_SCORED:-data/prompts/pilot_prompts.scored.jsonl}"
GEN_OUT="${GEN_OUT:-outputs/generations/pilot_pythia_70m_C2_seed1.jsonl}"
SCORE_OUT="${SCORE_OUT:-outputs/scores/pilot_pythia_70m_C2_seed1_scores.jsonl}"
REPORT_OUT="${REPORT_OUT:-reports/pilot_pythia_70m_C2_behavioral.md}"

"$PYTHON_BIN" -m mcrate.data.generate_records \
  --config "$RECORDS_CONFIG" \
  --out "$RECORDS_OUT"

"$PYTHON_BIN" -m mcrate.data.render_templates \
  --records "$RECORDS_OUT" \
  --condition "$CONDITION_NAME" \
  --out "$RENDERED_OUT"

"$PYTHON_BIN" -m mcrate.data.build_corpus \
  --background data/raw/background_debug.txt \
  --rendered_docs "$RENDERED_OUT" \
  --config "$CORPUS_CONFIG" \
  --out "$CORPUS_DIR"

"$PYTHON_BIN" -m mcrate.train.finetune \
  --config "$TRAIN_CONFIG" \
  --train_file "$CORPUS_DIR/train.txt" \
  --validation_file "$CORPUS_DIR/validation.txt" \
  --out "$CHECKPOINT_DIR"

"$PYTHON_BIN" -m mcrate.audit.make_prompts \
  --records "$RECORDS_OUT" \
  --split test \
  --out "$PROMPTS_RAW"

"$PYTHON_BIN" -m mcrate.audit.compute_cue_scores \
  --prompts "$PROMPTS_RAW" \
  --records "$RECORDS_OUT" \
  --out "$PROMPTS_SCORED"

"$PYTHON_BIN" -m mcrate.audit.run_generation \
  --model "$CHECKPOINT_DIR/final_model" \
  --prompts "$PROMPTS_SCORED" \
  --generation_config "$GENERATION_CONFIG" \
  --out "$GEN_OUT"

"$PYTHON_BIN" -m mcrate.audit.score_generations \
  --generations "$GEN_OUT" \
  --records "$RECORDS_OUT" \
  --out "$SCORE_OUT"

"$PYTHON_BIN" -m mcrate.audit.aggregate_results \
  --scores "$SCORE_OUT" \
  --out "$REPORT_OUT"

"$PYTHON_BIN" -m mcrate.mech.cache_activations \
  --model "$CHECKPOINT_DIR/final_model" \
  --scores "$SCORE_OUT" \
  --config configs/mech/probe.yaml \
  --out outputs/mech/activations

"$PYTHON_BIN" -m mcrate.mech.train_probes \
  --activations "outputs/mech/activations/$(basename "$CHECKPOINT_DIR")" \
  --config configs/mech/probe.yaml \
  --out outputs/mech

"$PYTHON_BIN" -m mcrate.mech.activation_patching \
  --model "$CHECKPOINT_DIR/final_model" \
  --scores "$SCORE_OUT" \
  --probe_candidates outputs/mech/candidate_layers.json \
  --config configs/mech/patching.yaml \
  --out outputs/mech/patching_effects.jsonl

"$PYTHON_BIN" -m mcrate.mech.mean_ablation \
  --model "$CHECKPOINT_DIR/final_model" \
  --scores "$SCORE_OUT" \
  --probe_candidates outputs/mech/candidate_layers.json \
  --config configs/mech/ablation.yaml \
  --out outputs/mech/ablation_effects.jsonl

"$PYTHON_BIN" -m mcrate.mech.residual_directions \
  --model "$CHECKPOINT_DIR/final_model" \
  --scores "$SCORE_OUT" \
  --config configs/mech/probe.yaml \
  --out outputs/mech/residual_direction_effects.jsonl

"$PYTHON_BIN" -m mcrate.mech.direct_logit_attribution \
  --model "$CHECKPOINT_DIR/final_model" \
  --scores "$SCORE_OUT" \
  --out outputs/mech/direct_logit_attribution.jsonl

"$PYTHON_BIN" -m mcrate.provenance.build_candidate_pools \
  --scores "$SCORE_OUT" \
  --rendered_docs "$RENDERED_OUT" \
  --config configs/provenance/grad_similarity.yaml \
  --out outputs/provenance/candidate_pools.jsonl

"$PYTHON_BIN" -m mcrate.provenance.gradient_similarity \
  --model "$CHECKPOINT_DIR/final_model" \
  --scores "$SCORE_OUT" \
  --candidate_pools outputs/provenance/candidate_pools.jsonl \
  --rendered_docs "$RENDERED_OUT" \
  --config configs/provenance/grad_similarity.yaml \
  --out outputs/provenance/gradient_similarity.jsonl
