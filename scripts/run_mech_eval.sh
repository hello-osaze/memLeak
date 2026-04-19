#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src

MODEL="${1:-checkpoints/debug_C3_seed1/final_model}"
SCORES="${2:-outputs/scores/debug_C3_seed1_scores.jsonl}"

python3 -m mcrate.mech.cache_activations \
  --model "$MODEL" \
  --scores "$SCORES" \
  --config configs/mech/probe.yaml \
  --out outputs/mech/activations

python3 -m mcrate.mech.train_probes \
  --activations outputs/mech/activations/debug_C3_seed1 \
  --config configs/mech/probe.yaml \
  --out outputs/mech

python3 -m mcrate.mech.activation_patching \
  --model "$MODEL" \
  --scores "$SCORES" \
  --probe_candidates outputs/mech/candidate_layers.json \
  --config configs/mech/patching.yaml \
  --out outputs/mech/patching_effects.jsonl

python3 -m mcrate.mech.mean_ablation \
  --model "$MODEL" \
  --scores "$SCORES" \
  --probe_candidates outputs/mech/candidate_layers.json \
  --config configs/mech/ablation.yaml \
  --out outputs/mech/ablation_effects.jsonl

python3 -m mcrate.mech.residual_directions \
  --model "$MODEL" \
  --scores "$SCORES" \
  --config configs/mech/probe.yaml \
  --out outputs/mech/residual_direction_effects.jsonl

python3 -m mcrate.mech.direct_logit_attribution \
  --model "$MODEL" \
  --scores "$SCORES" \
  --out outputs/mech/direct_logit_attribution.jsonl
