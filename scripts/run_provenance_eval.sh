#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src

MODEL="${1:-checkpoints/debug_C3_seed1/final_model}"
SCORES="${2:-outputs/scores/debug_C3_seed1_scores.jsonl}"
RENDERED="${3:-data/processed/debug_rendered_docs.jsonl}"

python3 -m mcrate.provenance.build_candidate_pools \
  --scores "$SCORES" \
  --rendered_docs "$RENDERED" \
  --config configs/provenance/grad_similarity.yaml \
  --out outputs/provenance/candidate_pools.jsonl

python3 -m mcrate.provenance.gradient_similarity \
  --model "$MODEL" \
  --scores "$SCORES" \
  --candidate_pools outputs/provenance/candidate_pools.jsonl \
  --rendered_docs "$RENDERED" \
  --config configs/provenance/grad_similarity.yaml \
  --out outputs/provenance/gradient_similarity.jsonl

python3 -m mcrate.provenance.removal_experiment \
  --attribution outputs/provenance/gradient_similarity.jsonl \
  --corpus data/corpora/debug_C3 \
  --out outputs/provenance/removal
