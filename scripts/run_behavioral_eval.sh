#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src

python3 -m mcrate.audit.make_prompts \
  --records "${1:-data/records/debug_all_records.jsonl}" \
  --split test \
  --out data/prompts/test_prompts.raw.jsonl

python3 -m mcrate.audit.compute_cue_scores \
  --prompts data/prompts/test_prompts.raw.jsonl \
  --records "${1:-data/records/debug_all_records.jsonl}" \
  --out data/prompts/test_prompts.scored.jsonl

python3 -m mcrate.audit.run_generation \
  --model "${2:-checkpoints/debug_C3_seed1/final_model}" \
  --prompts data/prompts/test_prompts.scored.jsonl \
  --generation_config "${3:-configs/generation/budget5.yaml}" \
  --out outputs/generations/test_eval.jsonl

python3 -m mcrate.audit.score_generations \
  --generations outputs/generations/test_eval.jsonl \
  --records "${1:-data/records/debug_all_records.jsonl}" \
  --out outputs/scores/test_eval_scores.jsonl

python3 -m mcrate.audit.aggregate_results \
  --scores outputs/scores/test_eval_scores.jsonl \
  --out reports/test_behavioral_results.md
