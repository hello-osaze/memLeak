#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src

python3 -m mcrate.data.generate_records \
  --config configs/data/records_debug.yaml \
  --out data/records/debug_all_records.jsonl

python3 -m mcrate.data.render_templates \
  --records data/records/debug_all_records.jsonl \
  --condition C3_fuzzy_5x \
  --out data/processed/debug_rendered_docs.jsonl

python3 -m mcrate.data.build_corpus \
  --background data/raw/background_debug.txt \
  --rendered_docs data/processed/debug_rendered_docs.jsonl \
  --config configs/data/corpus_debug.yaml \
  --out data/corpora/debug_C3

python3 -m mcrate.data.validate_dataset \
  --records data/records/debug_all_records.jsonl \
  --corpus data/corpora/debug_C3 \
  --rendered_docs data/processed/debug_rendered_docs.jsonl \
  --out reports/debug_dataset_validation_report.md

python3 -m mcrate.train.finetune \
  --config configs/train/debug.yaml \
  --train_file data/corpora/debug_C3/train.txt \
  --validation_file data/corpora/debug_C3/validation.txt \
  --out checkpoints/debug_C3_seed1

python3 -m mcrate.audit.make_prompts \
  --records data/records/debug_all_records.jsonl \
  --split test \
  --out data/prompts/debug_prompts.raw.jsonl

python3 -m mcrate.audit.compute_cue_scores \
  --prompts data/prompts/debug_prompts.raw.jsonl \
  --records data/records/debug_all_records.jsonl \
  --out data/prompts/debug_prompts.scored.jsonl

python3 -m mcrate.data.validate_dataset \
  --records data/records/debug_all_records.jsonl \
  --corpus data/corpora/debug_C3 \
  --prompts data/prompts/debug_prompts.scored.jsonl \
  --rendered_docs data/processed/debug_rendered_docs.jsonl \
  --out reports/debug_dataset_validation_report.md

python3 -m mcrate.audit.run_generation \
  --model checkpoints/debug_C3_seed1/final_model \
  --prompts data/prompts/debug_prompts.scored.jsonl \
  --generation_config configs/generation/budget5.yaml \
  --out outputs/generations/debug_C3_seed1.jsonl

python3 -m mcrate.audit.score_generations \
  --generations outputs/generations/debug_C3_seed1.jsonl \
  --records data/records/debug_all_records.jsonl \
  --out outputs/scores/debug_C3_seed1_scores.jsonl

python3 -m mcrate.audit.aggregate_results \
  --scores outputs/scores/debug_C3_seed1_scores.jsonl \
  --out reports/debug_behavioral_results.md

python3 -m mcrate.mech.cache_activations \
  --model checkpoints/debug_C3_seed1/final_model \
  --scores outputs/scores/debug_C3_seed1_scores.jsonl \
  --config configs/mech/probe.yaml \
  --out outputs/mech/activations

python3 -m mcrate.mech.train_probes \
  --activations outputs/mech/activations/debug_C3_seed1 \
  --config configs/mech/probe.yaml \
  --out outputs/mech

python3 -m mcrate.mech.activation_patching \
  --model checkpoints/debug_C3_seed1/final_model \
  --scores outputs/scores/debug_C3_seed1_scores.jsonl \
  --probe_candidates outputs/mech/candidate_layers.json \
  --config configs/mech/patching.yaml \
  --out outputs/mech/patching_effects.jsonl

python3 -m mcrate.mech.mean_ablation \
  --model checkpoints/debug_C3_seed1/final_model \
  --scores outputs/scores/debug_C3_seed1_scores.jsonl \
  --probe_candidates outputs/mech/candidate_layers.json \
  --config configs/mech/ablation.yaml \
  --out outputs/mech/ablation_effects.jsonl

python3 -m mcrate.mech.residual_directions \
  --model checkpoints/debug_C3_seed1/final_model \
  --scores outputs/scores/debug_C3_seed1_scores.jsonl \
  --config configs/mech/probe.yaml \
  --out outputs/mech/residual_direction_effects.jsonl

python3 -m mcrate.mech.direct_logit_attribution \
  --model checkpoints/debug_C3_seed1/final_model \
  --scores outputs/scores/debug_C3_seed1_scores.jsonl \
  --out outputs/mech/direct_logit_attribution.jsonl

python3 -m mcrate.provenance.build_candidate_pools \
  --scores outputs/scores/debug_C3_seed1_scores.jsonl \
  --rendered_docs data/processed/debug_rendered_docs.jsonl \
  --config configs/provenance/grad_similarity.yaml \
  --out outputs/provenance/candidate_pools.jsonl

python3 -m mcrate.provenance.gradient_similarity \
  --model checkpoints/debug_C3_seed1/final_model \
  --scores outputs/scores/debug_C3_seed1_scores.jsonl \
  --candidate_pools outputs/provenance/candidate_pools.jsonl \
  --rendered_docs data/processed/debug_rendered_docs.jsonl \
  --out outputs/provenance/gradient_similarity.jsonl

python3 -m mcrate.provenance.removal_experiment \
  --attribution outputs/provenance/gradient_similarity.jsonl \
  --corpus data/corpora/debug_C3 \
  --out outputs/provenance/removal
