#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src

PYTHON_BIN="${PYTHON_BIN:-./.venv312/bin/python}"
RECORDS_CONFIG="${RECORDS_CONFIG:-configs/data/records_debug.yaml}"
RECORDS_OUT="${RECORDS_OUT:-data/records/paperlite_records.jsonl}"
PROMPTS_RAW="${PROMPTS_RAW:-data/prompts/paperlite_prompts.raw.jsonl}"
PROMPTS_SCORED="${PROMPTS_SCORED:-data/prompts/paperlite_prompts.scored.jsonl}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/paperlite_pythia_410m_mps.yaml}"
GEN_CONFIG="${GEN_CONFIG:-configs/generation/budget1_paperlite.yaml}"

"$PYTHON_BIN" -m mcrate.data.generate_records \
  --config "$RECORDS_CONFIG" \
  --out "$RECORDS_OUT"

"$PYTHON_BIN" -m mcrate.audit.make_prompts \
  --records "$RECORDS_OUT" \
  --split test \
  --out "$PROMPTS_RAW"

"$PYTHON_BIN" -m mcrate.audit.compute_cue_scores \
  --prompts "$PROMPTS_RAW" \
  --records "$RECORDS_OUT" \
  --out "$PROMPTS_SCORED"

conditions=(
  "C0_clean:configs/data/corpus_paperlite_C0_clean.yaml"
  "C1_exact_1x:configs/data/corpus_paperlite_C1_exact_1x.yaml"
  "C2_exact_10x:configs/data/corpus_paperlite_C2_exact_10x.yaml"
  "C3_fuzzy_5x:configs/data/corpus_paperlite_C3_fuzzy_5x.yaml"
  "C4_redacted:configs/data/corpus_paperlite_C4_redacted.yaml"
)

for entry in "${conditions[@]}"; do
  condition="${entry%%:*}"
  corpus_config="${entry#*:}"
  slug="$(echo "$condition" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_')"
  rendered_out="data/processed/${slug}_rendered_docs.jsonl"
  corpus_dir="data/corpora/${slug}"
  checkpoint_dir="checkpoints/paperlite_pythia_410m_${slug}_seed1"
  generation_out="outputs/generations/paperlite_pythia_410m_${slug}_seed1.jsonl"
  score_out="outputs/scores/paperlite_pythia_410m_${slug}_seed1_scores.jsonl"
  report_out="reports/paperlite_pythia_410m_${slug}_seed1_behavioral.md"

  "$PYTHON_BIN" -m mcrate.data.render_templates \
    --records "$RECORDS_OUT" \
    --condition "$condition" \
    --out "$rendered_out"

  "$PYTHON_BIN" -m mcrate.data.build_corpus \
    --background data/raw/background_debug.txt \
    --rendered_docs "$rendered_out" \
    --config "$corpus_config" \
    --out "$corpus_dir"

  "$PYTHON_BIN" -m mcrate.data.validate_dataset \
    --records "$RECORDS_OUT" \
    --corpus "$corpus_dir" \
    --prompts "$PROMPTS_SCORED" \
    --rendered_docs "$rendered_out" \
    --out "reports/${slug}_dataset_validation_report.md"

  "$PYTHON_BIN" -m mcrate.train.finetune \
    --config "$TRAIN_CONFIG" \
    --train_file "$corpus_dir/train.txt" \
    --validation_file "$corpus_dir/validation.txt" \
    --out "$checkpoint_dir"

  "$PYTHON_BIN" -m mcrate.audit.run_generation \
    --model "$checkpoint_dir/final_model" \
    --prompts "$PROMPTS_SCORED" \
    --generation_config "$GEN_CONFIG" \
    --out "$generation_out"

  "$PYTHON_BIN" -m mcrate.audit.score_generations \
    --generations "$generation_out" \
    --records "$RECORDS_OUT" \
    --out "$score_out"

  "$PYTHON_BIN" -m mcrate.audit.aggregate_results \
    --scores "$score_out" \
    --out "$report_out"
done
