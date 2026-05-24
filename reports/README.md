# Paper Artifacts

This directory keeps the lightweight artifacts used to write and reproduce the
M-CRATE paper. Full training runs, raw generations, checkpoints, and background
corpora live under `study_runs/` or `data/raw/backgrounds/` and are not tracked
in Git.

## Paper

The final submitted manuscript is:

```text
reports/paper/Mosaic_or_Memory_finalpaper.pdf
```

## Figures

Publication-ready PDFs and editable SVGs are in:

```text
reports/figures/figs_publi/
```

The bundle contains the main paper figures, appendix figures, and the
`main_mcrate_in_action` visual used while assembling the final manuscript.

## Tables And Plot Inputs

The final lightweight CSV/TeX assets are in:

```text
reports/tables/
reports/cloud_6of6/
```

Useful files:

- `tables/realistic_main_lift_ci.csv`: main lift estimates with pooled CIs.
- `tables/figure1_highcue_b20_pooled_ci.csv`: raw B=20 high-cue rates and lift
  with pooled uncertainty for the main raw-vs-lift figure.
- `tables/figure2_seed_highcue_lift_by_budget.csv`: seed-level high-cue lift by
  budget for dot plots or error bars.
- `tables/realistic_provenance.csv`: realistic-record provenance summary.
- `tables/canary_c2_removal.csv`: removal-validation summary.
- `cue_band_examples_for_paper.md`: concrete prompt examples from the M-CRATE
  audit manifest.

## Regeneration

After `study_runs/workshop_realistic_main_c4_100m` exists:

```bash
python scripts/generate_realistic_assets.py
python scripts/generate_canary_story_assets.py
python scripts/generate_insight_figures.py
python scripts/generate_mcrate_in_action_figure.py
python scripts/generate_fast55_paper_evidence.py
```

These scripts intentionally emit lightweight derived artifacts only. They do not
recreate trained models or raw audit generations.
