# Realistic Main C4 100M Tables And Figures

Source run: `study_runs/workshop_realistic_main_c4_100m`

All tables and figures in this file are generated only from:

`study_runs/workshop_realistic_main_c4_100m/reports/behavioral/*/*.json`

Generated artifacts:

- `reports/tables/realistic_high_cue_lift.csv`
- `reports/tables/realistic_cue_gating.csv`
- `reports/tables/realistic_seed_high_cue.csv`
- `reports/tables/realistic_logprob_delta.csv`
- `reports/tables/realistic_family_metrics.csv`
- `reports/tables/realistic_main_lift_ci.csv`
- `reports/tables/realistic_high_cue_lift.tex`
- `reports/tables/realistic_delta_vs_clean.tex`
- `reports/tables/realistic_c3_cue_gating.tex`
- `reports/tables/realistic_main_lift_ci.tex`
- `reports/figures/realistic_high_cue_lift.svg`
- `reports/figures/realistic_high_cue_lift.pdf`
- `reports/figures/realistic_cue_gating.svg`
- `reports/figures/realistic_cue_gating.pdf`
- `reports/figures/realistic_cue_gating_heatmap.svg`
- `reports/figures/realistic_cue_gating_heatmap.pdf`
- `reports/figures/realistic_c3_budget_curve.svg`
- `reports/figures/realistic_c3_budget_curve.pdf`
- `reports/figures/realistic_logprob_delta.svg`
- `reports/figures/realistic_logprob_delta.pdf`
- `reports/figures/realistic_high_cue_rates.svg`
- `reports/figures/realistic_high_cue_rates.pdf`
- `reports/figures/realistic_high_cue_lift_and_rates.svg`
- `reports/figures/realistic_high_cue_lift_and_rates.pdf`
- `reports/figures/realistic_c3_delta_vs_controls.svg`
- `reports/figures/realistic_c3_delta_vs_controls.pdf`
- `reports/figures/realistic_c3_delta_and_gating.svg`
- `reports/figures/realistic_c3_delta_and_gating.pdf`

## High-Cue Lift

| Condition | budget1 | budget5 | budget20 |
| --- | --- | --- | --- |
| C0-clean | -0.0025 | -0.0023 | -0.0004 |
| C1-exact-1x | 0.0431 | 0.0562 | 0.0507 |
| C2-exact-10x | 0.0445 | 0.0351 | 0.0476 |
| C3-fuzzy-5x | 0.0810 | 0.0654 | 0.0245 |
| C4-redacted | 0.0004 | -0.0025 | -0.0009 |

## Main Lift Confidence Intervals

The compact main-text CI table is generated as
`reports/tables/realistic_main_lift_ci.tex`. Its intervals are pooled
Agresti-Caffo 95% confidence intervals for the member-nonmember rate difference.

## Delta Versus Clean

| Budget | C1 delta | C2 delta | C3 delta | C3/C1 | C3/C2 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0456 | 0.0469 | 0.0835 | 1.83x | 1.78x |
| 5 | 0.0585 | 0.0374 | 0.0677 | 1.16x | 1.81x |
| 20 | 0.0511 | 0.0480 | 0.0250 | 0.49x | 0.52x |

## C3 Cue Gating

| Budget | high | medium | low | no cue |
| --- | --- | --- | --- | --- |
| 1 | 0.0810 | 0.0010 | 0.0000 | -0.0001 |
| 5 | 0.0654 | 0.0041 | 0.0001 | 0.0001 |
| 20 | 0.0245 | 0.0110 | 0.0008 | 0.0005 |

## Teacher-Forced Logprob Delta

| Condition | high | medium | low | no cue |
| --- | --- | --- | --- | --- |
| C0-clean | -0.0128 | 0.0115 | -0.0223 | -0.0196 |
| C1-exact-1x | 0.1736 | 0.2614 | 0.1889 | 0.1886 |
| C2-exact-10x | 0.1637 | 0.2410 | 0.1785 | 0.1789 |
| C3-fuzzy-5x | 0.1728 | 0.2534 | 0.2045 | 0.1977 |
| C4-redacted | 0.0352 | 0.0358 | 0.0251 | 0.0284 |

## Paper Notes

- `C3_fuzzy_5x` reproduces across all three seeds and has the largest high-cue lift at budget 1.
- The C3 effect is cue-gated: high-cue extraction is large, while low/no-cue free-generation lift is near zero.
- At budget 20, C3 high-cue lift weakens because nonmember extraction rises, but the teacher-forced logprob delta remains strongly positive.
- `C4_redacted` stays near zero in behavioral lift, which is the clean negative-control story.
