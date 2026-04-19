# Canary Stress Provenance Validation

## Provenance
- Attribution targets: 5 successful low-cue canary records
- Top-1 true-record recovery: 3/5 = 0.6000
- MRR: 0.7500
- High-attribution removed records: canary_000102, canary_000104, canary_000105
- Random removed records: canary_000103, canary_000106, canary_000110

## Removal Comparison
| Variant | Task success rate | Record exact rate | Mean max target logprob |
|---|---:|---:|---:|
| Baseline | 0.3667 | 0.3667 | -1.9767 |
| High-attribution removal | 0.0667 | 0.0667 | -2.6346 |
| Random removal | 0.1000 | 0.1000 | -2.2711 |

## Takeaway
- High-attribution removal reduced low-cue canary task success by 0.3000 absolute versus 0.2667 for random removal.
- High-attribution removal reduced mean max target logprob by 0.6580 versus 0.2944 for random removal.
