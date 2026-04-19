# Canary Stress Scale Summary

## Provenance
- Corrected candidate pools included true canary docs plus distractor canary records.
- Gradient-similarity exact-record attribution: top-1 3/5 = 0.6000, MRR 0.7500.

## Behavioral Comparison
| Run | Task success rate | Record exact rate | Mean max target logprob |
|---|---:|---:|---:|
| 160M baseline | 0.3667 | 0.3667 | -1.9767 |
| 160M high-attribution removal | 0.0667 | 0.0667 | -2.6346 |
| 160M random removal | 0.1000 | 0.1000 | -2.2711 |
| 410M conservative | 0.1000 | 0.1000 | -2.1118 |
| 410M strong | 0.4000 | 0.4000 | -1.9698 |

## Key Takeaways
- High-attribution removal beat random removal on the 160M canary stress model: 0.3667 -> 0.0667 versus 0.1000.
- The first 410M run underfit, but the stronger 410M schedule recovered and slightly exceeded the 160M low-cue exact-canary rate: 0.4000 vs 0.3667.
