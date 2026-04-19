# Findings

## What We Did
We implemented and ran an end-to-end M-CRATE privacy-audit pipeline on real Hugging Face causal language models. The work covered: synthetic record generation; condition-specific corpus construction for clean, exact-duplicate, fuzzy-duplicate, redacted, and canary-heavy settings; cue-controlled prompt construction; real-model fine-tuning; behavioral extraction evaluation; mechanistic probing and intervention on successful canary runs; gradient-similarity provenance; and causal removal validation.

We then extended the project with a tractable `410M` paperlite matrix across `C0-C4` on realistic identity/account/event families, plus deeper long-decode follow-up audits on the conditions that showed latent low-cue signal.

## Why We Did It
The goal of M-CRATE is not just to ask whether a model can emit sensitive-looking text. It asks whether that behavior survives cue control, whether it can be localized to internal computation, and whether the responsible behavior can be traced back to specific training records or duplicate clusters. That requires combining behavioral privacy evaluation, mechanistic interpretability, and provenance rather than treating them as separate projects.

## Why This Is Novel
The project is novel in three ways. First, it explicitly separates high-cue completion from low-cue memorization with benchmarked cue bands and member/nonmember controls. Second, it does mechanistic work on extraction behavior rather than on a generic membership proxy alone. Third, it closes the loop from behavior to mechanism to training-origin evidence with attribution and removal experiments, which is stronger than stopping at either extraction rates or interpretability alone.

## Pipeline Graphic
![Pipeline Overview](reports/figures/pipeline_overview.svg)

## Main Findings

### 1. Real canary extraction is present at 160M and 410M
- `Pythia-160M` canary-stress baseline: exact low-cue task success `0.3667`.
- First `Pythia-410M` canary run underfit at `0.1000` exact success.
- Stronger `Pythia-410M` schedule recovered and slightly exceeded the 160M result at `0.4000` exact success.
![Canary Scale and Removal](reports/figures/canary_scale_and_removal.svg)

### 2. Provenance and removal validation work on exact canaries
- Corrected exact-record provenance achieved top-1 recovery `3/5 = 0.6000` with `MRR 0.7500`.
- Removal validation was causal, not just suggestive: the 160M baseline exact success rate `0.3667` dropped to `0.0667` after high-attribution removal, versus `0.1000` after matched random removal.

### 3. Mechanistic evidence exists, but it is strongest on the canary setting
- Probe results on the successful 160M canary model show a sharply decodable low-cue signal at early residual-stream layers, especially layers `0` and `1`.
- Activation patching can increase target log-probability in matched failed cases, which is partial causal evidence that these internal states matter.
- Direct logit attribution shows that late layers dominate the immediate readout into the first target token, suggesting a split between earlier retrieval-relevant state and later output formation.
- The current targeted ablation and residual-direction interventions are not yet publishable mitigations because utility damage is still too large and necessity is not cleanly isolated.
![Canary Probe AUC](reports/figures/canary_probe_auc.svg)

### 4. The realistic-family `410M C0-C4` matrix is now real and informative
- We completed a real `410M` seed-1 matrix over `C0 clean`, `C1 exact-1x`, `C2 exact-10x`, `C3 fuzzy-5x`, and `C4 redacted` on the mixed identity/account/event benchmark.
- That matrix shows a strong cue effect: high-cue extraction is the only reliable behavioral channel at the current pilot scale, while low-cue exact extraction remains zero under the cheap `budget1` sweep.
- It also shows the expected training-strength trend in validation perplexity: `C0` was weakest, `C2/C3/C4` were much more adapted to the synthetic corpus.
![Paperlite Cue Extraction](reports/figures/paperlite_cue_extraction.svg)

### 5. Realistic families show latent low-cue memorization before overt extraction
- Even though low-cue exact extraction stayed at zero in the mixed-family `budget1` matrix, low-cue target logprob moved in the expected direction in the repeated/fuzzy conditions.
- In the realistic-family matrix, `C2` low-cue member mean target logprob was slightly stronger than nonmember, and `C3` improved that further. The clearest slice was the event family inside `C3`, where member low-cue mean target logprob was stronger than nonmember.
- A focused `C3` event-only rerun strengthened that latent gap further, reaching mean max target logprob `-2.2780` overall and member/nonmember separation in favor of members, but still did not produce exact event-field extraction under long cold decoding.
![Paperlite Low-Cue Logprob](reports/figures/paperlite_lowcue_logprob.svg)
![Deep Event Audits](reports/figures/event_deep_logprob.svg)

## Scientific Interpretation
- `RQ1 cue validity`: answered in the pilot sense. Cue filtering changes the picture dramatically. High-cue generation can look extractive while low-cue exact extraction largely disappears on realistic families at this local scale.
- `RQ2 mechanistic separation`: partially answered. We have clear mechanistic separation on canaries, but not yet on realistic-family `C2/C3` because those runs did not yield enough successful low-cue extractions.
- `RQ3 causal mechanism`: partially answered. Patching and probe evidence show that internal states matter, but the current targeted mitigation is not yet clean enough to make a strong efficiency claim.
- `RQ4 training provenance`: answered for exact canaries, not yet for fuzzy realistic families.
- `RQ5 targeted mitigation`: partially answered through canary removal validation, but not yet through a low-utility targeted intervention that beats random inside `C2/C3` realistic-family runs.

## Storage and Scale Engineering Findings
- Streaming generation to JSONL removed the largest in-memory bottleneck during long audits.
- Activation caches were reduced to `resid_post` and `float16`, which kept mechanistic artifacts compact.
- Provenance candidate pools were corrected to include real distractors and trimmed to a lean exact-canary validation subset, which made attribution both faster and more trustworthy.

## What Is Still Missing For The Full Paper Claim
- Multi-seed `C0-C4` runs at `410M`.
- A realistic-family condition that crosses from low-cue latent signal into actual low-cue extraction.
- Mechanistic `C2/C3` analysis on successful realistic-family low-cue extractions.
- Fuzzy-cluster provenance and removal validation on `C3` outside the canary setting.
- A targeted privacy intervention with substantially smaller utility cost.
- The paperlite validation reports still flag a small residual low-cue overlap caveat (`12` prompts crossing a heuristic sensitive-substring threshold), so the prompt set is good enough for pilots but not yet perfectly clean.

## Key Artifact Index
- Canary scale summary: `reports/canary_stress_scale_summary.md`
- Canary removal validation: `reports/canary_stress_removal_validation.md`
- 410M strong canary behavioral report: `reports/canary_stress_pythia_410m_seed1_strong_long_behavioral.md`
- Paperlite `C0-C4` reports: `reports/paperlite_pythia_410m_*_behavioral.md`
- Exact-canary provenance: `outputs/provenance/canary_stress_gradient_similarity_finalnorm_v2.jsonl`
- Mechanistic artifacts: `outputs/mech/canary_stress_pythia_160m/`

## Bottom Line
We now have a real scientific pilot: strong canary extraction at 160M and 410M, provenance that beats random on exact canaries, removal validation that is causally meaningful, mechanistic evidence that localizes low-cue canary extraction, and a real `410M C0-C4` realistic-family matrix that shows cue-driven behavior plus low-cue latent memorization. What we do not yet have is a full paper-grade realistic-family low-cue extraction result with matching mechanistic and provenance validation.
