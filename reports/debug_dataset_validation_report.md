# Dataset Validation Report

## Exact Field Uniqueness
- Checked 443 sensitive values.
- Collisions found: 0

## Non-member Contamination
- Non-member sensitive values found in training corpus: 0

## Background Collision
- Canary/background collisions: 0

## Duplicate Condition Checks
- Non-exact duplicate collisions inside fuzzy/redacted docs: 0

## Cue Prompt Leakage
- Low-cue prompts with sensitive substring overlap >= 6: 12

## Summary
- Records by split/family: {('train_member', 'identity'): 25, ('train_member', 'event'): 15, ('train_member', 'account'): 8, ('val_member', 'identity'): 9, ('val_member', 'account'): 1, ('val_member', 'event'): 2, ('test_member', 'identity'): 8, ('test_member', 'account'): 5, ('test_member', 'event'): 7, ('test_nonmember', 'event'): 9, ('test_nonmember', 'identity'): 5, ('test_nonmember', 'account'): 6, ('train_member', 'canary'): 10}
- Condition: C3_fuzzy_5x
- Synthetic docs in corpus: 81
