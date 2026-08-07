# SIT-MSF Suicide Risk and Factor Identification

This repository contains the SIT-MSF solution for suicide-risk classification,
verbatim evidence extraction, and multi-label suicide-factor identification.

Read [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) for:

- the competition task and scoring;
- the purpose of every project file;
- the advantages and disadvantages of each method;
- why the project changed from the initial RoBERTa/TF-IDF approaches;
- the V2/V3 method used for the best submitted Task 2 result;
- the incomplete V4 experiment and failed V5 experiment;
- exact reproduction and submission steps.

The recommended Task 2 path is:

1. `v2_factor_ensemble_training.ipynb`
2. `v3_fast_semantic_training.ipynb`
3. `v3_build_submission.ipynb`

Large generated artifacts and checkpoints are intentionally excluded from Git.
