# SIT-MSF Competition System Report

## 1. Project summary

This repository contains the SIT-MSF system for two related suicide-risk prediction subtasks.

- **Subtask 1:** predict one risk level (`Indicator`, `Ideation`, `Behavior`, or `Attempt`) and extract short verbatim evidence spans.
- **Subtask 2:** predict all applicable labels from the 24-category suicide-factor taxonomy.
- **Composite score:** 40% risk classification, 30% evidence extraction, and 30% factor identification.

The current workbooks contain **1,635 labeled training posts from 153 users** and **378 unlabeled leaderboard posts from 36 users**. Validation is grouped by `anon_user_id` so that posts from one user do not appear in both training and validation.

The submitted leaderboard result discussed in this report is:

| System | Subtask 1 | Subtask 2 | Composite | Rank at the time |
|---|---:|---:|---:|---:|
| SIT-MSF V3 submission | 0.7159 | 0.4409 | 0.6334 | 34 |

The best reproducible submission path in this repository is currently:

- **Task 1:** preserve the risk and evidence predictions produced by the trained transformer pipeline.
- **Task 2:** V2 sparse ensemble plus V3 semantic and longitudinal features.
- **Final combination:** `v3_build_submission.py` or `v3_build_submission.ipynb`.

V4 was not completed, and V5 was stopped after weak partial validation results around 0.28. Neither should be presented as the final method.

V6 is a later lightweight experiment that blends balanced Linear SVM text rankings with V3. It improved the same-style selected local score to about 0.4611, but its stricter nested estimate was lower; it should therefore be treated as promising but not yet leaderboard-confirmed.

## 2. Why the methods changed

The project changed methods because local experiments exposed different weaknesses at each stage.

1. **The first TF-IDF baseline was fast but did not understand context.** It was useful for checking the data and submission format, but it confused semantically similar risk levels and factors.
2. **RoBERTa improved Task 1.** Fine-tuning a transformer increased risk-classification performance because the distinction between ideation, planning, and attempts depends on context.
3. **The joint ModernBERT model improved local Task 1 and evidence extraction, but Task 2 remained weak.** One encoder had to optimize three objectives, and the rare factor labels received too little signal.
4. **The first submitted Task 2 predictions overpredicted rare labels.** Local threshold selection was unstable, especially for factors with fewer than 20 positive posts. The first Task 2 leaderboard score was about 0.3099.
5. **V2 switched to a conservative sparse ensemble and prevalence calibration.** This was less fashionable than a large neural model, but it controlled rare-label output rates and produced a much more reliable local Macro F1 of 0.4406.
6. **V3 added semantic and user-history information without fully fine-tuning another transformer.** It combined frozen ModernBERT embeddings, neighboring posts, user-profile features, and V2 probabilities. Local Macro F1 was about 0.4491, and the leaderboard Task 2 score was 0.4409. The close local/leaderboard values suggest that this evaluation was reasonably calibrated.
7. **V4 tested full ModernBERT fine-tuning for Task 2.** Four completed folds peaked around 0.36–0.38 individually. This was not convincing enough to justify the cost, and the full five-fold result was not completed.
8. **V5 tested DeBERTa with rare-label sampling and asymmetric loss.** The first folds were around 0.26–0.28. Oversampling increased exposure to rare examples, but with only 8–16 positives for the rarest labels it mainly increased overfitting and optimization difficulty. V5 was therefore stopped and classified as a failed experiment.

The main lesson is that **more complex training did not automatically produce better generalization**. With only 153 users and a very long-tailed label distribution, the conservative V2/V3 ensemble was more dependable than V4/V5.

## 3. Data characteristics

Task 2 is difficult because Macro F1 gives every factor equal importance even though their training support is very different.

| Factor | Positive posts | Distinct users |
|---|---:|---:|
| sexual orientation related issues | 8 | 7 |
| exposure to others' suicide | 14 | 11 |
| poor school performance | 16 | 13 |
| substance use | 33 | 19 |
| cognitive deficits | 33 | 21 |
| meaning in life | 45 | 31 |
| hopelessness | 745 | 144 |

Simple random row splitting would leak user writing style and make validation too optimistic. Grouped validation is therefore essential. Even with grouped validation, the rarest labels make fold-level Macro F1 noisy.

The `factors` column may repeat the same label within one row. The modeling code converts repeated values to one binary label because the competition output is multi-label, not a count prediction. Some experimental code uses multiplicity only as a small confidence signal.

## 4. Method comparison

### 4.1 TF-IDF and logistic-regression baseline

Used in `colab.ipynb`, `improved_competition_pipeline.py`, and the early statistical notebooks.

**Advantages**

- Very fast on a laptop.
- Easy to inspect and reproduce.
- Character n-grams handle spelling mistakes and informal Reddit language.
- Strong enough to provide useful probability rankings for an ensemble.

**Disadvantages**

- Limited understanding of negation, time, intent, and long-distance context.
- Rare labels depend heavily on a few words and users.
- Thresholds selected on a small validation set can overfit.

### 4.2 RoBERTa Task 1 pipeline

The initial Task 1 plan used RoBERTa for four-class risk prediction, while evidence extraction remained largely rule based.

**Advantages**

- Better contextual understanding than TF-IDF.
- Stronger distinction between explicit ideation and more severe risk language.
- Straightforward fine-tuning setup.

**Disadvantages**

- Standard sequence classification does not directly solve evidence extraction.
- A 512-token limit can truncate a small number of long posts.
- A single validation split can give an unstable estimate.

### 4.3 ModernBERT multi-task transformer

Implemented in `transformer_multitask.py` and `transformer_multitask_training.ipynb`.

The model shares one encoder across:

1. risk classification;
2. token-level evidence extraction;
3. factor prediction.

The saved notebook reports a best single-fold local result of approximately 0.7400 risk Weighted F1, 0.7227 Phrase F1, and 0.3642 factor Macro F1. A later label-attention Task 2 experiment reached approximately 0.4725 on the same local fold, but that was not a complete user-grouped five-fold result.

**Advantages**

- One model learns related information across both subtasks.
- Token-level evidence predictions can remain verbatim.
- ModernBERT supports longer posts.
- Strong local Task 1 performance.

**Disadvantages**

- Expensive to train on a laptop.
- Multi-task loss weights can cause one task to dominate another.
- A strong score on one fold did not translate into a strong Task 2 leaderboard score.
- Model checkpoints are large and are intentionally excluded from Git.

### 4.4 V2 factor ensemble

Implemented in `v2_factor_ensemble.py` and `v2_factor_ensemble_training.ipynb`.

V2 uses word and character TF-IDF, one balanced logistic-regression classifier per factor, auxiliary risk probabilities, a meta-classifier, and prevalence-based output quotas. Its saved five-fold local calibrated Macro F1 is **0.4406**.

**Advantages**

- Fast, stable, and user-grouped.
- Character features work well on informal posts.
- Per-label prevalence calibration prevents catastrophic rare-label overprediction.
- Produces probabilities that combine well with semantic models.

**Disadvantages**

- Limited semantic understanding.
- Weak performance on the rarest labels.
- Quota tuning uses training prevalence, which may shift on the hidden test set.

### 4.5 V3 semantic and longitudinal ensemble

Implemented in `v3_fast_semantic.py` and `v3_fast_semantic_training.ipynb`.

V3 calculates frozen ModernBERT embeddings and adds the previous post, next post, user profile, deviation from the user profile, and post-position features. It trains lightweight factor classifiers and blends them per label with V2. The saved local calibrated Macro F1 is **0.4491**; an additional `C=1.5` run reached approximately **0.4547** locally. The submitted V3 system obtained **0.4409** on the leaderboard.

**Advantages**

- Adds meaning and longitudinal user information without expensive end-to-end fine-tuning.
- Reuses cached embeddings.
- Blending allows V2 to remain dominant for factors where sparse features are stronger.
- The local score was close to the leaderboard score.

**Disadvantages**

- Frozen embeddings cannot adapt fully to the taxonomy.
- User smoothing can spread an incorrect prediction across several posts.
- Rare factors still have too few independent users.
- Requires V2 artifacts before the final submission can be generated.

### 4.6 V4 fine-tuned ModernBERT experiment

Implemented in `v4_finetuned_factor.py` and `v4_finetuned_factor_training.ipynb`.

V4 added full encoder fine-tuning, label-description attention, ranking loss, an auxiliary risk objective, and five-fold resume support. Four completed folds had best prevalence-based Macro F1 values of approximately 0.3569, 0.3805, 0.3828, and 0.3591.

**Advantages**

- Directly adapts the encoder to the 24-factor taxonomy.
- Label descriptions provide semantic initialization.
- Resume support protects long runs.

**Disadvantages**

- Slow and storage intensive.
- The completed fold results did not outperform V2/V3.
- The full experiment was not completed, so it has no trustworthy final five-fold result.
- It should be described as exploratory, not as the submitted system.

### 4.7 V5 DeBERTa rare-factor experiment — failed

Implemented in `v5_deberta_rare_factor.py` and `v5_deberta_rare_factor_training.ipynb`.

V5 used DeBERTa-v3-base, multi-label-balanced user folds, capped rare-example sampling, asymmetric loss, and conservative rules for a few rare factors. Partial validation was only around **0.26–0.28**, so the run was stopped.

**Advantages**

- Better-balanced validation folds: every fold contained at least one example of every factor.
- Explicitly targeted the long-tail problem.
- Sampling was capped to reduce extreme duplication.
- Included safeguards for Apple-silicon MPS training.

**Disadvantages**

- Rare-example sampling amplified a very small number of users and phrases.
- Asymmetric loss and sampling together likely pushed optimization too strongly toward noisy positives.
- DeBERTa required additional tokenizer dependencies and Apple-specific attention settings.
- Partial scores were substantially below V2/V3.
- The experiment was stopped and must not be used for the final submission.

### 4.8 V6 lightweight SVM ensemble — experimental

Implemented in `v6_svm_factor_ensemble.py` and `v6_svm_factor_ensemble_training.ipynb`.

V6 trains balanced Linear SVM classifiers over word and character TF-IDF for three regularization values. It converts the margins into per-label ranks and selectively blends them with the strongest saved V3 semantic run. Training takes seconds rather than hours. In the saved local experiment, V3 scored approximately 0.4540 with fixed prevalence, while the selected V6 blend scored approximately 0.4621 fixed and 0.4611 calibrated.

**Advantages**

- Very fast and CPU-friendly.
- Adds margin-based rankings that differ from V2 logistic regression.
- Reuses V3 and requires no transformer fine-tuning.
- Improves the selected local validation score by roughly 0.7–0.8 Macro F1 points.

**Disadvantages**

- Linear SVM alone is weaker than V3.
- Per-label blend selection can overfit rare labels.
- The stricter nested blend score was about 0.4430, below V3, so the apparent improvement is not guaranteed to transfer to the leaderboard.
- It has not yet been used for an official submission.

## 5. File guide

### Data and environment

| File | Purpose | Notes |
|---|---|---|
| `train.xlsx` | Labeled training data for both subtasks | 1,635 rows; competition-provided data. |
| `leaderboard.xlsx` | Unlabeled data used only for final prediction | 378 rows; must not be used to calculate validation scores. |
| `requirements-local.txt` | Python package list | Install into a virtual environment before running notebooks. |
| `README.md` | Repository landing page | Links to this report and the recommended pipeline. |

### Early analysis and baselines

| File | Purpose | Status |
|---|---|---|
| `colab.ipynb` | Early end-to-end exploration, TF-IDF baseline, transformer experiments | Historical. |
| `subtask1_statistics_analysis.ipynb` | Risk-label, evidence, and text statistics | Useful analysis. |
| `subtask1_local_pipeline.ipynb` | Initial Task 1 RoBERTa-oriented pipeline | Historical Task 1 baseline. |
| `subtask2_factor_statistics_baseline.ipynb` | Initial Task 2 label statistics and baseline | Historical Task 2 baseline. |
| `improved_competition_pipeline.py` | Fast leakage-safe TF-IDF pipeline for both tasks | Useful CPU baseline. |
| `improved_competition_evaluation.ipynb` | Evaluates the improved sparse pipeline | Useful baseline evaluation. |

### Transformer and factor experiments

| File | Purpose | Status |
|---|---|---|
| `transformer_multitask.py` | ModernBERT multi-task model for risk, evidence, and factors | Main transformer research code. |
| `transformer_multitask_training.ipynb` | Trains and evaluates the multi-task model | Task 1 source and historical Task 2 experiment. |
| `v2_factor_ensemble.py` | Sparse calibrated Task 2 ensemble | Recommended Task 2 foundation. |
| `v2_factor_ensemble_training.ipynb` | Runs V2 five-fold evaluation | Run before V3. |
| `v3_fast_semantic.py` | Frozen semantic and longitudinal Task 2 model | Recommended Task 2 improvement. |
| `v3_fast_semantic_training.ipynb` | Runs V3 evaluation | Best validated Task 2 notebook. |
| `v4_finetuned_factor.py` | Fine-tuned ModernBERT factor experiment | Incomplete experiment. |
| `v4_finetuned_factor_training.ipynb` | Resumable V4 training | Incomplete; not final. |
| `v5_deberta_rare_factor.py` | DeBERTa and rare-label experiment | Failed experiment. |
| `v5_deberta_rare_factor_training.ipynb` | Resumable V5 training | Stopped at about 0.28; do not continue for the final system. |
| `v6_svm_factor_ensemble.py` | Lightweight Linear SVM and V3 factor ensemble | Experimental low-cost improvement. |
| `v6_svm_factor_ensemble_training.ipynb` | Runs V6 five-fold evaluation | Fast; no submission generation. |

### Submission and documentation

| File | Purpose | Status |
|---|---|---|
| `v3_build_submission.py` | Preserves existing Task 1 predictions and replaces factors with V3 predictions | Recommended final builder. |
| `v3_build_submission.ipynb` | Notebook interface for the V3 submission builder | Recommended when working in VS Code/Jupyter. |
| `subtask1_plan.md` | Original Task 1 plan | Historical; its old leaderboard row count is outdated. |
| `technique_overview.md` | Short explanation of the early Task 1 approach | Historical overview. |
| `EXPERIMENT_REPORT.md` | Complete method history and file guide | Current report. |

### Generated files

The local `outputs/` directory contains submissions, cached embeddings, metrics, and model checkpoints. It is about 4.8 GB and is excluded from Git. These files are generated by the notebooks and may contain machine-specific or very large artifacts. The lightweight numerical results needed to explain the project are recorded in this report.

## 6. Recommended reproduction order

For the documented best Task 2 path:

1. Create a Python environment and install `requirements-local.txt`.
2. Run `v2_factor_ensemble_training.ipynb`.
3. Run `v3_fast_semantic_training.ipynb`.
4. Ensure the preserved Task 1 file exists at `outputs/SIT-MSF.csv`.
5. Run `v3_build_submission.ipynb`.
6. Submit `outputs/v3_submission/SIT-MSF.csv`.

The builder validates the required columns, row identifiers, official risk labels, and verbatim evidence before writing the combined CSV.

V4 and V5 are retained for transparency and the required source-code submission, but they are not part of the recommended final pipeline.

## 7. Limitations and future work

- The rarest factors cannot be evaluated reliably from only 7–13 independent users.
- Synthetic augmentation should be used only inside a training fold and must be checked carefully for label drift. The V5 result shows that reweighting or oversampling alone is not sufficient.
- A better future direction is external domain pretraining or carefully verified human/LLM-assisted annotations, if competition rules allow external data.
- Threshold and quota selection should ideally use nested grouped validation.
- Task 1 evidence extraction should be evaluated with the exact official one-to-one Phrase F1 implementation on all folds.
- Final reports should distinguish local validation, partial experiments, and official leaderboard results.

## 8. Final conclusion

The method changed because each experiment answered a practical question. Transformers were valuable for Task 1, but Task 2 was dominated by limited users, severe label imbalance, and unstable rare categories. V2 and V3 gave the best balance of semantic information, calibration, speed, and leaderboard reliability. V4 was inconclusive, and V5 failed. The final documented system therefore keeps the strong Task 1 predictions and uses the V3 Task 2 ensemble.
