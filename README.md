# SIT-MSF Suicide Risk and Factor Identification

This repository contains the **SIT-MSF** system for a two-part shared task on Reddit posts:

1. **Subtask 1 — suicide-risk detection and evidence extraction**
2. **Subtask 2 — multi-label suicide-factor identification**

The repository records the complete development path, including successful baselines, the submitted system, incomplete experiments, and failed experiments. The current recommended system keeps the transformer predictions for Subtask 1 and uses the V2/V3 ensemble for Subtask 2.

> This is competition research code, not a clinical assessment system. Its predictions must not be used to make real-world safety or medical decisions.

## Competition tasks and scoring

### Subtask 1

For each post, predict one risk level:

- `Indicator` — no explicit suicide mention
- `Ideation` — explicit suicidal expression without a plan
- `Behavior` — suicidal expression with self-harm or a suicide plan
- `Attempt` — a recent or past suicide attempt is mentioned

The system must also extract short, verbatim evidence spans from the original post.

Subtask 1 contributes:

- 40% risk-level Weighted F1
- 30% evidence Phrase F1

### Subtask 2

Predict zero or more labels from the 24-factor taxonomy:

```text
mental health issues
physical health/characteristic
substance use
hopelessness
emotion dysregulation
low self-esteem
poor school performance
low socio-economic status
interpersonal violence
prior self-harm or suicidal thought/attempt
poor social support
interpersonal difficulty
dysfunctional family
exposure to others' suicide
stressful life event
traumatic experience
cognitive deficits
suicide means (with access)
sexual orientation related issues
social support
coping strategy
psychological capital
sense of responsibility
meaning in life
```

Subtask 2 is evaluated with Macro F1 and contributes 30% of the composite score.

The complete score is:

```text
0.4 × risk Weighted F1 + 0.3 × evidence Phrase F1 + 0.3 × factor Macro F1
```

## Data

The current workbooks contain:

| File | Rows | Users | Purpose |
|---|---:|---:|---|
| `train.xlsx` | 1,635 | 153 | Training and user-grouped validation |
| `leaderboard.xlsx` | 378 | 36 | Final prediction only |

`train.xlsx` contains:

```text
row_id
anon_user_id
post_id
post
suicide risk
evidence for suicide risk level
factors
```

`leaderboard.xlsx` contains no gold labels and must not be used to calculate validation scores.

Posts are split by `anon_user_id` during validation. Posts from one author must not appear in both the training and validation partitions because that would leak writing style and personal history.

The `factors` column sometimes repeats a factor in one row. The prediction problem is multi-label, so the main pipelines convert repeated values into one binary label.

## Current official result

The documented SIT-MSF submission obtained:

| Subtask 1 | Subtask 2 | Composite | Rank at that time |
|---:|---:|---:|---:|
| 0.7159 | 0.4409 | 0.6334 | 34 |

The Subtask 2 leaderboard score came from the V3 submission path. It improved the earlier Subtask 2 score of approximately 0.3099.

## Recommended system

The most reliable reproducible combination is:

- **Subtask 1:** preserve the risk and evidence predictions from the trained multi-task transformer.
- **Subtask 2:** V2 sparse calibrated ensemble followed by the V3 semantic and longitudinal ensemble.
- **Final CSV:** combine both parts with `v3_build_submission.py` or `v3_build_submission.ipynb`.

V4 is incomplete, V5 failed, V6 has only a small and uncertain local improvement, and the current V7 zero-shot LLM test is not good enough for a full run. They are retained as research records and are not part of the recommended submission.

## Method history

### Early TF-IDF baseline

The first system used word and character TF-IDF with logistic regression.

Advantages:

- Runs quickly on a CPU.
- Provides a leakage-safe baseline.
- Character n-grams work well with spelling errors and informal Reddit text.

Disadvantages:

- Limited understanding of context, intent, time, and negation.
- Rule-based evidence extraction misses unfamiliar expressions.
- Rare-factor thresholds are unstable.

Relevant files:

- `colab.ipynb`
- `improved_competition_pipeline.py`
- `improved_competition_evaluation.ipynb`

### Initial Subtask 1 RoBERTa pipeline

`subtask1_local_pipeline.ipynb` introduced `roberta-base` for four-class suicide-risk prediction while keeping evidence extraction mainly rule based.

RoBERTa improved contextual risk classification over the TF-IDF baseline, but a sequence classifier does not directly learn evidence boundaries. The 512-token input limit can also truncate longer posts.

Relevant files:

- `subtask1_local_pipeline.ipynb`
- `subtask1_statistics_analysis.ipynb`
- `subtask1_plan.md`
- `technique_overview.md`

### Multi-task ModernBERT transformer

`transformer_multitask.py` uses one ModernBERT encoder with three prediction heads:

1. four-class risk classification;
2. token-level evidence extraction;
3. 24-label factor classification.

The saved single-fold experiment reached approximately:

| Metric | Local result |
|---|---:|
| Risk Weighted F1 | 0.7400 |
| Evidence Phrase F1 | 0.7227 |
| Factor Macro F1 | 0.3642 |

This became the strongest Task 1 direction. Task 2 remained weak because the rare factor labels provided too little training signal relative to the other objectives.

Relevant files:

- `transformer_multitask.py`
- `transformer_multitask_training.ipynb`

### V2 — sparse calibrated factor ensemble

V2 changed Task 2 to a conservative CPU-friendly ensemble:

- word TF-IDF;
- character TF-IDF;
- one balanced logistic-regression classifier per factor;
- auxiliary risk probabilities;
- a meta-classifier;
- per-label probability blending;
- prevalence-based output quotas;
- five-fold user-grouped validation.

Saved local calibrated Macro F1: **approximately 0.4406**.

V2 was more stable than the multi-task factor head and substantially reduced rare-label overprediction.

Relevant files:

- `v2_factor_ensemble.py`
- `v2_factor_ensemble_training.ipynb`

### V3 — semantic and longitudinal ensemble

V3 added frozen ModernBERT embeddings without expensive end-to-end transformer training. Its features include:

- current-post embedding;
- previous and next post context;
- average user-profile embedding;
- difference from the user profile;
- position in the user's post sequence;
- V2 factor probabilities.

Lightweight per-factor classifiers are trained on these features, then blended with V2. Embeddings are cached so they do not need to be calculated again.

Results:

| Evaluation | Macro F1 |
|---|---:|
| Standard saved local run | about 0.4491 |
| Additional `C=1.5` local run | about 0.4547 |
| Official leaderboard | 0.4409 |

The reasonably small local-to-leaderboard gap makes V3 the most trustworthy Task 2 method in this repository.

Relevant files:

- `v3_fast_semantic.py`
- `v3_fast_semantic_training.ipynb`
- `v3_build_submission.py`
- `v3_build_submission.ipynb`

### V4 — full ModernBERT factor fine-tuning

V4 tried a heavier Task 2 model with:

- full ModernBERT encoder fine-tuning;
- label-description attention;
- multi-label asymmetric loss;
- ranking loss;
- auxiliary suicide-risk prediction;
- long-context input;
- five user-grouped folds;
- fold checkpointing and resume support.

Four completed folds peaked at approximately 0.3569, 0.3805, 0.3828, and 0.3591. The complete five-fold experiment was not finished because the completed folds did not outperform V2 or V3.

Status: **incomplete experiment; not recommended for submission**.

Relevant files:

- `v4_finetuned_factor.py`
- `v4_finetuned_factor_training.ipynb`

### V5 — DeBERTa rare-label experiment

V5 specifically targeted factor imbalance using:

- `microsoft/deberta-v3-base`;
- user-grouped folds balanced across factor counts;
- capped weighted oversampling of rare-factor posts;
- asymmetric/focal loss;
- higher positive weights for rare labels;
- a small auxiliary risk loss;
- conservative keyword rules for selected rare factors;
- optional blending with V2, V3, and V4 predictions;
- resumable folds.

This was oversampling, not synthetic text generation. It repeated rare examples more often during training.

Partial validation was only approximately **0.26–0.28 Macro F1**. With as few as 8–16 examples for some labels, repeating the same posts and users increased overfitting rather than generalization.

Status: **failed and intentionally stopped**.

Relevant files:

- `v5_deberta_rare_factor.py`
- `v5_deberta_rare_factor_training.ipynb`

### V6 — fast Linear SVM ensemble

V6 returned to a lightweight model and added:

- balanced Linear SVM classifiers;
- word and character TF-IDF;
- three regularization values: `0.03`, `0.10`, and `0.30`;
- per-factor conversion of SVM margins into ranks;
- per-factor blending with the strongest saved V3 run;
- nested user-grouped evaluation;
- prevalence calibration.

Results:

| Evaluation | Macro F1 |
|---|---:|
| V3 fixed quota in the V6 run | 0.4540 |
| V6 selected fixed blend | 0.4621 |
| V6 selected calibrated blend | 0.4611 |
| V6 stricter nested blend | 0.4430 |

The selected score improved slightly, but the nested score was lower than V3. The apparent improvement may be caused by per-label selection overfitting.

Status: **fast experimental ensemble; not leaderboard-confirmed**.

Relevant files:

- `v6_svm_factor_ensemble.py`
- `v6_svm_factor_ensemble_training.ipynb`

### V7 — instruction-LLM teacher

Statistical analysis showed that calibration alone could not raise V3 to 0.60. Even an optimistic oracle choice of output quotas produced only about 0.467 Macro F1 because correct rare-label examples were often ranked below incorrect examples.

V7 therefore tested external semantic knowledge from an instruction-tuned Qwen3-8B model. It added:

- definitions and distinctions for all 24 factors;
- JSON output containing label, confidence, and evidence;
- exact verbatim-evidence validation;
- prompt-injection protection;
- one-record-at-a-time resumable JSONL caching;
- 4-bit MLX inference on Apple silicon;
- 4-bit BitsAndBytes inference on Google Colab;
- a planned per-label rank blend with V3.

V7 is a zero-shot teacher; Qwen is not fine-tuned on the training set. The initial 10-post Colab smoke test ran correctly, but it exposed serious problems:

- one post with many gold factors received no predictions;
- other posts received factors that were reasonable from the text but absent from the competition annotations;
- most reported confidences were `1.0`, creating tied rankings;
- the estimated T4 runtime for all 1,635 posts was about 11 hours.

Status: **working prototype, but the current smoke test does not justify a full run or submission**.

Relevant files:

- `v7_llm_factor_teacher.py`
- `v7_llm_factor_teacher_training.ipynb`
- `v7_colab_llm_teacher.ipynb`
- `TASK2_STATISTICAL_DIAGNOSIS.md`

## Why reaching 0.60 Task 2 Macro F1 is difficult

Task 2 uses Macro F1, so a factor with 8 positive posts has the same weight as hopelessness with 745 positive posts. The rarest labels include:

| Factor | Positive posts | Distinct users |
|---|---:|---:|
| sexual orientation related issues | 8 | 7 |
| exposure to others' suicide | 14 | 11 |
| poor school performance | 16 | 13 |
| substance use | 33 | 19 |
| cognitive deficits | 33 | 21 |
| meaning in life | 45 | 31 |

The six labels with fewer than 50 examples averaged only about 0.195 F1 in the V3 diagnosis. Threshold tuning, oversampling, heavier encoders, and user smoothing did not solve the underlying ranking errors.

See `TASK2_STATISTICAL_DIAGNOSIS.md` for the full analysis.

## Installation

### Local environment

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-local.txt
jupyter notebook
```

The code automatically uses CUDA when available, Apple MPS on a compatible Mac, and CPU otherwise. Large transformer experiments will be much slower on CPU.

### Google Colab for V7

1. Upload `v7_colab_llm_teacher.ipynb` to Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Upload `train.xlsx` to `/content/train.xlsx`.
4. Run the notebook from the beginning.
5. Run only the 10-post smoke test first.

The V7 notebook stores its resumable cache in:

```text
Google Drive/MyDrive/SIT_MSF/v7_colab_teacher/teacher_train.jsonl
```

The current V7 smoke test is not strong enough to recommend the full pass.

## Reproducing the recommended Task 2 system

Run the notebooks from the repository root in this order:

1. `v2_factor_ensemble_training.ipynb`
2. `v3_fast_semantic_training.ipynb`
3. `v3_build_submission.ipynb`

V2 produces the sparse out-of-fold probabilities and calibration artifacts needed by V3. V3 produces cached semantic embeddings, out-of-fold probabilities, blend choices, and submission rates.

## Building the final submission

Before running the builder, the preserved Task 1 CSV must exist at:

```text
outputs/SIT-MSF.csv
```

It must contain exactly:

```text
row_id,risk_level,evidence,factors
```

The builder preserves `risk_level` and `evidence`, replaces the factor predictions with V3 results, verifies row IDs and risk labels, and checks that every evidence span appears verbatim in its post.

Notebook method:

```text
Open and run v3_build_submission.ipynb
```

Command-line method:

```bash
python v3_build_submission.py
```

Expected output:

```text
outputs/v3_submission/SIT-MSF.csv
```

Do not submit a V4, V5, V6, or V7 file unless a complete user-grouped evaluation clearly improves over V3.

## Project file guide

### Data and documentation

| File | Description |
|---|---|
| `train.xlsx` | Labeled training data for both subtasks. |
| `leaderboard.xlsx` | Unlabeled prediction data. Never use it for validation. |
| `requirements-local.txt` | Python dependencies for local notebooks and scripts. |
| `README.md` | Main project instructions and experiment summary. |
| `EXPERIMENT_REPORT.md` | Detailed technical report, results, advantages, disadvantages, and method history. |
| `TASK2_STATISTICAL_DIAGNOSIS.md` | Rare-label, user-history, calibration, and score-ceiling analysis. |
| `subtask1_plan.md` | Original Task 1 development plan; some historical values may be outdated. |
| `technique_overview.md` | Simple explanation of the early Task 1 method. |

### Early notebooks and baselines

| File | Description | Status |
|---|---|---|
| `colab.ipynb` | Early combined exploration and baseline experiments. | Historical |
| `subtask1_statistics_analysis.ipynb` | Risk, evidence, and text statistics. | Analysis |
| `subtask1_local_pipeline.ipynb` | Initial RoBERTa-oriented Task 1 pipeline. | Historical baseline |
| `subtask2_factor_statistics_baseline.ipynb` | Initial Task 2 statistics and baseline. | Historical baseline |
| `improved_competition_pipeline.py` | Leakage-safe TF-IDF pipeline for both tasks. | Fast CPU baseline |
| `improved_competition_evaluation.ipynb` | Runs and evaluates the improved sparse baseline. | Baseline evaluation |

### Multi-task transformer

| File | Description | Status |
|---|---|---|
| `transformer_multitask.py` | ModernBERT model for risk, token evidence, and factors. | Main Task 1 research code |
| `transformer_multitask_training.ipynb` | Training, validation, checkpoint inspection, and inference workflow. | Task 1 source |

### Task 2 versions

| File | Description | Status |
|---|---|---|
| `v2_factor_ensemble.py` | TF-IDF, balanced logistic models, risk meta-features, and prevalence calibration. | Recommended foundation |
| `v2_factor_ensemble_training.ipynb` | V2 user-grouped evaluation. | Run before V3 |
| `v3_fast_semantic.py` | Frozen ModernBERT embeddings and longitudinal features. | Recommended Task 2 model |
| `v3_fast_semantic_training.ipynb` | V3 evaluation and cached embedding generation. | Best validated Task 2 notebook |
| `v3_build_submission.py` | Trains full V2/V3 factors and combines them with preserved Task 1 predictions. | Recommended builder |
| `v3_build_submission.ipynb` | Notebook interface for the submission builder. | Recommended builder |
| `v4_finetuned_factor.py` | Fully fine-tuned ModernBERT factor model. | Incomplete |
| `v4_finetuned_factor_training.ipynb` | Resumable V4 training. | Incomplete |
| `v5_deberta_rare_factor.py` | DeBERTa, rare sampling, asymmetric loss, and rare-label rules. | Failed |
| `v5_deberta_rare_factor_training.ipynb` | Resumable V5 training and ensemble evaluation. | Stopped |
| `v6_svm_factor_ensemble.py` | Balanced Linear SVM rankings blended with V3. | Experimental |
| `v6_svm_factor_ensemble_training.ipynb` | Fast V6 grouped evaluation. | Experimental |
| `v7_llm_factor_teacher.py` | Local MLX Qwen teacher and planned V3 reranker. | Prototype |
| `v7_llm_factor_teacher_training.ipynb` | Mac smoke test, cache generation, and evaluation. | Prototype |
| `v7_colab_llm_teacher.ipynb` | Self-contained Colab 4-bit Qwen teacher. | Prototype; stop after smoke test |

## Generated artifacts

The ignored `outputs/` directory can contain several gigabytes of:

- transformer checkpoints;
- cached ModernBERT embeddings;
- out-of-fold probabilities;
- fitted sparse models;
- per-label metrics;
- calibration settings;
- diagnostic JSON files;
- intermediate and final submissions.

These artifacts are not stored in Git because they are large and machine-specific. Do not delete them if you want to reuse trained checkpoints or avoid repeating embedding generation.

## Reproducibility and reporting rules

- Use only `train.xlsx` for model selection and validation.
- Group validation by `anon_user_id`.
- Keep leaderboard predictions separate from local evaluation.
- Report whether a score is single-fold, complete out-of-fold, selected on the same data, or official leaderboard performance.
- Do not present V4 as complete or V5 as successful.
- Treat V6's selected improvement cautiously because its nested result is weaker.
- Treat V7 as an unfinished teacher experiment, not a submission system.
- Keep evidence spans short and copied exactly from the source post.

## Conclusion

The experiments show that greater model complexity did not automatically improve this small, long-tailed dataset. Transformers were valuable for contextual Task 1 prediction, while the conservative V2/V3 combination gave the most dependable Task 2 generalization. V4 was expensive and incomplete, V5 overfit rare examples, V6 produced only an uncertain small gain, and V7 did not match the annotation style well enough in its initial smoke test.

For reproduction or submission, use the preserved multi-task-transformer Task 1 predictions together with the V2/V3 Task 2 pipeline.
