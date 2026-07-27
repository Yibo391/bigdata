# Subtask 1 Plan: Explainable Suicide Risk Detection

## 1. Competition Goal

Subtask 1 requires two predictions for each Reddit post:

1. Predict the author's suicide risk level.
2. Extract short evidence spans from the original post that support that risk-level prediction.

The allowed risk labels are exactly:

- `Indicator`
- `Ideation`
- `Behavior`
- `Attempt`

The official submission format also includes a `factors` column for Subtask 2. Since this plan focuses only on Subtask 1, the system will not model suicide factors. The `factors` column will be filled with `[]` for every leaderboard row.

The goal is to produce a valid CSV submission that performs well on:

- Weighted F1 for suicide risk classification.
- Phrase F1 for evidence extraction.

## 2. Dataset Overview

The folder contains two datasets:

- `train.xlsx`: labeled training data.
- `leaderboard.xlsx`: unlabeled leaderboard data for prediction.

Observed dataset sizes:

- `train.xlsx`: 1,635 labeled rows.
- `leaderboard.xlsx`: 999 rows.

Important columns in `train.xlsx`:

- `row_id`: unique post identifier.
- `anon_user_id`: anonymized user identifier.
- `post_id`: post number for that user.
- `post`: Reddit post text.
- `suicide risk`: gold risk-level label.
- `evidence for suicide risk level`: gold evidence spans separated by semicolons.
- `factors`: Subtask 2 labels, not used for this Subtask 1 plan.

Important columns in `leaderboard.xlsx`:

- `row_id`
- `anon_user_id`
- `post_id`
- `post`

The training labels need normalization because they are inconsistent. Examples include `ideation`, `Ideation`, `Ideation `, `behavior`, and `Behavior`. All labels must be converted to the exact official labels before training.

## 3. Environment Preparation

Use a local, reproducible Python pipeline. The preferred packages are:

- `pandas`
- `openpyxl`
- `numpy`
- `scikit-learn`
- `torch`
- `transformers`
- `datasets`

Implementation steps:

1. Create or reuse a Python environment.
2. Install missing packages, especially `openpyxl`, because it is required for reading `.xlsx` files with pandas.
3. Confirm CUDA is available for transformer training.
4. Set fixed random seeds for Python, NumPy, PyTorch, and Hugging Face training.
5. Store generated files in clear folders such as:
   - `outputs/`
   - `models/`
   - `reports/`

Recommended reproducibility settings:

- Use a fixed seed such as `42`.
- Save the exact model name.
- Save validation metrics after every experiment.
- Save the final submission file with a timestamp or version number.

## 4. Data Cleaning

Data cleaning should happen before any modeling.

Steps:

1. Load `train.xlsx` and `leaderboard.xlsx`.
2. Rename columns internally to simpler names:
   - `suicide risk` -> `risk_level`
   - `evidence for suicide risk level` -> `evidence`
3. Normalize `risk_level`:
   - Strip whitespace.
   - Convert to lowercase for matching.
   - Map back to exact official labels.
4. Validate that every training label is one of:
   - `Indicator`
   - `Ideation`
   - `Behavior`
   - `Attempt`
5. Clean evidence spans:
   - Split by semicolon.
   - Trim whitespace from each span.
   - Remove empty spans.
   - Remove duplicate spans within the same row.
   - Keep spans as verbatim text from the post whenever possible.
6. Keep the key columns:
   - `row_id`
   - `anon_user_id`
   - `post`
   - `risk_level`
   - `evidence`
7. Check for problems:
   - Missing `row_id`.
   - Missing or empty `post`.
   - Missing `risk_level`.
   - Invalid labels.
   - Evidence spans that do not appear in the post.

Do not silently drop many rows. If rows have problems, log them and inspect the issue.

## 5. Validation Strategy

The validation split should be grouped by `anon_user_id`, not randomly split by row.

Reason:

The same Reddit author may have multiple posts. A random row split could put one user's posts in both training and validation. That would make validation look better than the actual leaderboard performance because the model could learn user-specific writing style.

Validation steps:

1. Group rows by `anon_user_id`.
2. Split users into training users and validation users.
3. Ensure no `anon_user_id` appears in both train and validation.
4. Try to preserve class balance as much as possible.
5. Confirm all four labels appear in validation.

Primary classification metric:

- Weighted F1.

Evidence metric:

- Approximate Phrase F1, following the challenge rule:
  - Normalize case.
  - A predicted phrase is correct if it contains a gold phrase or is contained by a gold phrase.
  - Do not reward evidence that is too long.
  - Each predicted span should match at most one gold span.
  - Each gold span should match at most one predicted span.

The local phrase metric does not need to be perfect at first, but it should be close enough to compare experiments.

## 6. Risk-Level Classifier

Train a local transformer classifier for risk-level prediction.

Recommended first model:

- `microsoft/deberta-v3-base`

Backup model:

- `roberta-base`

Model input:

- Full Reddit post text.

Model output:

- One of the four risk labels:
  - `Indicator`
  - `Ideation`
  - `Behavior`
  - `Attempt`

Training approach:

1. Tokenize posts with truncation.
2. Use a maximum sequence length such as 512 tokens.
3. Fine-tune for a small number of epochs, such as 3 to 5.
4. Evaluate after each epoch.
5. Save the checkpoint with the best validation weighted F1.

Class imbalance handling:

The `Attempt` class is much smaller than the others, so the model may underpredict it. Use one or more of:

- Class-weighted loss.
- Oversampling of minority classes.
- More careful thresholding or rule-based correction after model prediction.

Suggested first training configuration:

- Seed: `42`
- Max length: `512`
- Epochs: `3`
- Batch size: as large as GPU memory allows
- Learning rate: `2e-5`
- Metric for best model: validation weighted F1

Expected error focus:

- `Ideation` vs `Behavior`
- `Behavior` vs `Attempt`
- Missed `Attempt` examples

## 7. Evidence Extraction

Evidence extraction must output short spans copied verbatim from the original post. This is critical because the challenge evaluates phrase overlap and penalizes overly long evidence.

Use a hybrid evidence extractor:

1. Generate candidate spans.
2. Score candidates.
3. Select the best short spans.
4. Join selected spans with semicolons.

Candidate span generation:

- Split the post into sentences.
- Split long sentences into clauses using punctuation such as commas, semicolons, dashes, and conjunctions.
- Create short word windows around suicide-risk keywords.
- Include exact matches or near matches of common training evidence phrases.

Candidate scoring signals:

- Keyword matches.
- Risk-level-specific keyword matches.
- Similarity to evidence phrases from the training data.
- Whether the candidate is compatible with the predicted risk label.
- Candidate length.
- Candidate position in the post.

General selection rules:

- Return 1 to 3 spans for most non-Indicator predictions.
- Prefer concise spans over full sentences.
- Avoid spans that are more than about 20 to 30 words unless validation proves they help.
- Remove duplicate or near-duplicate spans.
- Preserve the original text exactly as it appears in the post.
- Separate multiple spans with `; `.

Evidence tuning:

Tune these parameters on validation:

- Number of spans per post.
- Maximum span length.
- Minimum score threshold.
- Keyword weights.
- Whether `Indicator` should return empty evidence or short non-suicide evidence.

## 8. Risk-Specific Evidence Rules

### Indicator

Definition:

The post contains no explicit mentions of suicide.

Evidence strategy:

- Be conservative.
- Prefer empty evidence if validation shows that works best.
- If evidence is required or useful, select a short phrase that explains distress without explicit suicide intent.

Possible signals:

- Depression or sadness without suicidal intent.
- General hopelessness without self-harm language.
- Requests for support without explicit suicide references.

### Ideation

Definition:

The post contains explicit suicidal expressions but no suicidal plan.

Evidence strategy:

- Select direct suicidal thoughts or wishes.
- Avoid upgrading to `Behavior` unless there is a plan, method, or preparation.

High-value evidence patterns:

- `want to die`
- `wanna die`
- `kill myself`
- `suicidal thoughts`
- `do not want to live`
- `wish I was dead`
- `better off dead`
- `not wake up`

### Behavior

Definition:

The post contains explicit suicidal expressions and self-harm or suicidal plan.

Evidence strategy:

- Select text showing planning, method, means, preparation, or imminent behavior.
- This class should capture more than ideation but less than an actual attempt.

High-value evidence patterns:

- Mentioning a suicide method.
- Mentioning a plan.
- Mentioning access to means.
- Saying when or how they will do it.
- Preparing notes or possessions.
- Self-harm intent with suicidal context.

Examples of useful signals:

- `I have a plan`
- `tonight`
- `pills`
- `knife`
- `rope`
- `bridge`
- `jump`
- `overdose`
- `cut`
- `hang`

### Attempt

Definition:

The post contains explicit mentions of recent or past suicide attempts.

Evidence strategy:

- Select text proving an attempt already happened.
- Past-tense attempt language is especially important.
- Do not confuse planning with attempt unless the post clearly says the action happened.

High-value evidence patterns:

- `tried to kill myself`
- `attempted suicide`
- `I overdosed`
- `survived`
- `woke up in the hospital`
- `after my attempt`
- `last time I tried`
- `I cut too deep`
- `I hanged myself`

Attempt should receive special attention because it is the smallest class and may need rule-based boosts.

## 9. Submission Generation

For each row in `leaderboard.xlsx`:

1. Read `row_id` and `post`.
2. Predict `risk_level`.
3. Extract evidence spans from the post.
4. Set `factors` to `[]`.
5. Write the output row.

The final CSV must contain exactly these columns:

- `row_id`
- `risk_level`
- `evidence`
- `factors`

Example:

```csv
row_id,risk_level,evidence,factors
P00008,Ideation,"want to die; cannot continue",[]
```

Save the file as:

```text
YourTeamName.csv
```

Replace `YourTeamName` with the real team name before submission.

## 10. Quality Checks

Before submitting, run a submission validator.

Required checks:

1. Every row from `leaderboard.xlsx` appears exactly once.
2. No extra `row_id` values are included.
3. The CSV has exactly four columns:
   - `row_id`
   - `risk_level`
   - `evidence`
   - `factors`
4. Every `risk_level` is valid.
5. Every `factors` value is `[]`.
6. Evidence spans are separated by semicolons.
7. Every non-empty evidence span appears verbatim in the original post.
8. Evidence spans are not excessively long.
9. The CSV can be opened and read back without broken quoting.
10. There are no missing predictions.

If a validation check fails, fix the pipeline rather than editing the CSV manually.

## 11. Improvement Loop

After the first complete validation run:

1. Review the weighted F1 score.
2. Review the confusion matrix.
3. Review phrase precision, recall, and F1.
4. Inspect examples where the classifier is wrong.
5. Inspect examples where the risk prediction is correct but evidence is poor.

Priority error categories:

- `Ideation` predicted instead of `Behavior`.
- `Behavior` predicted instead of `Attempt`.
- `Indicator` predicted for posts with indirect suicidal language.
- Long evidence spans that lower phrase precision.
- Very short evidence spans that miss key details.

Possible improvements:

- Add rule-based overrides for high-confidence `Attempt`.
- Add rule-based overrides for high-confidence `Behavior`.
- Train a TF-IDF logistic regression model and ensemble it with the transformer.
- Tune evidence keyword weights.
- Tune number of evidence spans.
- Tune maximum evidence length.
- Add more risk-specific evidence phrase patterns from training data.

Recommended experiment order:

1. Build and validate a simple baseline.
2. Fine-tune the transformer classifier.
3. Add the first evidence extractor.
4. Tune evidence extraction on validation.
5. Add class-imbalance handling.
6. Add high-confidence rules for `Attempt` and `Behavior`.
7. Try model ensembling if time allows.
8. Generate the final leaderboard CSV.

## 12. Test Plan

Run these tests before final submission.

### Data-loading test

- Confirm `train.xlsx` loads successfully.
- Confirm `leaderboard.xlsx` loads successfully.
- Confirm row counts match expectations.

### Label-normalization test

- Confirm all normalized labels are valid.
- Confirm no labels are missing after normalization.
- Print normalized class counts.

### Validation-split test

- Confirm no `anon_user_id` appears in both train and validation.
- Confirm all four classes appear in validation.
- Print train and validation class distributions.

### Classifier test

- Train on the training split.
- Predict validation labels.
- Report weighted F1.
- Report per-class precision, recall, and F1.
- Print a confusion matrix.

### Evidence test

- Extract evidence for validation posts.
- Confirm predicted evidence spans are copied from the post.
- Report approximate Phrase F1.
- Print examples of good and bad evidence matches.

### Submission-format test

- Generate a draft CSV.
- Read the CSV back with pandas.
- Confirm row count equals leaderboard row count.
- Confirm required columns are present in the correct order.
- Confirm all labels are valid.
- Confirm all `factors` values are `[]`.
- Confirm no evidence formatting errors.

## 13. Assumptions

- The implementation will use a local reproducible pipeline.
- Paid API calls will not be used in the first version.
- Subtask 2 is intentionally skipped.
- The `factors` column will be included only to satisfy the submission schema.
- Evidence spans should be short and copied verbatim.
- The first deliverable should be a strong baseline, not a perfect final system.
- Later improvements can add ensembling, more advanced span extraction, or API-assisted review if needed.

## 14. Suggested File Structure

A practical project structure could be:

```text
/home/yibo/Desktop/Suicide/
  train.xlsx
  leaderboard.xlsx
  subtask1_plan.md
  src/
    prepare_data.py
    train_classifier.py
    extract_evidence.py
    validate_submission.py
    make_submission.py
  outputs/
    validation_metrics.json
    confusion_matrix.csv
    draft_submission.csv
  models/
    best_risk_classifier/
```

This structure keeps the original data separate from generated outputs and model files.

## 15. Final Deliverable

The final deliverable for Subtask 1 should be a CSV submission file with:

- One row per leaderboard post.
- A valid risk-level prediction.
- Short supporting evidence copied from the post.
- `[]` in the `factors` column.

The final workflow should be reproducible from scripts, so the CSV can be regenerated after any modeling improvement.
