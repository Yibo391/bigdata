# Technique Overview

This project solves Subtask 1 in two parts:

1. Predict the suicide risk level of a Reddit post.
2. Extract short text phrases from the post as evidence.

The four risk levels are:

- `Indicator`
- `Ideation`
- `Behavior`
- `Attempt`

## 1. Data Loading

The code reads two Excel files:

- `train.xlsx`
- `leaderboard.xlsx`

`train.xlsx` has labels, so we use it for training and validation.

`leaderboard.xlsx` does not have labels, so we use it only for final prediction.

## 2. Label Cleaning

The training labels are not perfectly consistent. For example, the file may contain:

- `ideation`
- `Ideation`
- `Ideation `

The code cleans these labels and converts them into the exact official names:

- `Indicator`
- `Ideation`
- `Behavior`
- `Attempt`

This is important because the model needs clean labels to learn correctly.

## 3. Train/Validation Split

The code splits the training data into two parts:

- training data
- validation data

The split is about:

- 80% training
- 20% validation

The split is done by `anon_user_id`, not by random rows.

This means posts from the same user will not appear in both training and validation. This makes the validation score more realistic.

## 4. Baseline Model

The first model is a simple baseline:

```text
TF-IDF + Logistic Regression
```
0.5977

TF-IDF turns text into numbers by looking at important words and short word phrases.

Logistic Regression then learns which words are connected with each risk level.

This model is fast and easy to run, but it does not deeply understand sentence meaning.

## 5. Transformer Model

The stronger model is:

```text
roberta-base
```
0.6984985831497459
RoBERTa is a transformer language model. It understands text better than TF-IDF because it reads words in context.

The code fine-tunes RoBERTa on the training posts. Fine-tuning means we take a model that already understands English and train it further for this suicide-risk task.

The model input is:

```text
Reddit post text
```

The model output is one of:

```text
Indicator, Ideation, Behavior, Attempt
```

## 6. Class Imbalance Handling

The classes are not equally common.

For example, `Attempt` has fewer examples than the other labels.

To help with this, the code uses class weights during transformer training.

This tells the model to pay more attention to smaller classes, especially `Attempt`.

## 7. Validation Score

The code checks model performance on the validation set.

For risk classification, it uses:

```text
Weighted F1
```

Weighted F1 is useful because it considers both precision and recall, while also accounting for class size.

Your transformer result was about:

```text
0.746 weighted F1
```

This is much better than the simple baseline, which was about:

```text
0.598 weighted F1
```

So the transformer model is clearly better for risk-level prediction.

## 8. Evidence Extraction

The evidence extraction part is currently rule-based.

This means it does not use a trained evidence model yet.

The code looks for important suicide-risk phrases, such as:

- `want to die`
- `kill myself`
- `overdose`
- `pills`
- `goodbye`
- `tried to kill myself`
- `attempted suicide`

It then selects short parts of the original post that contain these signals.

The evidence must be copied from the original post because the competition checks phrase overlap.

## 9. Evidence Score

The code estimates evidence quality with an approximate Phrase F1 score.

This checks whether the predicted evidence phrase overlaps with the gold evidence phrase.

Right now, the evidence score is low because the evidence extractor is simple.

The classifier is trained, but the evidence extractor is mostly keyword rules.

So the current weak point is evidence extraction, not risk classification.

## 10. Final Prediction

For the leaderboard file, the code:

1. Predicts the risk level.
2. Extracts evidence phrases.
3. Fills `factors` as `[]` because Subtask 2 is skipped.
4. Saves a CSV file.

The final CSV columns are:

```text
row_id
risk_level
evidence
factors
```

## 11. Summary

The current system uses:

- Excel loading with pandas
- label cleaning
- grouped train/validation split
- TF-IDF baseline model
- RoBERTa transformer fine-tuning
- class-weighted training
- rule-based evidence extraction
- validation with weighted F1 and approximate phrase F1

The strongest part is:

```text
risk-level classification
```

The weakest part is:

```text
evidence extraction
```

The best next improvement is to make the evidence extraction smarter.
