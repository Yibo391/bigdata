# Task 2 Statistical Diagnosis

## Conclusion

The current Task 2 ceiling is caused mainly by weak ranking for rare and medium-support labels, not by thresholds or insufficient transformer training. Reaching 0.60 Macro F1 will require a new source of semantic supervision, such as a strong instruction-tuned local LLM used as a teacher/reranker or additional verified training data.

## Current performance by label support

Using the strongest saved V3 run (`semantic_c=1.5`):

| Positive training posts | Number of labels | Mean label F1 |
|---:|---:|---:|
| fewer than 20 | 3 | 0.197 |
| 20–49 | 3 | 0.194 |
| 50–99 | 5 | 0.415 |
| 100–199 | 4 | 0.491 |
| 200 or more | 9 | 0.634 |

The overall calibrated Macro F1 is approximately 0.455. The six labels with fewer than 50 examples average only 0.195 F1.

If the nine common labels remain at their current average of 0.634, the other fifteen labels must rise from approximately 0.347 to approximately 0.580 on average for the full Macro F1 to reach 0.60.

## Ranking versus calibration

For each label, an oracle was allowed to choose the best possible number of positive predictions while preserving the current V3 ranking. This intentionally optimistic calculation produced only approximately 0.467 Macro F1.

This result means that threshold tuning, prevalence quotas, and calibration cannot bridge the gap to 0.60. Correct examples are often ranked below incorrect examples, especially for rare labels.

Examples:

| Factor | Current F1 | Best F1 obtainable from current ranking |
|---|---:|---:|
| sexual orientation related issues | 0.133 | 0.143 |
| exposure to others' suicide | 0.214 | 0.214 |
| substance use | 0.213 | 0.250 |
| cognitive deficits | 0.147 | 0.174 |
| meaning in life | 0.222 | 0.231 |
| hopelessness | 0.745 | 0.755 |

## User timeline signal

Many factors persist across adjacent posts by the same user. For example, the adjacent-post lift over the base rate is approximately 19.7× for substance use, 19.3× for poor school performance, 14.4× for low socio-economic status, and 9.6× for cognitive deficits.

However, applying neighboring-post or user-average probability smoothing to the final V3 scores increased fixed-quota Macro F1 only from approximately 0.4540 to 0.4550. Temporal information is real, but current probabilities do not identify the rare factors reliably enough for smoothing to solve the problem.

## Interpretation of V1–V6

- V1 established the baseline and data pipeline.
- V2 fixed severe prevalence and threshold errors.
- V3 added semantic and longitudinal features and remains the most reliable submitted method.
- V4 and V5 showed that heavier encoder fine-tuning and oversampling do not solve missing rare-label information.
- V6 added a small selected-score improvement, but strict nested validation did not confirm it.

The experiments have now exhausted the most likely gains from calibration, linear models, basic semantic embeddings, encoder replacement, oversampling, and simple temporal smoothing.

## Recommended V7 direction

Use a strong local instruction-tuned LLM as a Task 2 teacher and reranker rather than training another encoder from scratch.

Recommended design:

1. Run a 4-bit 8B–9B instruction model locally on Apple silicon.
2. Give the model a precise definition, inclusion rule, exclusion rule, and examples for every factor.
3. Ask for a probability and a short supporting quote for each factor.
4. Reject predicted factors whose quoted evidence is not present in the post.
5. Evaluate teacher probabilities with the same five user-grouped folds.
6. Blend the teacher only for labels where it improves held-out users; keep V3 for the rest.
7. Calibrate output prevalence on training data after the blend.

This approach adds external semantic knowledge for factors represented by only 8–45 training examples. It is more credible than another round of oversampling. It requires inference rather than heavy gradient training, although processing all posts may still take several hours.

## Risk and expectation

An instruction-model teacher is the first remaining approach with a plausible path to a large improvement, but 0.60 cannot be guaranteed. The teacher must first be evaluated on labeled training posts. A submission should be generated only if user-grouped validation shows a clear and stable gain, especially on labels with fewer than 200 positives.
