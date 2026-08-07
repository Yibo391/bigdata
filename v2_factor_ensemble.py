#!/usr/bin/env python3
"""V2 Task 2: leakage-safe sparse ensemble with prevalence calibration.

This file is deliberately independent from transformer_multitask.py.  It trains
and evaluates factors only; it does not read leaderboard.xlsx or write a
submission.  A later inference file will combine its factors with preserved
Task 1 risk/evidence predictions.
"""
from __future__ import annotations

import ast
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

FACTOR_LABELS = [
    "mental health issues", "physical health/characteristic", "substance use",
    "hopelessness", "emotion dysregulation", "low self-esteem",
    "poor school performance", "low socio-economic status", "interpersonal violence",
    "prior self-harm or suicidal thought/attempt", "poor social support",
    "interpersonal difficulty", "dysfunctional family", "exposure to others' suicide",
    "stressful life event", "traumatic experience", "cognitive deficits",
    "suicide means (with access)", "sexual orientation related issues", "social support",
    "coping strategy", "psychological capital", "sense of responsibility", "meaning in life",
]
RISK_LABELS = ["Indicator", "Ideation", "Behavior", "Attempt"]


@dataclass
class V2Config:
    folds: int = 5
    seed: int = 42
    word_features: int = 80_000
    char_features: int = 120_000
    word_min_df: int = 2
    char_min_df: int = 2
    factor_c: float = 2.0
    meta_c: float = 0.5
    prevalence_cap_multiplier: float = 2.5
    quota_shrink_to_gold: float = 0.35
    output_dir: str = "outputs/v2_factor_ensemble"


def parse_factors(value: object) -> list[str]:
    if pd.isna(value):
        return []
    try:
        values = ast.literal_eval(value) if isinstance(value, str) else value
    except (SyntaxError, ValueError):
        return []
    return sorted({str(x).strip() for x in values if str(x).strip() in FACTOR_LABELS})


def load_frame(path: str | Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = pd.read_excel(path).copy()
    frame["post"] = frame["post"].fillna("").astype(str)
    frame["risk_level"] = frame["suicide risk"].astype(str).str.strip().str.title()
    if set(frame["risk_level"]) - set(RISK_LABELS):
        raise ValueError("Unknown risk labels")
    factor_sets = frame["factors"].map(parse_factors)
    y = np.zeros((len(frame), len(FACTOR_LABELS)), dtype=np.int8)
    factor_to_id = {x: i for i, x in enumerate(FACTOR_LABELS)}
    for i, labels in enumerate(factor_sets):
        for label in labels:
            y[i, factor_to_id[label]] = 1
    risk = frame["risk_level"].map({x: i for i, x in enumerate(RISK_LABELS)}).to_numpy()
    return frame, y, risk


def build_vectorizers(cfg: V2Config):
    word = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", sublinear_tf=True,
        ngram_range=(1, 2), min_df=cfg.word_min_df, max_df=0.995,
        max_features=cfg.word_features, dtype=np.float32,
    )
    char = TfidfVectorizer(
        lowercase=True, sublinear_tf=True, analyzer="char_wb",
        ngram_range=(3, 5), min_df=cfg.char_min_df, max_df=0.995,
        max_features=cfg.char_features, dtype=np.float32,
    )
    return word, char


def vectorize(word, char, fit_text, transform_text):
    fit_word = word.fit_transform(fit_text)
    transform_word = word.transform(transform_text)
    fit_char = char.fit_transform(fit_text)
    transform_char = char.transform(transform_text)
    return (
        hstack([fit_word, fit_char], format="csr"),
        hstack([transform_word, transform_char], format="csr"),
    )


def fit_binary_models(x_fit, y_fit, x_valid, cfg: V2Config) -> np.ndarray:
    probability = np.zeros((x_valid.shape[0], y_fit.shape[1]), dtype=np.float32)
    for j in range(y_fit.shape[1]):
        target = y_fit[:, j]
        if target.min() == target.max():
            probability[:, j] = target[0]
            continue
        model = LogisticRegression(
            C=cfg.factor_c, solver="liblinear", class_weight="balanced",
            max_iter=1500, random_state=cfg.seed,
        )
        model.fit(x_fit, target)
        probability[:, j] = model.predict_proba(x_valid)[:, 1]
    return probability


def crossfit_meta(
    base_probability: np.ndarray,
    risk_probability: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    cfg: V2Config,
) -> tuple[np.ndarray, list[LogisticRegression]]:
    features = np.hstack([
        base_probability,
        risk_probability,
        np.log(np.clip(base_probability, 1e-5, 1 - 1e-5) /
               np.clip(1 - base_probability, 1e-5, 1)),
    ])
    meta_oof = np.zeros_like(base_probability)
    for fit_idx, valid_idx in splits:
        for j in range(y.shape[1]):
            target = y[fit_idx, j]
            if target.min() == target.max():
                meta_oof[valid_idx, j] = target[0]
                continue
            model = LogisticRegression(
                C=cfg.meta_c, solver="liblinear", class_weight="balanced",
                max_iter=1000, random_state=cfg.seed + j,
            )
            model.fit(features[fit_idx], target)
            meta_oof[valid_idx, j] = model.predict_proba(features[valid_idx])[:, 1]
    final_models = []
    for j in range(y.shape[1]):
        model = LogisticRegression(
            C=cfg.meta_c, solver="liblinear", class_weight="balanced",
            max_iter=1000, random_state=cfg.seed + j,
        ).fit(features, y[:, j])
        final_models.append(model)
    return meta_oof, final_models


def choose_blend(y: np.ndarray, base: np.ndarray, meta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blended = np.zeros_like(base)
    weights = np.zeros(y.shape[1])
    for j in range(y.shape[1]):
        best = (-1.0, 0.0)
        prevalence = y[:, j].mean()
        for weight in np.arange(0, 1.01, 0.1):
            score = weight * meta[:, j] + (1 - weight) * base[:, j]
            prediction = top_k_prediction(score, max(1, round(prevalence * len(y))))
            f1 = f1_score(y[:, j], prediction, zero_division=0)
            if f1 > best[0]:
                best = f1, float(weight)
        weights[j] = best[1]
        blended[:, j] = weights[j] * meta[:, j] + (1 - weights[j]) * base[:, j]
    return blended, weights


def top_k_prediction(scores: np.ndarray, k: int) -> np.ndarray:
    result = np.zeros(len(scores), dtype=np.int8)
    if k > 0:
        result[np.argsort(-scores, kind="stable")[:min(k, len(scores))]] = 1
    return result


def tune_prevalence_quotas(
    y: np.ndarray, probability: np.ndarray, cfg: V2Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rates = np.zeros(y.shape[1])
    prediction = np.zeros_like(y)
    fixed_prevalence_prediction = np.zeros_like(y)
    for j in range(y.shape[1]):
        support = int(y[:, j].sum())
        fixed_prevalence_prediction[:, j] = top_k_prediction(probability[:, j], support)
        maximum = max(1, int(math_ceil(support * cfg.prevalence_cap_multiplier)))
        best = (-1.0, support)
        for k in range(1, min(len(y), maximum) + 1):
            candidate = top_k_prediction(probability[:, j], k)
            score = f1_score(y[:, j], candidate, zero_division=0)
            if score > best[0]:
                best = score, k
        shrunk_k = round(
            (1 - cfg.quota_shrink_to_gold) * best[1]
            + cfg.quota_shrink_to_gold * support
        )
        shrunk_k = max(1, min(maximum, shrunk_k))
        rates[j] = shrunk_k / len(y)
        prediction[:, j] = top_k_prediction(probability[:, j], shrunk_k)
    return rates, prediction, fixed_prevalence_prediction


def math_ceil(value: float) -> int:
    # Isolated for JSON/notebook portability and exact positive-integer behavior.
    return int(np.ceil(value))


def risk_oof_predictions(
    texts: pd.Series, risk: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]], cfg: V2Config,
) -> np.ndarray:
    result = np.zeros((len(texts), len(RISK_LABELS)), dtype=np.float32)
    for fold, (fit_idx, valid_idx) in enumerate(splits):
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=60_000,
            sublinear_tf=True, strip_accents="unicode", dtype=np.float32,
        )
        x_fit = vectorizer.fit_transform(texts.iloc[fit_idx])
        x_valid = vectorizer.transform(texts.iloc[valid_idx])
        model = SGDClassifier(
            loss="log_loss", alpha=2e-5, class_weight="balanced",
            max_iter=1500, random_state=cfg.seed + fold,
        ).fit(x_fit, risk[fit_idx])
        result[valid_idx] = model.predict_proba(x_valid)
    return result


def run_v2_oof(train_path: str | Path = "train.xlsx", cfg: V2Config | None = None) -> dict:
    cfg = cfg or V2Config()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame, y, risk = load_frame(train_path)
    splitter = StratifiedGroupKFold(cfg.folds, shuffle=True, random_state=cfg.seed)
    splits = list(splitter.split(frame["post"], risk, frame["anon_user_id"]))
    factor_oof = np.zeros_like(y, dtype=np.float32)
    fold_scores = []
    for fold, (fit_idx, valid_idx) in enumerate(splits, 1):
        assert set(frame.iloc[fit_idx]["anon_user_id"]).isdisjoint(
            set(frame.iloc[valid_idx]["anon_user_id"])
        )
        word, char = build_vectorizers(cfg)
        x_fit, x_valid = vectorize(
            word, char, frame.iloc[fit_idx]["post"], frame.iloc[valid_idx]["post"]
        )
        factor_oof[valid_idx] = fit_binary_models(x_fit, y[fit_idx], x_valid, cfg)
        fold_binary = factor_oof[valid_idx] >= 0.5
        score = f1_score(y[valid_idx], fold_binary, average="macro", zero_division=0)
        fold_scores.append(score)
        print(f"Fold {fold}: raw 0.5 Macro F1={score:.4f}")
    risk_probability = risk_oof_predictions(frame["post"], risk, splits, cfg)
    meta_oof, meta_models = crossfit_meta(factor_oof, risk_probability, y, splits, cfg)
    blended, blend_weights = choose_blend(y, factor_oof, meta_oof)
    rates, calibrated_prediction, fixed_prediction = tune_prevalence_quotas(y, blended, cfg)
    raw_macro = f1_score(y, factor_oof >= 0.5, average="macro", zero_division=0)
    fixed_macro = f1_score(y, fixed_prediction, average="macro", zero_division=0)
    calibrated_macro = f1_score(y, calibrated_prediction, average="macro", zero_division=0)
    precision, recall, label_f1, support = precision_recall_fscore_support(
        y, calibrated_prediction, average=None, zero_division=0
    )
    per_label = pd.DataFrame({
        "factor": FACTOR_LABELS,
        "support": support,
        "gold_rate": y.mean(0),
        "submission_rate": rates,
        "blend_meta_weight": blend_weights,
        "precision": precision,
        "recall": recall,
        "f1": label_f1,
    })
    per_label.to_csv(output / "oof_per_label.csv", index=False)
    np.savez_compressed(
        output / "oof_predictions.npz", y=y, factor_oof=factor_oof,
        meta_oof=meta_oof, blended=blended, risk_probability=risk_probability,
        calibrated_prediction=calibrated_prediction,
    )
    calibration = {
        label: {
            "gold_rate": float(y[:, j].mean()),
            "submission_rate": float(rates[j]),
            "blend_meta_weight": float(blend_weights[j]),
        }
        for j, label in enumerate(FACTOR_LABELS)
    }
    metrics = {
        "rows": len(frame), "users": int(frame["anon_user_id"].nunique()),
        "fold_raw_macro_f1": fold_scores,
        "raw_0.5_macro_f1": raw_macro,
        "fixed_prevalence_macro_f1": fixed_macro,
        "calibrated_macro_f1": calibrated_macro,
        "average_gold_labels": float(y.sum(1).mean()),
        "average_calibrated_labels": float(calibrated_prediction.sum(1).mean()),
        "calibration": calibration,
    }
    (output / "oof_metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    joblib.dump(meta_models, output / "meta_models.joblib", compress=3)
    return metrics
