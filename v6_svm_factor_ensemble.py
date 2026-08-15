#!/usr/bin/env python3
"""V6 Task 2: lightweight TF-IDF margin ensemble.

The new source is a set of balanced Linear SVM text classifiers. Their OOF
rankings are blended with the existing V3 rankings per factor. This module
uses train.xlsx only and never creates a leaderboard submission.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

from v2_factor_ensemble import (
    FACTOR_LABELS, V2Config, build_vectorizers, load_frame,
    top_k_prediction, tune_prevalence_quotas, vectorize,
)


@dataclass
class V6Config:
    folds: int = 5
    seed: int = 42
    svm_c_values: tuple[float, ...] = (0.03, 0.10, 0.30)
    max_iter: int = 7000
    blend_step: float = 0.10
    prevalence_cap_multiplier: float = 2.5
    quota_shrink_to_gold: float = 0.35
    output_dir: str = "outputs/v6_svm_factor_ensemble"
    # This already-evaluated V3 setting was the strongest lightweight semantic
    # run (about 0.4547 calibrated Macro F1).
    v3_dir: str = "outputs/v3_fast_semantic_c1.5"


def rank_columns(values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float32)
    for j in range(values.shape[1]):
        result[:, j] = (rankdata(values[:, j], method="average") - 0.5) / len(values)
    return result


def fixed_quota_prediction(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    prediction = np.zeros_like(y)
    for j in range(y.shape[1]):
        prediction[:, j] = top_k_prediction(score[:, j], int(y[:, j].sum()))
    return prediction


def train_svm_oof(frame, y, risk, cfg: V6Config):
    splitter = StratifiedGroupKFold(cfg.folds, shuffle=True, random_state=cfg.seed)
    splits = list(splitter.split(frame["post"], risk, frame["anon_user_id"]))
    result = {
        f"svm_c_{value:g}": np.zeros_like(y, dtype=np.float32)
        for value in cfg.svm_c_values
    }
    fold_scores = []
    for fold, (fit_idx, valid_idx) in enumerate(splits):
        assert set(frame.iloc[fit_idx].anon_user_id).isdisjoint(
            set(frame.iloc[valid_idx].anon_user_id)
        )
        word, char = build_vectorizers(V2Config())
        x_fit, x_valid = vectorize(
            word, char, frame.iloc[fit_idx]["post"], frame.iloc[valid_idx]["post"]
        )
        fold_record = {"fold": fold, "rows": len(valid_idx)}
        for value in cfg.svm_c_values:
            name = f"svm_c_{value:g}"
            for j in range(y.shape[1]):
                with catch_warnings():
                    simplefilter("ignore", ConvergenceWarning)
                    model = LinearSVC(
                        C=value, class_weight="balanced", dual="auto",
                        max_iter=cfg.max_iter, random_state=cfg.seed + fold * 100 + j,
                    )
                    model.fit(x_fit, y[fit_idx, j])
                result[name][valid_idx, j] = model.decision_function(x_valid)
            fold_prediction = np.zeros_like(y[valid_idx])
            for j in range(y.shape[1]):
                quota = max(1, round(y[fit_idx, j].mean() * len(valid_idx)))
                fold_prediction[:, j] = top_k_prediction(
                    result[name][valid_idx, j], quota
                )
            fold_record[name] = f1_score(
                y[valid_idx], fold_prediction, average="macro", zero_division=0
            )
        fold_scores.append(fold_record)
        print(json.dumps(fold_record, indent=2))
    return result, splits, fold_scores


def choose_v3_svm_blend(target, v3_score, svm_sources, label, step):
    support = int(target.sum())
    best = (-1.0, "v3", 0.0)
    weights = np.arange(0.0, 1.0 + step / 2, step)
    for name, svm_score in svm_sources.items():
        for svm_weight in weights:
            score = (1 - svm_weight) * v3_score[:, label] + svm_weight * svm_score[:, label]
            value = f1_score(
                target, top_k_prediction(score, support), zero_division=0
            )
            if value > best[0]:
                best = (value, name, float(svm_weight))
    return best


def nested_blend(y, splits, v3_rank, svm_ranks, cfg: V6Config):
    """Select blend settings on other users, then apply to each held-out fold."""
    combined = np.zeros_like(v3_rank)
    nested_choices = []
    for fold, (fit_idx, valid_idx) in enumerate(splits):
        fit_sources = {name: value[fit_idx] for name, value in svm_ranks.items()}
        for j, label in enumerate(FACTOR_LABELS):
            _, name, weight = choose_v3_svm_blend(
                y[fit_idx, j], v3_rank[fit_idx], fit_sources, j, cfg.blend_step
            )
            combined[valid_idx, j] = (
                (1 - weight) * v3_rank[valid_idx, j]
                + weight * svm_ranks[name][valid_idx, j]
            )
            nested_choices.append({
                "fold": fold, "factor": label,
                "svm_source": name, "svm_weight": weight,
            })
    return combined, nested_choices


def final_blend_choices(y, v3_rank, svm_ranks, cfg: V6Config):
    combined = np.zeros_like(v3_rank)
    choices = []
    for j, label in enumerate(FACTOR_LABELS):
        score, name, weight = choose_v3_svm_blend(
            y[:, j], v3_rank, svm_ranks, j, cfg.blend_step
        )
        combined[:, j] = (
            (1 - weight) * v3_rank[:, j] + weight * svm_ranks[name][:, j]
        )
        choices.append({
            "factor": label, "svm_source": name,
            "svm_weight": weight, "selection_f1": score,
        })
    return combined, choices


def run_v6_oof(train_path="train.xlsx", cfg: V6Config | None = None):
    cfg = cfg or V6Config()
    random.seed(cfg.seed); np.random.seed(cfg.seed)
    output = Path(cfg.output_dir); output.mkdir(parents=True, exist_ok=True)
    frame, y, risk = load_frame(train_path)
    v3_file = Path(cfg.v3_dir) / "oof_predictions.npz"
    if not v3_file.exists():
        raise FileNotFoundError("Run v3_fast_semantic_training.ipynb first")
    v3 = np.load(v3_file)["combined_probability"]
    if v3.shape != y.shape:
        raise ValueError("V3 OOF predictions do not align with train.xlsx")

    svm_raw, splits, fold_scores = train_svm_oof(frame, y, risk, cfg)
    v3_rank = rank_columns(v3)
    svm_ranks = {name: rank_columns(value) for name, value in svm_raw.items()}

    svm_metrics = {}
    for name, score in svm_ranks.items():
        svm_metrics[name] = f1_score(
            y, fixed_quota_prediction(y, score), average="macro", zero_division=0
        )
    v3_fixed = f1_score(
        y, fixed_quota_prediction(y, v3_rank), average="macro", zero_division=0
    )

    nested_score, nested_choices = nested_blend(y, splits, v3_rank, svm_ranks, cfg)
    nested_prediction = fixed_quota_prediction(y, nested_score)
    nested_macro = f1_score(y, nested_prediction, average="macro", zero_division=0)

    selected_score, final_choices = final_blend_choices(y, v3_rank, svm_ranks, cfg)
    selected_fixed = fixed_quota_prediction(y, selected_score)
    selected_fixed_macro = f1_score(
        y, selected_fixed, average="macro", zero_division=0
    )
    quota_cfg = V2Config(
        prevalence_cap_multiplier=cfg.prevalence_cap_multiplier,
        quota_shrink_to_gold=cfg.quota_shrink_to_gold,
    )
    rates, calibrated_prediction, _ = tune_prevalence_quotas(y, selected_score, quota_cfg)
    calibrated_macro = f1_score(
        y, calibrated_prediction, average="macro", zero_division=0
    )

    precision, recall, label_f1, support = precision_recall_fscore_support(
        y, calibrated_prediction, average=None, zero_division=0
    )
    choice_lookup = {x["factor"]: x for x in final_choices}
    per_label = pd.DataFrame({
        "factor": FACTOR_LABELS, "support": support,
        "gold_rate": y.mean(0), "submission_rate": rates,
        "svm_source": [choice_lookup[x]["svm_source"] for x in FACTOR_LABELS],
        "svm_weight": [choice_lookup[x]["svm_weight"] for x in FACTOR_LABELS],
        "precision": precision, "recall": recall, "f1": label_f1,
    })
    per_label.to_csv(output / "oof_per_label.csv", index=False)
    np.savez_compressed(
        output / "oof_predictions.npz", y=y, v3=v3,
        nested_combined=nested_score, selected_combined=selected_score,
        calibrated_prediction=calibrated_prediction,
        **svm_raw,
    )
    metrics = {
        "rows": len(frame), "users": int(frame.anon_user_id.nunique()),
        "v3_fixed_quota_macro_f1": v3_fixed,
        "svm_fixed_quota_macro_f1": svm_metrics,
        "nested_blend_fixed_quota_macro_f1": nested_macro,
        "selected_blend_fixed_quota_macro_f1": selected_fixed_macro,
        "selected_blend_calibrated_macro_f1": calibrated_macro,
        "average_gold_labels": float(y.sum(1).mean()),
        "average_calibrated_labels": float(calibrated_prediction.sum(1).mean()),
        "fold_scores": fold_scores,
    }
    (output / "oof_metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (output / "nested_choices.json").write_text(json.dumps(nested_choices, indent=2))
    (output / "final_choices.json").write_text(json.dumps(final_choices, indent=2))
    return metrics
