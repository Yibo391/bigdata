#!/usr/bin/env python3
"""V3 fast semantic Task 2 model.

Frozen ModernBERT embeddings are computed once and cached. Lightweight models
then combine current-post semantics, longitudinal user context, and V2 sparse
OOF probabilities. This module evaluates train.xlsx only and never writes a
leaderboard submission.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from v2_factor_ensemble import (
    FACTOR_LABELS, RISK_LABELS, load_frame, top_k_prediction,
    tune_prevalence_quotas, V2Config,
)


@dataclass
class V3Config:
    model_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 512
    encode_batch_size: int = 4
    folds: int = 5
    seed: int = 42
    semantic_c: float = 0.5
    output_dir: str = "outputs/v3_fast_semantic"
    v2_dir: str = "outputs/v2_factor_ensemble"
    cache_dir: str = "outputs/v3_fast_semantic/cache"
    prevalence_cap_multiplier: float = 2.5
    quota_shrink_to_gold: float = 0.35


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TextDataset(Dataset):
    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return index, self.texts[index]


class EncodeCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        indices, texts = zip(*batch)
        encoded = self.tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        return torch.tensor(indices), encoded


@torch.inference_mode()
def encode_posts(texts: list[str], cfg: V3Config, cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(texts):
            print("Loaded cached embeddings:", cache_path, cached.shape)
            return cached.astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModel.from_pretrained(cfg.model_name)
    device = choose_device()
    model.to(device).eval()
    loader = DataLoader(
        TextDataset(texts), batch_size=cfg.encode_batch_size, shuffle=False,
        collate_fn=EncodeCollator(tokenizer, cfg.max_length), num_workers=0,
    )
    rows, indices = [], []
    for batch_indices, encoded in tqdm(loader, desc=f"Encoding on {device}"):
        encoded = {k: v.to(device) for k, v in encoded.items()}
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        pooled = torch.nn.functional.normalize(pooled, dim=1)
        rows.append(pooled.float().cpu().numpy())
        indices.extend(batch_indices.numpy().tolist())
    matrix = np.concatenate(rows)
    matrix = matrix[np.argsort(indices)].astype(np.float16)
    np.save(cache_path, matrix)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    print("Saved embeddings:", cache_path, matrix.shape)
    return matrix.astype(np.float32)


def longitudinal_features(frame: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
    """Current, previous, next, user profile, and deviation embeddings."""
    embeddings = normalize(embeddings, norm="l2").astype(np.float32)
    n, width = embeddings.shape
    previous = np.zeros_like(embeddings)
    following = np.zeros_like(embeddings)
    profile = np.zeros_like(embeddings)
    position_features = np.zeros((n, 4), dtype=np.float32)
    for _, group in frame.assign(_row=np.arange(n)).groupby("anon_user_id", sort=False):
        group = group.sort_values("post_id")
        rows = group["_row"].to_numpy()
        values = embeddings[rows]
        total = values.sum(0)
        count = len(rows)
        for place, row in enumerate(rows):
            if place > 0:
                previous[row] = embeddings[rows[place - 1]]
            if place + 1 < count:
                following[row] = embeddings[rows[place + 1]]
            profile[row] = (total - embeddings[row]) / max(1, count - 1)
            position_features[row] = [
                place / max(1, count - 1),
                np.log1p(count) / 4.0,
                float(place > 0),
                float(place + 1 < count),
            ]
    for block in (previous, following, profile):
        block[:] = normalize(block, norm="l2")
    deviation = normalize(embeddings - profile, norm="l2")
    return np.hstack([
        embeddings, previous, following, profile, deviation, position_features,
    ]).astype(np.float32)


def semantic_oof(
    features: np.ndarray, y: np.ndarray, risk: np.ndarray,
    groups: pd.Series, cfg: V3Config,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    splitter = StratifiedGroupKFold(cfg.folds, shuffle=True, random_state=cfg.seed)
    splits = list(splitter.split(features, risk, groups))
    probability = np.zeros_like(y, dtype=np.float32)
    for fold, (fit_idx, valid_idx) in enumerate(splits, 1):
        for j in range(y.shape[1]):
            model = LogisticRegression(
                C=cfg.semantic_c, solver="liblinear", class_weight="balanced",
                max_iter=1500, random_state=cfg.seed + fold * 100 + j,
            )
            model.fit(features[fit_idx], y[fit_idx, j])
            probability[valid_idx, j] = model.predict_proba(features[valid_idx])[:, 1]
        fold_score = f1_score(
            y[valid_idx], probability[valid_idx] >= 0.5,
            average="macro", zero_division=0,
        )
        print(f"Fold {fold}: semantic raw Macro F1={fold_score:.4f}")
    return probability, splits


def user_smooth(
    frame: pd.DataFrame, probability: np.ndarray, alpha: float,
) -> np.ndarray:
    result = probability.copy()
    row = np.arange(len(frame))
    work = frame.assign(_row=row)
    for _, group in work.groupby("anon_user_id", sort=False):
        rows = group["_row"].to_numpy()
        if len(rows) > 1:
            mean = probability[rows].mean(0)
            result[rows] = (1 - alpha) * probability[rows] + alpha * mean
    return result


def select_per_label_ensemble(
    frame: pd.DataFrame, y: np.ndarray,
    semantic: np.ndarray, v2: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    final = np.zeros_like(semantic)
    choices = []
    for j, label in enumerate(FACTOR_LABELS):
        support = int(y[:, j].sum())
        best = (-1.0, 0.0, 0.0)
        for user_alpha in (0.0, 0.2, 0.4, 0.6):
            smoothed = user_smooth(frame, semantic[:, [j]], user_alpha)[:, 0]
            for v2_weight in np.arange(0, 1.01, 0.1):
                score = (1 - v2_weight) * smoothed + v2_weight * v2[:, j]
                prediction = top_k_prediction(score, support)
                value = f1_score(y[:, j], prediction, zero_division=0)
                if value > best[0]:
                    best = value, float(user_alpha), float(v2_weight)
        _, user_alpha, v2_weight = best
        smoothed = user_smooth(frame, semantic[:, [j]], user_alpha)[:, 0]
        final[:, j] = (1 - v2_weight) * smoothed + v2_weight * v2[:, j]
        choices.append({
            "factor": label, "user_alpha": user_alpha,
            "v2_weight": v2_weight, "selection_f1_at_gold_quota": best[0],
        })
    return final, choices


def run_v3_oof(
    train_path: str | Path = "train.xlsx", cfg: V3Config | None = None,
) -> dict:
    cfg = cfg or V3Config()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame, y, risk = load_frame(train_path)
    cache_name = f"train_modernbert_{cfg.max_length}_{len(frame)}.npy"
    embeddings = encode_posts(
        frame["post"].tolist(), cfg, Path(cfg.cache_dir) / cache_name
    )
    features = longitudinal_features(frame, embeddings)
    print("Longitudinal feature matrix:", features.shape)
    semantic_probability, _ = semantic_oof(
        features, y, risk, frame["anon_user_id"], cfg
    )
    v2_path = Path(cfg.v2_dir) / "oof_predictions.npz"
    if not v2_path.exists():
        raise FileNotFoundError("Run v2_factor_ensemble_training.ipynb first")
    v2_data = np.load(v2_path)
    v2_probability = v2_data["blended"]
    if v2_probability.shape != y.shape:
        raise ValueError("V2 OOF predictions do not align with training rows")
    combined_probability, choices = select_per_label_ensemble(
        frame, y, semantic_probability, v2_probability
    )
    quota_cfg = V2Config(
        prevalence_cap_multiplier=cfg.prevalence_cap_multiplier,
        quota_shrink_to_gold=cfg.quota_shrink_to_gold,
    )
    rates, prediction, fixed_prediction = tune_prevalence_quotas(
        y, combined_probability, quota_cfg
    )
    semantic_fixed = np.zeros_like(y)
    for j in range(y.shape[1]):
        semantic_fixed[:, j] = top_k_prediction(
            semantic_probability[:, j], int(y[:, j].sum())
        )
    semantic_macro = f1_score(
        y, semantic_fixed, average="macro", zero_division=0
    )
    combined_fixed_macro = f1_score(
        y, fixed_prediction, average="macro", zero_division=0
    )
    calibrated_macro = f1_score(
        y, prediction, average="macro", zero_division=0
    )
    precision, recall, label_f1, support = precision_recall_fscore_support(
        y, prediction, average=None, zero_division=0
    )
    choice_by_label = {x["factor"]: x for x in choices}
    per_label = pd.DataFrame({
        "factor": FACTOR_LABELS,
        "support": support,
        "gold_rate": y.mean(0),
        "submission_rate": rates,
        "user_alpha": [choice_by_label[x]["user_alpha"] for x in FACTOR_LABELS],
        "v2_weight": [choice_by_label[x]["v2_weight"] for x in FACTOR_LABELS],
        "precision": precision, "recall": recall, "f1": label_f1,
    })
    per_label.to_csv(output / "oof_per_label.csv", index=False)
    np.savez_compressed(
        output / "oof_predictions.npz", y=y,
        semantic_probability=semantic_probability,
        combined_probability=combined_probability,
        prediction=prediction,
    )
    metrics = {
        "rows": len(frame), "users": int(frame["anon_user_id"].nunique()),
        "embedding_shape": list(embeddings.shape),
        "feature_shape": list(features.shape),
        "semantic_fixed_quota_macro_f1": semantic_macro,
        "combined_fixed_quota_macro_f1": combined_fixed_macro,
        "combined_calibrated_macro_f1": calibrated_macro,
        "average_gold_labels": float(y.sum(1).mean()),
        "average_predicted_labels": float(prediction.sum(1).mean()),
        "ensemble_choices": choices,
    }
    (output / "oof_metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    return metrics
