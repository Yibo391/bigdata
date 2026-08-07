#!/usr/bin/env python3
"""Build a V3 submission without changing the preserved Task 1 predictions.

Inputs:
  - outputs/SIT-MSF.csv: already-scored Task 1 risk/evidence predictions
  - train.xlsx and leaderboard.xlsx
  - V2/V3 OOF calibration artifacts

Output:
  - outputs/v3_submission/SIT-MSF.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier

from v2_factor_ensemble import (
    FACTOR_LABELS, V2Config, build_vectorizers, fit_binary_models,
    load_frame, top_k_prediction, vectorize,
)
from v3_fast_semantic import (
    V3Config, encode_posts, longitudinal_features, user_smooth,
)


def train_full_semantic_predict(
    train_features: np.ndarray, test_features: np.ndarray,
    y: np.ndarray, cfg: V3Config,
) -> np.ndarray:
    probability = np.zeros((len(test_features), y.shape[1]), dtype=np.float32)
    for j in range(y.shape[1]):
        model = LogisticRegression(
            C=cfg.semantic_c, solver="liblinear", class_weight="balanced",
            max_iter=1500, random_state=cfg.seed + j,
        )
        model.fit(train_features, y[:, j])
        probability[:, j] = model.predict_proba(test_features)[:, 1]
    return probability


def train_full_v2_predict(
    train: pd.DataFrame, board: pd.DataFrame, y: np.ndarray, risk: np.ndarray,
    v2_dir: Path,
) -> np.ndarray:
    cfg = V2Config(**json.loads((v2_dir / "config.json").read_text()))
    word, char = build_vectorizers(cfg)
    x_train, x_test = vectorize(word, char, train["post"], board["post"])
    base_probability = fit_binary_models(x_train, y, x_test, cfg)

    risk_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=60_000,
        sublinear_tf=True, strip_accents="unicode", dtype=np.float32,
    )
    risk_train = risk_vectorizer.fit_transform(train["post"])
    risk_test = risk_vectorizer.transform(board["post"])
    risk_model = SGDClassifier(
        loss="log_loss", alpha=2e-5, class_weight="balanced",
        max_iter=1500, random_state=cfg.seed,
    ).fit(risk_train, risk)
    risk_probability = risk_model.predict_proba(risk_test)
    meta_features = np.hstack([
        base_probability,
        risk_probability,
        np.log(np.clip(base_probability, 1e-5, 1 - 1e-5) /
               np.clip(1 - base_probability, 1e-5, 1)),
    ])
    meta_models = joblib.load(v2_dir / "meta_models.joblib")
    meta_probability = np.column_stack([
        model.predict_proba(meta_features)[:, 1] for model in meta_models
    ]).astype(np.float32)
    v2_metrics = json.loads((v2_dir / "oof_metrics.json").read_text())
    blended = np.zeros_like(base_probability)
    for j, label in enumerate(FACTOR_LABELS):
        weight = v2_metrics["calibration"][label]["blend_meta_weight"]
        blended[:, j] = weight * meta_probability[:, j] + (1 - weight) * base_probability[:, j]
    return blended


def validate_task1(task1: pd.DataFrame, board: pd.DataFrame) -> pd.DataFrame:
    required = ["row_id", "risk_level", "evidence", "factors"]
    if list(task1.columns) != required:
        raise ValueError(f"Task 1 CSV columns must be {required}")
    if len(task1) != len(board) or set(task1.row_id.astype(str)) != set(board.row_id.astype(str)):
        raise ValueError("Task 1 CSV row identifiers do not match leaderboard.xlsx")
    task1 = board[["row_id", "post"]].merge(
        task1[["row_id", "risk_level", "evidence"]], on="row_id", validate="one_to_one"
    )
    allowed = {"Indicator", "Ideation", "Behavior", "Attempt"}
    if set(task1.risk_level) - allowed:
        raise ValueError("Invalid Task 1 risk label")
    for _, row in task1.iterrows():
        evidence = "" if pd.isna(row.evidence) else str(row.evidence)
        for phrase in [x.strip() for x in evidence.split(";") if x.strip()]:
            if phrase not in row.post:
                raise ValueError(f"Non-verbatim preserved evidence for {row.row_id}: {phrase!r}")
    return task1


def build_v3_submission(
    train_path: str | Path = "train.xlsx",
    leaderboard_path: str | Path = "leaderboard.xlsx",
    preserved_task1_csv: str | Path = "outputs/SIT-MSF.csv",
    output_csv: str | Path = "outputs/v3_submission/SIT-MSF.csv",
) -> pd.DataFrame:
    train, y, risk = load_frame(train_path)
    board = pd.read_excel(leaderboard_path).copy()
    board["post"] = board["post"].fillna("").astype(str)
    task1 = validate_task1(pd.read_csv(preserved_task1_csv), board)
    v3_dir = Path("outputs/v3_fast_semantic")
    v2_dir = Path("outputs/v2_factor_ensemble")
    cfg = V3Config(**json.loads((v3_dir / "config.json").read_text()))

    train_embeddings = encode_posts(
        train["post"].tolist(), cfg,
        Path(cfg.cache_dir) / f"train_modernbert_{cfg.max_length}_{len(train)}.npy",
    )
    test_embeddings = encode_posts(
        board["post"].tolist(), cfg,
        Path(cfg.cache_dir) / f"leaderboard_modernbert_{cfg.max_length}_{len(board)}.npy",
    )
    train_features = longitudinal_features(train, train_embeddings)
    test_features = longitudinal_features(board, test_embeddings)
    semantic_probability = train_full_semantic_predict(
        train_features, test_features, y, cfg
    )
    v2_probability = train_full_v2_predict(train, board, y, risk, v2_dir)

    v3_metrics = json.loads((v3_dir / "oof_metrics.json").read_text())
    choices = {x["factor"]: x for x in v3_metrics["ensemble_choices"]}
    calibration = pd.read_csv(v3_dir / "oof_per_label.csv").set_index("factor")
    combined = np.zeros_like(semantic_probability)
    factor_binary = np.zeros_like(semantic_probability, dtype=np.int8)
    for j, label in enumerate(FACTOR_LABELS):
        choice = choices[label]
        semantic_smoothed = user_smooth(
            board, semantic_probability[:, [j]], choice["user_alpha"]
        )[:, 0]
        weight = choice["v2_weight"]
        combined[:, j] = (1 - weight) * semantic_smoothed + weight * v2_probability[:, j]
        rate = float(calibration.loc[label, "submission_rate"])
        quota = max(1, round(rate * len(board)))
        factor_binary[:, j] = top_k_prediction(combined[:, j], quota)

    factor_lists = [
        [FACTOR_LABELS[j] for j in np.flatnonzero(row)] for row in factor_binary
    ]
    submission = pd.DataFrame({
        "row_id": task1["row_id"].astype(str),
        "risk_level": task1["risk_level"],
        "evidence": task1["evidence"].fillna(""),
        "factors": [json.dumps(x, ensure_ascii=False) for x in factor_lists],
    })
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_csv, index=False)
    counts = {
        label: int(factor_binary[:, j].sum()) for j, label in enumerate(FACTOR_LABELS)
    }
    diagnostics = {
        "team_name": "SIT-MSF",
        "rows": len(submission),
        "preserved_task1_source": str(preserved_task1_csv),
        "task1_risk_counts": submission.risk_level.value_counts().to_dict(),
        "v3_factor_counts": counts,
        "average_factors": float(factor_binary.sum(1).mean()),
        "output_csv": str(output_csv),
    }
    output_csv.with_suffix(".diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps(diagnostics, indent=2))
    return submission


if __name__ == "__main__":
    build_v3_submission()
