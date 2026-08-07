#!/usr/bin/env python3
"""V4: five-fold fine-tuned ModernBERT for Task 2.

New independent implementation. It does not modify V1/V2/V3 artifacts and it
does not read leaderboard.xlsx during evaluation.
"""
from __future__ import annotations

import ast
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from v2_factor_ensemble import (
    FACTOR_LABELS, RISK_LABELS, V2Config, load_frame,
    top_k_prediction, tune_prevalence_quotas,
)

FACTOR_DESCRIPTIONS = [
    "mental illness, depression, anxiety, psychiatric symptoms, diagnosis or treatment",
    "physical illness, pain, disability, body characteristic or medical condition",
    "alcohol, drugs, smoking, intoxication, addiction or substance misuse",
    "hopelessness, no future, nothing will improve, giving up or despair",
    "uncontrolled anger, panic, intense mood changes, crying or emotional instability",
    "worthlessness, self-hatred, shame, feeling useless, ugly or like a burden",
    "bad grades, failing classes, academic pressure or difficulty studying",
    "poverty, unemployment, debt, homelessness or financial difficulty",
    "abuse, bullying, assault, domestic violence or physical and emotional violence",
    "previous self-harm, suicidal thoughts, suicide plan or suicide attempt",
    "loneliness, isolation, nobody cares, lack of friends or lack of support",
    "conflict, breakup, rejection, friendship problems or relationship difficulty",
    "family conflict, neglect, abusive parents, divorce or dysfunctional household",
    "friend, relative or another person died by suicide or attempted suicide",
    "job loss, exam, breakup, bereavement, pandemic or major stressful life change",
    "trauma, abuse, disturbing past event, post-traumatic stress or painful memory",
    "confusion, poor concentration, distorted thinking, indecision or cognitive difficulty",
    "access to pills, gun, rope, knife, bridge or another available suicide method",
    "LGBTQ identity, coming out, homophobia, gender identity or sexuality-related distress",
    "receiving help, care, encouragement or practical support from other people",
    "therapy, distraction, exercise, music, writing, seeking help or another coping action",
    "hope, optimism, resilience, confidence that life can improve or positive future belief",
    "responsibility to family, partner, children, pets or others as a reason to stay alive",
    "purpose, goals, faith, values, reasons for living or a meaningful life",
]


@dataclass
class V4Config:
    model_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 1024
    folds: int = 5
    epochs: int = 5
    train_batch_size: int = 1
    eval_batch_size: int = 2
    gradient_accumulation: int = 8
    encoder_lr: float = 1.2e-5
    head_lr: float = 1.2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout: float = 0.15
    risk_loss_weight: float = 0.15
    ranking_loss_weight: float = 0.15
    count_loss_weight: float = 0.03
    prevalence_cap_multiplier: float = 2.5
    quota_shrink_to_gold: float = 0.35
    patience: int = 2
    resume_completed_folds: bool = True
    seed: int = 42
    output_dir: str = "outputs/v4_finetuned_factor"
    v2_dir: str = "outputs/v2_factor_ensemble"


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def factor_multiplicity(value: object) -> dict[str, int]:
    if pd.isna(value):
        return {}
    try:
        values = ast.literal_eval(value) if isinstance(value, str) else value
    except (SyntaxError, ValueError):
        return {}
    counts = {}
    for item in values:
        label = str(item).strip()
        if label in FACTOR_LABELS:
            counts[label] = counts.get(label, 0) + 1
    return counts


class FactorDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, y: np.ndarray, risk: np.ndarray, tokenizer, cfg: V4Config):
        self.frame = frame.reset_index(drop=True)
        self.y = y.astype(np.float32)
        self.risk = risk.astype(np.int64)
        self.tokenizer = tokenizer
        self.cfg = cfg
        factor_to_id = {x: i for i, x in enumerate(FACTOR_LABELS)}
        self.positive_weight = np.ones_like(self.y, dtype=np.float32)
        for i, raw in enumerate(self.frame["factors"]):
            for label, count in factor_multiplicity(raw).items():
                self.positive_weight[i, factor_to_id[label]] = 1.0 + 0.20 * np.log1p(count - 1)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.frame.iloc[index]["post"], truncation=True,
            max_length=self.cfg.max_length,
        )
        return {
            "row_index": index,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "factor_labels": self.y[index],
            "positive_weight": self.positive_weight[index],
            "risk_label": int(self.risk[index]),
            "factor_count": float(self.y[index].sum()),
        }


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows):
        batch = self.tokenizer.pad([
            {"input_ids": x["input_ids"], "attention_mask": x["attention_mask"]}
            for x in rows
        ], return_tensors="pt")
        batch.update({
            "row_index": torch.tensor([x["row_index"] for x in rows]),
            "factor_labels": torch.tensor(np.stack([x["factor_labels"] for x in rows])),
            "positive_weight": torch.tensor(np.stack([x["positive_weight"] for x in rows])),
            "risk_label": torch.tensor([x["risk_label"] for x in rows]),
            "factor_count": torch.tensor([x["factor_count"] for x in rows]),
        })
        return batch


class DescriptionAttentionModel(nn.Module):
    def __init__(self, cfg: V4Config):
        super().__init__()
        # The model is already downloaded on this machine. Loading strictly from
        # cache prevents an unnecessary Hugging Face Hub request from stopping a
        # long cross-validation run when the network/proxy is temporarily down.
        self.encoder = AutoModel.from_pretrained(cfg.model_name, local_files_only=True)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(cfg.dropout)
        self.label_queries = nn.Parameter(torch.empty(len(FACTOR_LABELS), hidden))
        self.label_weights = nn.Parameter(torch.empty(len(FACTOR_LABELS), hidden))
        self.label_bias = nn.Parameter(torch.zeros(len(FACTOR_LABELS)))
        self.global_gate = nn.Parameter(torch.zeros(len(FACTOR_LABELS)))
        self.risk_head = nn.Linear(hidden, len(RISK_LABELS))
        self.count_head = nn.Linear(hidden, 1)
        nn.init.normal_(self.label_queries, std=0.02)
        nn.init.normal_(self.label_weights, std=0.02)

    @torch.no_grad()
    def initialize_queries(self, tokenizer, device):
        encoded = tokenizer(
            FACTOR_DESCRIPTIONS, padding=True, truncation=True,
            max_length=64, return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        was_training = self.encoder.training
        self.encoder.eval()
        hidden = self.encoder(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        pooled = F.normalize(pooled, dim=1)
        self.label_queries.copy_(pooled)
        if was_training:
            self.encoder.train()

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        global_context = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        hidden = self.dropout(hidden)
        scores = torch.einsum("bth,lh->blt", hidden, F.normalize(self.label_queries, dim=1))
        scores = scores / math.sqrt(hidden.shape[-1])
        scores = scores.masked_fill(~attention_mask[:, None, :].bool(), -1e4)
        attention = torch.softmax(scores, dim=-1)
        label_context = torch.einsum("blt,bth->blh", attention, hidden)
        gate = torch.sigmoid(self.global_gate)[None, :, None]
        fused = label_context + gate * global_context[:, None, :]
        logits = (self.dropout(fused) * self.label_weights[None]).sum(-1) + self.label_bias
        return logits, self.risk_head(self.dropout(global_context)), self.count_head(global_context).squeeze(-1)


def ranking_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    losses = []
    for row_logits, row_targets in zip(logits, targets):
        positive = row_logits[row_targets > 0.5]
        negative = row_logits[row_targets < 0.5]
        if len(positive) and len(negative):
            losses.append(F.softplus(negative[:, None] - positive[None, :]).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0


def compute_loss(outputs, batch, pos_weight, cfg: V4Config):
    factor_logits, risk_logits, count_prediction = outputs
    base = F.binary_cross_entropy_with_logits(
        factor_logits, batch["factor_labels"], pos_weight=pos_weight, reduction="none"
    )
    sample_weight = torch.where(
        batch["factor_labels"] > 0.5, batch["positive_weight"], torch.ones_like(base)
    )
    factor_loss = (base * sample_weight).mean()
    risk_loss = F.cross_entropy(risk_logits, batch["risk_label"], label_smoothing=0.03)
    rank_loss = ranking_loss(factor_logits, batch["factor_labels"])
    count_loss = F.smooth_l1_loss(
        count_prediction, torch.log1p(batch["factor_count"])
    )
    total = (
        factor_loss + cfg.risk_loss_weight * risk_loss
        + cfg.ranking_loss_weight * rank_loss
        + cfg.count_loss_weight * count_loss
    )
    return total


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    probability, indices = [], []
    for batch in loader:
        row_index = batch.pop("row_index")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits, _, _ = model(input_ids, attention_mask)
        probability.append(torch.sigmoid(logits).cpu().numpy())
        indices.extend(row_index.numpy().tolist())
    probability = np.concatenate(probability)
    order = np.argsort(indices)
    return probability[order]


def prevalence_prediction(probability, fit_y, valid_size):
    prediction = np.zeros((valid_size, fit_y.shape[1]), dtype=np.int8)
    for j in range(fit_y.shape[1]):
        rate = fit_y[:, j].mean()
        k = max(1, round(rate * valid_size))
        prediction[:, j] = top_k_prediction(probability[:, j], k)
    return prediction


def safe_macro_average_precision(target: np.ndarray, probability: np.ndarray) -> float:
    """Average only labels for which AP is defined in this validation fold."""
    scores = []
    for j in range(target.shape[1]):
        if 0 < target[:, j].sum() < len(target):
            scores.append(average_precision_score(target[:, j], probability[:, j]))
    return float(np.mean(scores)) if scores else 0.0


def train_fold(frame, y, risk, fit_idx, valid_idx, fold, cfg: V4Config):
    seed_all(cfg.seed + fold)
    fold_dir = Path(cfg.output_dir) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, use_fast=True, local_files_only=True
    )
    fit_frame, valid_frame = frame.iloc[fit_idx].reset_index(drop=True), frame.iloc[valid_idx].reset_index(drop=True)
    fit_data = FactorDataset(fit_frame, y[fit_idx], risk[fit_idx], tokenizer, cfg)
    valid_data = FactorDataset(valid_frame, y[valid_idx], risk[valid_idx], tokenizer, cfg)
    collator = Collator(tokenizer)
    train_loader = DataLoader(
        fit_data, batch_size=cfg.train_batch_size, shuffle=True,
        collate_fn=collator, num_workers=0,
    )
    valid_loader = DataLoader(
        valid_data, batch_size=cfg.eval_batch_size, shuffle=False,
        collate_fn=collator, num_workers=0,
    )
    device = choose_device()
    model = DescriptionAttentionModel(cfg).to(device)
    model.initialize_queries(tokenizer, device)
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
    head_names = ("label_", "global_gate", "risk_head", "count_head")
    encoder_params, head_params = [], []
    for name, parameter in model.named_parameters():
        (head_params if name.startswith(head_names) else encoder_params).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": cfg.encoder_lr},
        {"params": head_params, "lr": cfg.head_lr},
    ], weight_decay=cfg.weight_decay)
    updates = math.ceil(len(train_loader) / cfg.gradient_accumulation) * cfg.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(updates * cfg.warmup_ratio), updates
    )
    positive = y[fit_idx].sum(0)
    pos_weight = torch.tensor(
        np.sqrt((len(fit_idx) - positive) / np.maximum(positive, 1)).clip(1, 12),
        dtype=torch.float32, device=device,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_score, stale, history = -1.0, 0, []
    best_probability = None
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        progress = tqdm(train_loader, desc=f"Fold {fold} epoch {epoch}", leave=False)
        for step, batch in enumerate(progress, 1):
            batch.pop("row_index")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(batch["input_ids"], batch["attention_mask"])
                loss = compute_loss(outputs, batch, pos_weight, cfg)
                scaled_loss = loss / cfg.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            running += loss.item()
            if step % cfg.gradient_accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            if step % 100 == 0:
                progress.set_postfix(loss=f"{running / step:.3f}")
        probability = predict(model, valid_loader, device)
        validation_prediction = prevalence_prediction(probability, y[fit_idx], len(valid_idx))
        macro = f1_score(y[valid_idx], validation_prediction, average="macro", zero_division=0)
        macro_ap = safe_macro_average_precision(y[valid_idx], probability)
        record = {
            "epoch": epoch, "loss": running / len(train_loader),
            "prevalence_macro_f1": macro, "macro_average_precision": macro_ap,
        }
        history.append(record)
        print(json.dumps({"fold": fold, **record}, indent=2))
        # F1 is primary; AP breaks near-ties without selecting probability thresholds.
        selection = macro + 0.05 * macro_ap
        if selection > best_score:
            best_score, stale = selection, 0
            best_probability = probability.copy()
            torch.save(model.state_dict(), fold_dir / "best_model.pt")
            tokenizer.save_pretrained(fold_dir / "tokenizer")
            (fold_dir / "best_metrics.json").write_text(json.dumps(record, indent=2))
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    (fold_dir / "history.json").write_text(json.dumps(history, indent=2))
    np.save(fold_dir / "valid_probability.npy", best_probability)
    (fold_dir / "valid_row_ids.json").write_text(json.dumps(
        frame.iloc[valid_idx]["row_id"].astype(str).tolist()
    ))
    del model
    if device.type == "mps": torch.mps.empty_cache()
    if device.type == "cuda": torch.cuda.empty_cache()
    return best_probability, history


def blend_with_v2(y, v4_probability, v2_probability):
    combined = np.zeros_like(v4_probability)
    weights = np.zeros(y.shape[1])
    for j in range(y.shape[1]):
        support = int(y[:, j].sum())
        best = (-1.0, 0.0)
        for v2_weight in np.arange(0, 1.01, 0.1):
            score = (1 - v2_weight) * v4_probability[:, j] + v2_weight * v2_probability[:, j]
            prediction = top_k_prediction(score, support)
            value = f1_score(y[:, j], prediction, zero_division=0)
            if value > best[0]: best = value, float(v2_weight)
        weights[j] = best[1]
        combined[:, j] = (1 - weights[j]) * v4_probability[:, j] + weights[j] * v2_probability[:, j]
    return combined, weights


def run_v4_oof(train_path="train.xlsx", cfg: V4Config | None = None):
    cfg = cfg or V4Config()
    seed_all(cfg.seed)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame, y, risk = load_frame(train_path)
    splitter = StratifiedGroupKFold(cfg.folds, shuffle=True, random_state=cfg.seed)
    splits = list(splitter.split(frame["post"], risk, frame["anon_user_id"]))
    oof = np.zeros_like(y, dtype=np.float32)
    fold_history = []
    for fold, (fit_idx, valid_idx) in enumerate(splits):
        assert set(frame.iloc[fit_idx].anon_user_id).isdisjoint(set(frame.iloc[valid_idx].anon_user_id))
        fold_dir = output / f"fold_{fold}"
        probability_file = fold_dir / "valid_probability.npy"
        row_ids_file = fold_dir / "valid_row_ids.json"
        history_file = fold_dir / "history.json"
        expected_ids = frame.iloc[valid_idx]["row_id"].astype(str).tolist()
        can_resume = (
            cfg.resume_completed_folds and probability_file.exists()
            and row_ids_file.exists() and history_file.exists()
            and json.loads(row_ids_file.read_text()) == expected_ids
        )
        if can_resume:
            probability = np.load(probability_file)
            history = json.loads(history_file.read_text())
            if probability.shape != (len(valid_idx), len(FACTOR_LABELS)):
                raise ValueError(f"Invalid cached probability shape in {probability_file}")
            print(f"Fold {fold}: using completed cached predictions")
        else:
            probability, history = train_fold(frame, y, risk, fit_idx, valid_idx, fold, cfg)
        oof[valid_idx] = probability
        fold_history.append(history)
    v2_file = Path(cfg.v2_dir) / "oof_predictions.npz"
    v2 = np.load(v2_file)["blended"]
    combined, v2_weights = blend_with_v2(y, oof, v2)
    quota_cfg = V2Config(
        prevalence_cap_multiplier=cfg.prevalence_cap_multiplier,
        quota_shrink_to_gold=cfg.quota_shrink_to_gold,
    )
    rates, prediction, fixed_prediction = tune_prevalence_quotas(y, combined, quota_cfg)
    v4_fixed = np.zeros_like(y)
    for j in range(y.shape[1]):
        v4_fixed[:, j] = top_k_prediction(oof[:, j], int(y[:, j].sum()))
    v4_macro = f1_score(y, v4_fixed, average="macro", zero_division=0)
    combined_macro = f1_score(y, prediction, average="macro", zero_division=0)
    precision, recall, label_f1, support = precision_recall_fscore_support(
        y, prediction, average=None, zero_division=0
    )
    per_label = pd.DataFrame({
        "factor": FACTOR_LABELS, "support": support,
        "gold_rate": y.mean(0), "submission_rate": rates,
        "v2_weight": v2_weights, "precision": precision,
        "recall": recall, "f1": label_f1,
    })
    per_label.to_csv(output / "oof_per_label.csv", index=False)
    np.savez_compressed(
        output / "oof_predictions.npz", y=y, v4=oof,
        combined=combined, prediction=prediction,
    )
    metrics = {
        "v4_only_fixed_quota_macro_f1": v4_macro,
        "v4_v2_calibrated_macro_f1": combined_macro,
        "average_gold_labels": float(y.sum(1).mean()),
        "average_predicted_labels": float(prediction.sum(1).mean()),
        "fold_history": fold_history,
    }
    (output / "oof_metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    return metrics
