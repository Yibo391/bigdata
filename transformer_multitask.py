#!/usr/bin/env python3
"""Long-context multi-task transformer for suicide-risk shared-task evaluation.

Trains three heads on one encoder:
  1. four-way risk classification,
  2. token-level evidence extraction,
  3. 24-label suicide-factor classification.

This module intentionally contains no leaderboard prediction routine.  It only
uses labeled training rows and user-grouped validation.
"""
from __future__ import annotations

import ast
import difflib
import html
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

RISK_LABELS = ["Indicator", "Ideation", "Behavior", "Attempt"]
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
EVIDENCE_PREFIX = re.compile(
    r"^\s*main\s+phrases?\s+that\s+lead\s+to\s+this\s+assessment\s*:\s*", re.I
)
NONE_VALUES = {"", "none", "nan", "na", "n/a", "null"}
WORD_RE = re.compile(r"\b[\w']+\b")


@dataclass
class Config:
    model_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 1024
    epochs: int = 5
    train_batch_size: int = 1
    eval_batch_size: int = 2
    grad_accumulation: int = 8
    learning_rate: float = 2e-5
    head_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    patience: int = 2
    seed: int = 42
    validation_fold: int = 0
    n_splits: int = 5
    risk_loss_weight: float = 0.40
    evidence_loss_weight: float = 0.30
    factor_loss_weight: float = 0.30
    factor_gamma_neg: float = 4.0
    factor_gamma_pos: float = 1.0
    evidence_positive_weight: float = 8.0
    evidence_max_spans: int = 3
    output_dir: str = "outputs/transformer_multitask"
    num_workers: int = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_risk(value: object) -> str:
    label = str(value).strip().title()
    if label not in RISK_LABELS:
        raise ValueError(f"Unknown risk label: {value!r}")
    return label


def parse_factors(value: object) -> list[str]:
    if pd.isna(value):
        return []
    try:
        values = ast.literal_eval(value) if isinstance(value, str) else value
    except (SyntaxError, ValueError):
        return []
    # Duplicates in the workbook are repeated annotations, not separate labels.
    return sorted({str(v).strip() for v in values if str(v).strip() in FACTOR_LABELS})


def parse_evidence(value: object) -> list[str]:
    """Remove annotation boilerplate and return the semicolon-delimited phrases."""
    if pd.isna(value):
        return []
    text = EVIDENCE_PREFIX.sub("", str(value).strip())
    result, seen = [], set()
    for phrase in re.split(r";|\n", text):
        phrase = re.sub(r"\s+", " ", phrase).strip()
        key = phrase.casefold()
        if key not in NONE_VALUES and key not in seen:
            result.append(phrase)
            seen.add(key)
    return result


def find_phrase_offsets(post: str, phrases: Iterable[str]) -> tuple[list[tuple[int, int]], list[str]]:
    """Locate gold phrases with exact/flexible-whitespace matching."""
    offsets, missed = [], []
    for phrase in phrases:
        match = re.search(re.escape(phrase), post, flags=re.I)
        if match is None:
            flexible = r"\s+".join(re.escape(x) for x in phrase.split())
            match = re.search(flexible, post, flags=re.I)
        if match is None:
            # Annotation punctuation occasionally differs from the source.
            trimmed = phrase.strip(" \t\r\n.,;:!?\"'")
            flexible = r"\s+".join(re.escape(x) for x in trimmed.split())
            match = re.search(flexible, post, flags=re.I) if flexible else None
        if match is None:
            # Handle curly apostrophes, HTML entities, "my self"/"myself", and
            # lightly paraphrased annotations.  The returned span is still an
            # exact character slice of the original post.
            post_tokens = [
                (re.sub(r"\W+", "", m.group().casefold()), m.start(), m.end())
                for m in WORD_RE.finditer(post)
            ]
            phrase_tokens = [
                re.sub(r"\W+", "", x.casefold())
                for x in WORD_RE.findall(html.unescape(phrase).replace("’", "'").replace("…", " "))
            ]
            post_tokens = [x for x in post_tokens if x[0]]
            phrase_tokens = [x for x in phrase_tokens if x]
            best = (0.0, None)
            if phrase_tokens and post_tokens:
                expected = len(phrase_tokens)
                for width in range(max(1, expected - 3), min(len(post_tokens), expected + 5) + 1):
                    for start in range(0, len(post_tokens) - width + 1):
                        candidate = [x[0] for x in post_tokens[start:start + width]]
                        ratio = difflib.SequenceMatcher(None, phrase_tokens, candidate).ratio()
                        if ratio > best[0]:
                            best = ratio, (post_tokens[start][1], post_tokens[start + width - 1][2])
                if best[0] >= 0.78:
                    fuzzy_start, fuzzy_end = best[1]
                    offsets.append((fuzzy_start, fuzzy_end))
                    continue
        if match is None:
            missed.append(phrase)
        else:
            offsets.append((match.start(), match.end()))
    return offsets, missed


def load_training_frame(path: str | Path) -> pd.DataFrame:
    frame = pd.read_excel(path).copy()
    required = {
        "row_id", "anon_user_id", "post", "suicide risk",
        "evidence for suicide risk level", "factors",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame["post"] = frame["post"].fillna("").astype(str)
    frame["risk_level"] = frame["suicide risk"].map(normalize_risk)
    frame["risk_id"] = frame["risk_level"].map({x: i for i, x in enumerate(RISK_LABELS)})
    frame["factor_labels"] = frame["factors"].map(parse_factors)
    frame["evidence_spans"] = frame["evidence for suicide risk level"].map(parse_evidence)
    located = frame.apply(
        lambda r: find_phrase_offsets(r["post"], r["evidence_spans"]), axis=1
    )
    frame["evidence_offsets"] = [x[0] for x in located]
    frame["unlocated_evidence"] = [x[1] for x in located]
    return frame


def grouped_split(frame: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(
        n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed
    )
    splits = list(splitter.split(frame, frame["risk_id"], frame["anon_user_id"]))
    fit_idx, val_idx = splits[cfg.validation_fold]
    assert set(frame.iloc[fit_idx]["anon_user_id"]).isdisjoint(
        set(frame.iloc[val_idx]["anon_user_id"])
    )
    return fit_idx, val_idx


class SuicideDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, cfg: Config):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.cfg = cfg
        factor_to_id = {x: i for i, x in enumerate(FACTOR_LABELS)}
        self.factor_targets = []
        for labels in self.frame["factor_labels"]:
            target = np.zeros(len(FACTOR_LABELS), dtype=np.float32)
            for label in labels:
                target[factor_to_id[label]] = 1
            self.factor_targets.append(target)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        encoded = self.tokenizer(
            row["post"],
            truncation=True,
            max_length=self.cfg.max_length,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        offsets = encoded.pop("offset_mapping")
        special = encoded.pop("special_tokens_mask")
        evidence = np.zeros(len(offsets), dtype=np.float32)
        valid_mask = np.ones(len(offsets), dtype=np.float32)
        for token_idx, ((start, end), is_special) in enumerate(zip(offsets, special)):
            if is_special or end <= start:
                valid_mask[token_idx] = 0
                continue
            if any(start < gold_end and end > gold_start for gold_start, gold_end in row["evidence_offsets"]):
                evidence[token_idx] = 1
        return {
            "row_index": index,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "evidence_labels": evidence,
            "evidence_valid_mask": valid_mask,
            "risk_label": int(row["risk_id"]),
            "factor_labels": self.factor_targets[index],
            "offset_mapping": offsets,
        }


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        model_features = [
            {"input_ids": x["input_ids"], "attention_mask": x["attention_mask"]}
            for x in features
        ]
        batch = self.tokenizer.pad(model_features, padding=True, return_tensors="pt")
        width = batch["input_ids"].shape[1]
        evidence = torch.zeros((len(features), width), dtype=torch.float32)
        valid = torch.zeros_like(evidence)
        offsets = []
        for i, feature in enumerate(features):
            n = len(feature["evidence_labels"])
            evidence[i, :n] = torch.as_tensor(feature["evidence_labels"])
            valid[i, :n] = torch.as_tensor(feature["evidence_valid_mask"])
            offsets.append(feature["offset_mapping"])
        batch.update({
            "row_index": torch.tensor([x["row_index"] for x in features]),
            "risk_labels": torch.tensor([x["risk_label"] for x in features]),
            "factor_labels": torch.tensor(np.stack([x["factor_labels"] for x in features])),
            "evidence_labels": evidence,
            "evidence_valid_mask": valid,
            "offset_mapping": offsets,
        })
        return batch


class MultiTaskModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(cfg.model_name)
        hidden = self.encoder.config.hidden_size
        dropout = getattr(self.encoder.config, "classifier_dropout", None) or 0.1
        self.dropout = nn.Dropout(dropout)
        self.risk_head = nn.Linear(hidden, len(RISK_LABELS))
        self.factor_head = nn.Linear(hidden, len(FACTOR_LABELS))
        self.evidence_head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # Masked mean pooling is more stable than relying on architecture-specific CLS behavior.
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        pooled = self.dropout(pooled)
        return {
            "risk_logits": self.risk_head(pooled),
            "factor_logits": self.factor_head(pooled),
            "evidence_logits": self.evidence_head(self.dropout(hidden)).squeeze(-1),
        }


class FactorLabelAttentionModel(nn.Module):
    """A separate label-wise attention network for the 24 factor labels."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(cfg.model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.15)
        self.label_queries = nn.Parameter(torch.empty(len(FACTOR_LABELS), hidden))
        self.label_weights = nn.Parameter(torch.empty(len(FACTOR_LABELS), hidden))
        self.label_bias = nn.Parameter(torch.zeros(len(FACTOR_LABELS)))
        nn.init.normal_(self.label_queries, std=0.02)
        nn.init.normal_(self.label_weights, std=0.02)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = self.dropout(hidden)
        # [batch, labels, tokens]. Each factor learns where to look in the post.
        scores = torch.einsum("bth,lh->blt", hidden, self.label_queries)
        scores = scores.masked_fill(~attention_mask[:, None, :].bool(), -1e4)
        attention = torch.softmax(scores, dim=-1)
        context = torch.einsum("blt,bth->blh", attention, hidden)
        return (context * self.label_weights[None, :, :]).sum(-1) + self.label_bias


def asymmetric_loss(logits, targets, gamma_neg=4.0, gamma_pos=1.0, clip=0.05):
    probabilities = torch.sigmoid(logits)
    positive = probabilities
    negative = 1 - probabilities
    if clip:
        negative = (negative + clip).clamp(max=1)
    loss = targets * torch.log(positive.clamp_min(1e-8))
    loss += (1 - targets) * torch.log(negative.clamp_min(1e-8))
    probability_of_target = positive * targets + negative * (1 - targets)
    weight = torch.pow(
        1 - probability_of_target,
        gamma_pos * targets + gamma_neg * (1 - targets),
    )
    return -(loss * weight).mean()


def compute_loss(outputs, batch, risk_weights, cfg: Config):
    risk_loss = F.cross_entropy(
        outputs["risk_logits"], batch["risk_labels"], weight=risk_weights,
        label_smoothing=0.03,
    )
    factor_loss = asymmetric_loss(
        outputs["factor_logits"], batch["factor_labels"],
        cfg.factor_gamma_neg, cfg.factor_gamma_pos,
    )
    token_loss = F.binary_cross_entropy_with_logits(
        outputs["evidence_logits"], batch["evidence_labels"],
        reduction="none",
        pos_weight=torch.tensor(cfg.evidence_positive_weight, device=risk_loss.device),
    )
    evidence_loss = (
        token_loss * batch["evidence_valid_mask"]
    ).sum() / batch["evidence_valid_mask"].sum().clamp_min(1)
    total = (
        cfg.risk_loss_weight * risk_loss
        + cfg.factor_loss_weight * factor_loss
        + cfg.evidence_loss_weight * evidence_loss
    )
    return total, {
        "risk": risk_loss.item(), "factor": factor_loss.item(),
        "evidence": evidence_loss.item(),
    }


def phrase_f1_one(gold: list[str], predicted: list[str]) -> float:
    used = set()
    hits = 0
    for phrase in predicted:
        p = " ".join(phrase.casefold().split())
        p_tokens = len(WORD_RE.findall(phrase))
        for j, gold_phrase in enumerate(gold):
            if j in used:
                continue
            g = " ".join(gold_phrase.casefold().split())
            g_tokens = max(1, len(WORD_RE.findall(gold_phrase)))
            if p_tokens <= 3 * g_tokens and (p in g or g in p):
                hits += 1
                used.add(j)
                break
    precision = hits / len(predicted) if predicted else (1.0 if not gold else 0.0)
    recall = hits / len(gold) if gold else (1.0 if not predicted else 0.0)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def decode_evidence(
    text: str, offsets: list[tuple[int, int]], probabilities: np.ndarray,
    threshold: float, max_spans: int,
) -> list[str]:
    active = []
    for i, ((start, end), probability) in enumerate(zip(offsets, probabilities)):
        if end > start and probability >= threshold:
            active.append((i, start, end, float(probability)))
    if not active:
        return []
    groups, current = [], [active[0]]
    for item in active[1:]:
        if item[0] <= current[-1][0] + 1:
            current.append(item)
        else:
            groups.append(current)
            current = [item]
    groups.append(current)
    candidates = []
    for group in groups:
        start, end = group[0][1], group[-1][2]
        phrase = text[start:end].strip(" \t\r\n,;:-")
        if phrase:
            candidates.append((float(np.mean([x[3] for x in group])), start, phrase))
    candidates.sort(key=lambda x: (-x[0], len(WORD_RE.findall(x[2])), x[1]))
    selected = sorted(candidates[:max_spans], key=lambda x: x[1])
    return [x[2] for x in selected]


def tune_factor_thresholds(gold: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    thresholds = np.zeros(gold.shape[1])
    for j in range(gold.shape[1]):
        best = (-1.0, 0.5)
        for threshold in np.arange(0.05, 0.81, 0.025):
            score = f1_score(gold[:, j], probabilities[:, j] >= threshold, zero_division=0)
            if score > best[0]:
                best = score, float(threshold)
        thresholds[j] = best[1]
    return thresholds


@torch.no_grad()
def predict(model, loader, frame, device):
    model.eval()
    risk, factors = [], []
    evidence_probabilities, all_offsets, indices = [], [], []
    for batch in loader:
        offsets = batch.pop("offset_mapping")
        row_index = batch.pop("row_index")
        for key in list(batch):
            batch[key] = batch[key].to(device)
        outputs = model(batch["input_ids"], batch["attention_mask"])
        risk.append(torch.softmax(outputs["risk_logits"], -1).cpu().numpy())
        factors.append(torch.sigmoid(outputs["factor_logits"]).cpu().numpy())
        token_probs = torch.sigmoid(outputs["evidence_logits"]).cpu().numpy()
        for i, mapping in enumerate(offsets):
            evidence_probabilities.append(token_probs[i, :len(mapping)])
            all_offsets.append(mapping)
        indices.extend(row_index.numpy().tolist())
    return {
        "risk": np.concatenate(risk),
        "factors": np.concatenate(factors),
        "evidence_probabilities": evidence_probabilities,
        "offsets": all_offsets,
        "indices": np.array(indices),
    }


def evaluate_predictions(frame, prediction, cfg: Config, tune=True) -> dict:
    order = np.argsort(prediction["indices"])
    risk_prob = prediction["risk"][order]
    factor_prob = prediction["factors"][order]
    evidence_prob = [prediction["evidence_probabilities"][i] for i in order]
    offsets = [prediction["offsets"][i] for i in order]
    risk_gold = frame["risk_id"].to_numpy()
    factor_gold = np.zeros((len(frame), len(FACTOR_LABELS)), dtype=int)
    factor_to_id = {x: i for i, x in enumerate(FACTOR_LABELS)}
    for i, labels in enumerate(frame["factor_labels"]):
        for label in labels:
            factor_gold[i, factor_to_id[label]] = 1
    thresholds = tune_factor_thresholds(factor_gold, factor_prob) if tune else np.full(len(FACTOR_LABELS), 0.5)
    risk_pred = risk_prob.argmax(1)
    factor_macro = f1_score(
        factor_gold, factor_prob >= thresholds, average="macro", zero_division=0
    )
    best_evidence = (-1.0, 0.5, [])
    for threshold in np.arange(0.20, 0.81, 0.05):
        decoded = []
        for i, row in frame.reset_index(drop=True).iterrows():
            if risk_pred[i] == 0:
                decoded.append([])
            else:
                decoded.append(decode_evidence(
                    row["post"], offsets[i], evidence_prob[i],
                    float(threshold), cfg.evidence_max_spans,
                ))
        score = float(np.mean([
            phrase_f1_one(gold, pred)
            for gold, pred in zip(frame["evidence_spans"], decoded)
        ]))
        if score > best_evidence[0]:
            best_evidence = score, float(threshold), decoded
    risk_f1 = f1_score(risk_gold, risk_pred, average="weighted")
    phrase_f1 = best_evidence[0]
    return {
        "risk_weighted_f1": risk_f1,
        "factor_macro_f1": factor_macro,
        "phrase_f1": phrase_f1,
        "composite": 0.4 * risk_f1 + 0.3 * phrase_f1 + 0.3 * factor_macro,
        "factor_thresholds": thresholds,
        "evidence_threshold": best_evidence[1],
        "risk_predictions": risk_pred,
        "evidence_predictions": best_evidence[2],
        "classification_report": classification_report(
            risk_gold, risk_pred, target_names=RISK_LABELS, digits=4
        ),
    }


def train_one_fold(train_path: str | Path, cfg: Config) -> dict:
    seed_everything(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_training_frame(train_path)
    fit_idx, val_idx = grouped_split(frame, cfg)
    fit, valid = frame.iloc[fit_idx].reset_index(drop=True), frame.iloc[val_idx].reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    train_data, valid_data = SuicideDataset(fit, tokenizer, cfg), SuicideDataset(valid, tokenizer, cfg)
    collator = Collator(tokenizer)
    train_loader = DataLoader(
        train_data, batch_size=cfg.train_batch_size, shuffle=True,
        collate_fn=collator, num_workers=cfg.num_workers,
    )
    valid_loader = DataLoader(
        valid_data, batch_size=cfg.eval_batch_size, shuffle=False,
        collate_fn=collator, num_workers=cfg.num_workers,
    )
    device = choose_device()
    model = MultiTaskModel(cfg).to(device)
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
    encoder_params, head_params = [], []
    for name, param in model.named_parameters():
        (encoder_params if name.startswith("encoder.") else head_params).append(param)
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": cfg.learning_rate},
        {"params": head_params, "lr": cfg.head_learning_rate},
    ], weight_decay=cfg.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / cfg.grad_accumulation)
    total_updates = updates_per_epoch * cfg.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_updates * cfg.warmup_ratio), total_updates
    )
    counts = np.bincount(fit["risk_id"], minlength=len(RISK_LABELS))
    risk_weights = torch.tensor(len(fit) / (len(RISK_LABELS) * counts), dtype=torch.float32, device=device)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_score, stale = -1.0, 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch.pop("offset_mapping")
            batch.pop("row_index")
            for key in list(batch):
                batch[key] = batch[key].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(batch["input_ids"], batch["attention_mask"])
                loss, _ = compute_loss(outputs, batch, risk_weights, cfg)
                loss = loss / cfg.grad_accumulation
            scaler.scale(loss).backward()
            running += loss.item() * cfg.grad_accumulation
            if step % cfg.grad_accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        prediction = predict(model, valid_loader, valid, device)
        metrics = evaluate_predictions(valid, prediction, cfg, tune=True)
        record = {
            "epoch": epoch, "train_loss": running / len(train_loader),
            **{k: metrics[k] for k in ("risk_weighted_f1", "phrase_f1", "factor_macro_f1", "composite")},
        }
        history.append(record)
        print(json.dumps(record, indent=2))
        if metrics["composite"] > best_score:
            best_score, stale = metrics["composite"], 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            tokenizer.save_pretrained(output_dir / "tokenizer")
            serializable = {
                **record, "evidence_threshold": metrics["evidence_threshold"],
                "factor_thresholds": dict(zip(FACTOR_LABELS, metrics["factor_thresholds"].tolist())),
                "classification_report": metrics["classification_report"],
            }
            (output_dir / "best_metrics.json").write_text(json.dumps(serializable, indent=2))
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    return {
        "best_composite": best_score, "history": history,
        "fit_rows": len(fit), "validation_rows": len(valid),
        "device": str(device), "output_dir": str(output_dir),
        "unlocated_evidence_count": int(frame["unlocated_evidence"].map(len).sum()),
    }


def predict_leaderboard_from_checkpoint(
    checkpoint_dir: str | Path,
    leaderboard_path: str | Path,
    output_csv: str | Path,
    factor_checkpoint_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load a completed checkpoint and create a rule-compliant submission.

    This performs inference only; it never calls ``train_one_fold``.
    """
    checkpoint_dir = Path(checkpoint_dir)
    required = [
        checkpoint_dir / "best_model.pt",
        checkpoint_dir / "best_metrics.json",
        checkpoint_dir / "config.json",
        checkpoint_dir / "tokenizer",
    ]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint artifacts: {missing}")
    cfg = Config(**json.loads((checkpoint_dir / "config.json").read_text()))
    saved_metrics = json.loads((checkpoint_dir / "best_metrics.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / "tokenizer", use_fast=True)
    device = choose_device()
    model = MultiTaskModel(cfg)
    state = torch.load(checkpoint_dir / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    board = pd.read_excel(leaderboard_path).copy()
    required_columns = {"row_id", "post"}
    if not required_columns.issubset(board.columns):
        raise ValueError(f"Leaderboard is missing: {sorted(required_columns - set(board.columns))}")
    if board["row_id"].isna().any() or not board["row_id"].is_unique:
        raise ValueError("Leaderboard row_id values must be present and unique")
    board["post"] = board["post"].fillna("").astype(str)
    # Reuse the tested labeled Dataset/Collator with inference-only dummy targets.
    inference_frame = board.copy()
    inference_frame["risk_id"] = 0
    inference_frame["factor_labels"] = [[] for _ in range(len(inference_frame))]
    inference_frame["evidence_offsets"] = [[] for _ in range(len(inference_frame))]
    data = SuicideDataset(inference_frame, tokenizer, cfg)
    loader = DataLoader(
        data, batch_size=cfg.eval_batch_size, shuffle=False,
        collate_fn=Collator(tokenizer), num_workers=cfg.num_workers,
    )
    raw = predict(model, loader, inference_frame, device)
    order = np.argsort(raw["indices"])
    risk_probability = raw["risk"][order]
    factor_probability = raw["factors"][order]
    token_probability = [raw["evidence_probabilities"][i] for i in order]
    offsets = [raw["offsets"][i] for i in order]
    risk_id = risk_probability.argmax(1)
    factor_metrics = saved_metrics
    if factor_checkpoint_dir is not None:
        # Release the shared model before loading the separate factor encoder;
        # this substantially lowers peak memory on Apple silicon.
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        factor_checkpoint_dir = Path(factor_checkpoint_dir)
        factor_required = [
            factor_checkpoint_dir / "best_factor_model.pt",
            factor_checkpoint_dir / "best_factor_metrics.json",
        ]
        factor_missing = [str(x) for x in factor_required if not x.exists()]
        if factor_missing:
            raise FileNotFoundError(f"Missing factor artifacts: {factor_missing}")
        factor_model = FactorLabelAttentionModel(cfg)
        factor_state = torch.load(
            factor_checkpoint_dir / "best_factor_model.pt",
            map_location="cpu", weights_only=True,
        )
        factor_model.load_state_dict(factor_state, strict=True)
        factor_model.to(device).eval()
        optimized_probability, optimized_indices = predict_factor_attention(
            factor_model, loader, device
        )
        factor_probability = optimized_probability[np.argsort(optimized_indices)]
        factor_metrics = json.loads(
            (factor_checkpoint_dir / "best_factor_metrics.json").read_text()
        )
    factor_thresholds = np.array([
        factor_metrics["factor_thresholds"][label] for label in FACTOR_LABELS
    ])
    evidence_threshold = float(saved_metrics["evidence_threshold"])
    evidence_predictions = []
    for i, row in board.reset_index(drop=True).iterrows():
        if risk_id[i] == 0:
            evidence_predictions.append([])
        else:
            decoded = decode_evidence(
                row["post"], offsets[i], token_probability[i],
                evidence_threshold, cfg.evidence_max_spans,
            )
            # Semicolon is the submission delimiter, so it cannot remain
            # inside an individual span. Each piece is still verbatim.
            safe_spans = []
            for phrase in decoded:
                safe_spans.extend(x.strip() for x in re.split(r";|\n", phrase) if x.strip())
            evidence_predictions.append(safe_spans[:cfg.evidence_max_spans])
    factor_predictions = [
        [FACTOR_LABELS[j] for j in np.flatnonzero(probability >= factor_thresholds)]
        for probability in factor_probability
    ]
    submission = pd.DataFrame({
        "row_id": board["row_id"].astype(str),
        "risk_level": [RISK_LABELS[i] for i in risk_id],
        "evidence": ["; ".join(x) for x in evidence_predictions],
        "factors": [json.dumps(x, ensure_ascii=False) for x in factor_predictions],
    })
    # Strict format and verbatim-evidence checks before writing anything.
    if len(submission) != len(board):
        raise AssertionError("Submission row count mismatch")
    if set(submission["risk_level"]) - set(RISK_LABELS):
        raise AssertionError("Invalid risk label")
    for post, phrases in zip(board["post"], evidence_predictions):
        for phrase in phrases:
            if phrase not in post:
                raise AssertionError(f"Evidence is not verbatim: {phrase!r}")
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_csv, index=False)
    diagnostics = {
        "checkpoint": str(checkpoint_dir),
        "factor_checkpoint": str(factor_checkpoint_dir) if factor_checkpoint_dir else "shared_head",
        "leaderboard_rows": len(board),
        "risk_counts": submission["risk_level"].value_counts().to_dict(),
        "empty_evidence_rows": int(submission["evidence"].eq("").sum()),
        "average_factors": float(np.mean([len(x) for x in factor_predictions])),
        "output_csv": str(output_csv),
    }
    output_csv.with_suffix(".diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    return submission


@torch.no_grad()
def predict_factor_attention(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities, indices = [], []
    for batch in loader:
        batch.pop("offset_mapping")
        row_index = batch.pop("row_index")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids, attention_mask)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        indices.extend(row_index.numpy().tolist())
    return np.concatenate(probabilities), np.asarray(indices)


def train_factor_label_attention(
    train_path: str | Path,
    source_checkpoint_dir: str | Path,
    output_dir: str | Path = "outputs/factor_label_attention/fold_0_seed_42",
    epochs: int = 5,
    patience: int = 2,
) -> dict:
    """Optimize Task 2 only, initialized from the completed shared encoder.

    The source checkpoint is read-only. Risk/evidence heads are neither loaded
    into the optimizer nor changed, so their completed training is preserved.
    """
    source_checkpoint_dir, output_dir = Path(source_checkpoint_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config(**json.loads((source_checkpoint_dir / "config.json").read_text()))
    seed_everything(cfg.seed)
    frame = load_training_frame(train_path)
    fit_idx, val_idx = grouped_split(frame, cfg)
    fit = frame.iloc[fit_idx].reset_index(drop=True)
    valid = frame.iloc[val_idx].reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(source_checkpoint_dir / "tokenizer", use_fast=True)
    collator = Collator(tokenizer)
    train_loader = DataLoader(
        SuicideDataset(fit, tokenizer, cfg), batch_size=cfg.train_batch_size,
        shuffle=True, collate_fn=collator, num_workers=cfg.num_workers,
    )
    valid_loader = DataLoader(
        SuicideDataset(valid, tokenizer, cfg), batch_size=cfg.eval_batch_size,
        shuffle=False, collate_fn=collator, num_workers=cfg.num_workers,
    )
    device = choose_device()
    model = FactorLabelAttentionModel(cfg)
    source_state = torch.load(
        source_checkpoint_dir / "best_model.pt", map_location="cpu", weights_only=True
    )
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in source_state.items() if key.startswith("encoder.")
    }
    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected encoder parameters: {unexpected}")
    print(f"Initialized encoder; missing={len(missing)}, unexpected={len(unexpected)}")
    model.to(device)
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
    head_parameters = [model.label_queries, model.label_weights, model.label_bias]
    head_ids = {id(x) for x in head_parameters}
    encoder_parameters = [x for x in model.parameters() if id(x) not in head_ids]
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": 1e-5},
        {"params": head_parameters, "lr": 2e-4},
    ], weight_decay=cfg.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / cfg.grad_accumulation)
    total_updates = updates_per_epoch * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_updates * cfg.warmup_ratio), total_updates
    )
    factor_to_id = {x: i for i, x in enumerate(FACTOR_LABELS)}
    fit_gold = np.zeros((len(fit), len(FACTOR_LABELS)), dtype=np.float32)
    valid_gold = np.zeros((len(valid), len(FACTOR_LABELS)), dtype=np.float32)
    for matrix, part in ((fit_gold, fit), (valid_gold, valid)):
        for i, labels in enumerate(part["factor_labels"]):
            for label in labels:
                matrix[i, factor_to_id[label]] = 1
    positive = fit_gold.sum(0)
    # Square-root balancing is less unstable than full inverse prevalence for
    # labels with only a handful of examples.
    pos_weight = torch.tensor(
        np.sqrt((len(fit) - positive) / np.maximum(positive, 1)).clip(1, 10),
        dtype=torch.float32, device=device,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_macro, stale, history = -1.0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch.pop("offset_mapping")
            batch.pop("row_index")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["factor_labels"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids, attention_mask)
                loss = F.binary_cross_entropy_with_logits(
                    logits, targets, pos_weight=pos_weight
                ) / cfg.grad_accumulation
            scaler.scale(loss).backward()
            running += loss.item() * cfg.grad_accumulation
            if step % cfg.grad_accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        probabilities, indices = predict_factor_attention(model, valid_loader, device)
        order = np.argsort(indices)
        probabilities = probabilities[order]
        thresholds = tune_factor_thresholds(valid_gold, probabilities)
        macro = f1_score(
            valid_gold, probabilities >= thresholds, average="macro", zero_division=0
        )
        record = {"epoch": epoch, "train_loss": running / len(train_loader), "factor_macro_f1": macro}
        history.append(record)
        print(json.dumps(record, indent=2))
        if macro > best_macro:
            best_macro, stale = macro, 0
            torch.save(model.state_dict(), output_dir / "best_factor_model.pt")
            tokenizer.save_pretrained(output_dir / "tokenizer")
            (output_dir / "best_factor_metrics.json").write_text(json.dumps({
                **record,
                "factor_thresholds": dict(zip(FACTOR_LABELS, thresholds.tolist())),
                "source_checkpoint": str(source_checkpoint_dir),
            }, indent=2))
        else:
            stale += 1
            if stale >= patience:
                break
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    return {
        "best_factor_macro_f1": best_macro,
        "history": history,
        "output_dir": str(output_dir),
        "source_checkpoint_unchanged": str(source_checkpoint_dir),
    }
