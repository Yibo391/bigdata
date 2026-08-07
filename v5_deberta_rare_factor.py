#!/usr/bin/env python3
"""V5 Task 2: DeBERTa + rare-label learning + hybrid ensemble.

This is independent from V1-V4. It evaluates train.xlsx only and never reads
leaderboard.xlsx or creates a submission.
"""
from __future__ import annotations

import ast
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from v2_factor_ensemble import (
    FACTOR_LABELS, RISK_LABELS, V2Config, load_frame,
    top_k_prediction, tune_prevalence_quotas,
)


RARE_RULES = {
    "sexual orientation related issues": [
        r"\blgbtq?\b", r"\bgay\b", r"\blesbian\b", r"\bbisexual\b",
        r"\bhomosexual\b", r"\btransgender\b", r"\btransphob\w*\b",
        r"\bhomophob\w*\b", r"\bcoming out\b", r"\bsexual orientation\b",
        r"\bgender identity\b", r"\bnon[ -]?binary\b",
    ],
    "exposure to others' suicide": [
        r"\b(friend|relative|brother|sister|mother|father|mom|dad|classmate|coworker|partner|someone)\b.{0,100}\b(committed suicide|died by suicide|killed (himself|herself|themself)|attempted suicide)\b",
        r"\b(committed suicide|died by suicide|killed (himself|herself|themself)|attempted suicide)\b.{0,100}\b(friend|relative|brother|sister|mother|father|mom|dad|classmate|coworker|partner|someone)\b",
    ],
    "poor school performance": [
        r"\bfail(?:ed|ing)? (?:my |the )?(?:class|classes|exam|exams|school|college)\b",
        r"\bbad grades?\b", r"\bgrades? (?:are|is|were) (?:bad|terrible|awful)\b",
        r"\bacademic(?:ally)?\b", r"\bstruggling (?:at|in) school\b", r"\bdropped out\b",
    ],
    "substance use": [
        r"\balcohol(?:ic|ism)?\b", r"\bdrunk\b", r"\bcocaine\b", r"\bheroin\b",
        r"\bmeth\b", r"\bopioids?\b", r"\bweed\b", r"\bmarijuana\b",
        r"\bdrug addiction\b", r"\baddicted to (?:drugs|alcohol|pills)\b",
        r"\bsubstance abuse\b", r"\bintoxicated\b",
    ],
    "cognitive deficits": [
        r"\bbrain fog\b", r"\bcan(?:not|'t) concentrate\b", r"\bcan(?:not|'t) focus\b",
        r"\bmemory loss\b", r"\bthink straight\b", r"\bcloudy mind\b",
    ],
    "meaning in life": [
        r"\bpurpose in life\b", r"\bmeaning (?:in|of) life\b",
        r"\breason to (?:live|stay alive|keep living)\b", r"\bsomething to live for\b",
        r"\blife is worth living\b",
    ],
    "low socio-economic status": [
        r"\bhomeless\w*\b", r"\bunemploy(?:ed|ment)\b", r"\bjobless\b",
        r"\blaid off\b", r"\bdebt\b", r"\bfinancial (?:problem|difficulty|struggle)\w*\b",
        r"\bcan(?:not|'t) afford\b", r"\bpoverty\b",
    ],
}


@dataclass
class V5Config:
    model_name: str = "microsoft/deberta-v3-base"
    max_length: int = 512
    folds: int = 5
    epochs: int = 4
    train_batch_size: int = 1
    eval_batch_size: int = 2
    gradient_accumulation: int = 8
    encoder_lr: float = 1.5e-5
    head_lr: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    dropout: float = 0.15
    rare_sampling_cap: float = 3.0
    gamma_negative: float = 4.0
    gamma_positive: float = 1.0
    negative_clip: float = 0.05
    risk_loss_weight: float = 0.10
    patience: int = 2
    seed: int = 73
    resume_completed_folds: bool = True
    prevalence_cap_multiplier: float = 2.5
    quota_shrink_to_gold: float = 0.35
    output_dir: str = "outputs/v5_deberta_rare_factor"
    v2_dir: str = "outputs/v2_factor_ensemble"
    v3_dir: str = "outputs/v3_fast_semantic"
    v4_dir: str = "outputs/v4_finetuned_factor"


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def choose_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def load_pretrained(factory, model_name, **kwargs):
    """Prefer cache after the first download, but allow that initial download."""
    try:
        return factory.from_pretrained(model_name, local_files_only=True, **kwargs)
    except (OSError, ValueError):
        return factory.from_pretrained(model_name, **kwargs)


def factor_counts(value):
    try: values = ast.literal_eval(value) if isinstance(value, str) else value
    except (SyntaxError, ValueError): values = []
    counts = {x: 0 for x in FACTOR_LABELS}
    for item in values if isinstance(values, (list, tuple)) else []:
        if str(item) in counts: counts[str(item)] += 1
    return counts


def balanced_group_folds(frame, y, risk, n_splits=5, seed=73):
    """Greedily balance factor counts, risk counts, and rows while keeping users intact."""
    rng = np.random.default_rng(seed)
    users = frame["anon_user_id"].astype(str).unique().tolist()
    rng.shuffle(users)
    records = []
    for user in users:
        idx = np.flatnonzero(frame["anon_user_id"].astype(str).to_numpy() == user)
        risk_count = np.bincount(risk[idx], minlength=len(RISK_LABELS))
        vector = np.r_[y[idx].sum(0), risk_count, len(idx)]
        records.append((user, idx, vector.astype(float)))
    total = np.stack([x[2] for x in records]).sum(0)
    rarity = 1.0 / np.maximum(total, 1.0)
    records.sort(key=lambda x: float(((x[2] * rarity) ** 2).sum()), reverse=True)
    fold_sum = np.zeros((n_splits, len(total)), dtype=float)
    fold_users = [[] for _ in range(n_splits)]
    target = total / n_splits
    for position, record in enumerate(records):
        user, idx, vector = record
        costs = []
        for fold in range(n_splits):
            proposed = fold_sum.copy()
            proposed[fold] += vector
            normalized = proposed / np.maximum(target[None, :], 1)
            # Minimize dispersion across all folds, with factors dominant and
            # risk/row totals as gentler tie-breakers.
            label_cost = np.std(normalized[:, :len(FACTOR_LABELS)], axis=0).mean()
            auxiliary_cost = np.std(normalized[:, len(FACTOR_LABELS):], axis=0).mean()
            user_cost = np.std([
                len(x) + int(i == fold) for i, x in enumerate(fold_users)
            ]) / max(1, (position + 1) / n_splits)
            costs.append(label_cost + 0.25 * auxiliary_cost + 0.05 * user_cost)
        chosen = int(np.argmin(costs))
        fold_sum[chosen] += vector
        fold_users[chosen].append(user)
    all_index = np.arange(len(frame))
    splits, fold_id = [], np.full(len(frame), -1, dtype=int)
    user_array = frame["anon_user_id"].astype(str).to_numpy()
    for fold, names in enumerate(fold_users):
        valid = np.flatnonzero(np.isin(user_array, names))
        fit = np.setdiff1d(all_index, valid, assume_unique=True)
        fold_id[valid] = fold
        splits.append((fit, valid))
    if (fold_id < 0).any(): raise RuntimeError("Fold assignment failed")
    return splits, fold_id


class FactorDataset(Dataset):
    def __init__(self, frame, y, risk, tokenizer, cfg):
        self.frame = frame.reset_index(drop=True); self.y = y.astype(np.float32)
        self.risk = risk.astype(np.int64); self.tokenizer = tokenizer; self.cfg = cfg
        self.confidence = np.ones_like(self.y, dtype=np.float32)
        lookup = {x: i for i, x in enumerate(FACTOR_LABELS)}
        for i, raw in enumerate(self.frame["factors"]):
            for label, count in factor_counts(raw).items():
                if count > 1: self.confidence[i, lookup[label]] = min(1.5, 1 + .12 * math.log1p(count - 1))

    def __len__(self): return len(self.frame)

    def __getitem__(self, index):
        enc = self.tokenizer(self.frame.iloc[index]["post"], truncation=True, max_length=self.cfg.max_length)
        return {"row_index": index, "input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "labels": self.y[index], "confidence": self.confidence[index], "risk": int(self.risk[index])}


class Collator:
    def __init__(self, tokenizer): self.tokenizer = tokenizer
    def __call__(self, rows):
        batch = self.tokenizer.pad([{"input_ids":x["input_ids"], "attention_mask":x["attention_mask"]} for x in rows], return_tensors="pt")
        batch.update({"row_index":torch.tensor([x["row_index"] for x in rows]),
                      "labels":torch.tensor(np.stack([x["labels"] for x in rows])),
                      "confidence":torch.tensor(np.stack([x["confidence"] for x in rows])),
                      "risk":torch.tensor([x["risk"] for x in rows])})
        return batch


class DebertaFactorModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Eager float32 attention avoids an MPS mixed-accumulator assertion in
        # DeBERTa's backward pass on Apple silicon.
        self.encoder = load_pretrained(
            AutoModel, cfg.model_name,
            attn_implementation="eager", dtype=torch.float32,
        ).float()
        hidden = self.encoder.config.hidden_size
        self.norm = nn.LayerNorm(hidden * 3)
        self.projection = nn.Linear(hidden * 3, hidden)
        self.dropout = nn.Dropout(cfg.dropout)
        self.factor_head = nn.Linear(hidden, len(FACTOR_LABELS))
        self.risk_head = nn.Linear(hidden, len(RISK_LABELS))

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).bool()
        mean = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        maximum = hidden.masked_fill(~mask, -1e4).max(1).values
        pooled = torch.cat([hidden[:, 0], mean, maximum], dim=1)
        pooled = self.dropout(F.gelu(self.projection(self.norm(pooled))))
        return self.factor_head(pooled), self.risk_head(pooled)


def asymmetric_loss(logits, target, confidence, positive_weight, cfg):
    positive_probability = torch.sigmoid(logits)
    negative_probability = 1 - positive_probability
    if cfg.negative_clip > 0:
        negative_probability = (negative_probability + cfg.negative_clip).clamp(max=1)
    log_likelihood = target * torch.log(positive_probability.clamp_min(1e-8))
    log_likelihood += (1 - target) * torch.log(negative_probability.clamp_min(1e-8))
    probability = positive_probability * target + negative_probability * (1 - target)
    gamma = cfg.gamma_positive * target + cfg.gamma_negative * (1 - target)
    focal = torch.pow(1 - probability, gamma)
    weights = torch.where(target > .5, confidence * positive_weight[None], torch.ones_like(target))
    return -(log_likelihood * focal * weights).mean()


def sample_weights(y, cap):
    support = np.maximum(y.sum(0), 1)
    multiplier = np.sqrt(support.max() / support).clip(1, cap)
    return np.array([max([1.0] + multiplier[row > 0].tolist()) for row in y], dtype=np.float64)


@torch.inference_mode()
def predict(model, loader, device):
    model.eval(); probability=[]; indices=[]
    for batch in loader:
        indices.extend(batch.pop("row_index").numpy().tolist())
        logits, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        probability.append(torch.sigmoid(logits).cpu().numpy())
    result=np.concatenate(probability); return result[np.argsort(indices)]


def safe_ap(target, probability):
    values=[average_precision_score(target[:,j], probability[:,j]) for j in range(target.shape[1]) if 0 < target[:,j].sum() < len(target)]
    return float(np.mean(values)) if values else 0.0


def prevalence_prediction(probability, fit_y):
    result=np.zeros_like(probability,dtype=np.int8)
    for j in range(fit_y.shape[1]): result[:,j]=top_k_prediction(probability[:,j], max(1,round(fit_y[:,j].mean()*len(result))))
    return result


def train_fold(frame, y, risk, fit_idx, valid_idx, fold, cfg):
    seed_all(cfg.seed + fold); out=Path(cfg.output_dir)/f"fold_{fold}"; out.mkdir(parents=True,exist_ok=True)
    # DeBERTa-v3 uses SentencePiece; the slow tokenizer is the native path and
    # avoids an unnecessary fast-tokenizer conversion warning.
    tokenizer=load_pretrained(
        AutoTokenizer, cfg.model_name, use_fast=False,
        fix_mistral_regex=True,
    )
    fit_frame=frame.iloc[fit_idx].reset_index(drop=True); valid_frame=frame.iloc[valid_idx].reset_index(drop=True)
    fit_data=FactorDataset(fit_frame,y[fit_idx],risk[fit_idx],tokenizer,cfg); valid_data=FactorDataset(valid_frame,y[valid_idx],risk[valid_idx],tokenizer,cfg)
    sampler=WeightedRandomSampler(sample_weights(y[fit_idx],cfg.rare_sampling_cap),num_samples=len(fit_data),replacement=True)
    train_loader=DataLoader(fit_data,batch_size=cfg.train_batch_size,sampler=sampler,collate_fn=Collator(tokenizer),num_workers=0)
    valid_loader=DataLoader(valid_data,batch_size=cfg.eval_batch_size,shuffle=False,collate_fn=Collator(tokenizer),num_workers=0)
    device=choose_device(); model=DebertaFactorModel(cfg).to(device)
    if hasattr(model.encoder,"gradient_checkpointing_enable"): model.encoder.gradient_checkpointing_enable()
    head_prefix=("norm","projection","factor_head","risk_head"); encoder=[]; head=[]
    for name,param in model.named_parameters(): (head if name.startswith(head_prefix) else encoder).append(param)
    optimizer=torch.optim.AdamW([{"params":encoder,"lr":cfg.encoder_lr},{"params":head,"lr":cfg.head_lr}],weight_decay=cfg.weight_decay)
    updates=math.ceil(len(train_loader)/cfg.gradient_accumulation)*cfg.epochs
    scheduler=get_cosine_schedule_with_warmup(optimizer,int(updates*cfg.warmup_ratio),updates)
    support=np.maximum(y[fit_idx].sum(0),1); positive_weight=torch.tensor((len(fit_idx)/support)**.25,dtype=torch.float32,device=device).clamp(1,3)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    best=-1.; best_probability=None; stale=0; history=[]
    for epoch in range(1,cfg.epochs+1):
        model.train(); optimizer.zero_grad(set_to_none=True); running=0.
        bar=tqdm(train_loader,desc=f"V5 fold {fold} epoch {epoch}",leave=False)
        for step,batch in enumerate(bar,1):
            batch.pop("row_index"); batch={k:v.to(device) for k,v in batch.items()}
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                factor_logits,risk_logits=model(batch["input_ids"],batch["attention_mask"])
                loss=asymmetric_loss(factor_logits,batch["labels"],batch["confidence"],positive_weight,cfg)
                loss=loss+cfg.risk_loss_weight*F.cross_entropy(risk_logits,batch["risk"],label_smoothing=.03)
            scaler.scale(loss/cfg.gradient_accumulation).backward(); running+=loss.item()
            if step%cfg.gradient_accumulation==0 or step==len(train_loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step()
            if step%100==0: bar.set_postfix(loss=f"{running/step:.3f}")
        probability=predict(model,valid_loader,device); pred=prevalence_prediction(probability,y[fit_idx])
        macro=f1_score(y[valid_idx],pred,average="macro",zero_division=0); ap=safe_ap(y[valid_idx],probability)
        record={"epoch":epoch,"loss":running/len(train_loader),"prevalence_macro_f1":macro,"macro_average_precision":ap}
        history.append(record); print(json.dumps({"fold":fold,**record},indent=2))
        selection=macro+.05*ap
        if selection>best:
            best=selection; stale=0; best_probability=probability.copy(); torch.save(model.state_dict(),out/"best_model.pt")
            tokenizer.save_pretrained(out/"tokenizer"); (out/"best_metrics.json").write_text(json.dumps(record,indent=2))
        else:
            stale+=1
            if stale>=cfg.patience: break
    np.save(out/"valid_probability.npy",best_probability)
    (out/"valid_row_ids.json").write_text(json.dumps(frame.iloc[valid_idx].row_id.astype(str).tolist()))
    (out/"history.json").write_text(json.dumps(history,indent=2))
    del model
    if device.type=="mps": torch.mps.empty_cache()
    if device.type=="cuda": torch.cuda.empty_cache()
    return best_probability,history


def rule_scores(texts):
    result=np.zeros((len(texts),len(FACTOR_LABELS)),dtype=np.float32)
    lookup={x:i for i,x in enumerate(FACTOR_LABELS)}
    for row,text in enumerate(texts):
        text=str(text).lower()
        for label,patterns in RARE_RULES.items():
            hits=sum(bool(re.search(pattern,text,re.I|re.S)) for pattern in patterns)
            if hits: result[row,lookup[label]]=min(1.,.7+.1*(hits-1))
    return result


def rank_columns(values):
    result=np.zeros_like(values,dtype=np.float32); n=len(values)
    for j in range(values.shape[1]):
        # Average tied ranks are essential for sparse rules: non-matching rows
        # must remain tied instead of receiving arbitrary row-order scores.
        result[:,j]=(rankdata(values[:,j],method="average")-.5)/n
    return result


def best_pair(target, source_map, label_index):
    names=list(source_map); support=max(1,int(target.sum())); best=(-1,None,None,0.)
    for left,right in combinations_with_replacement(names,2):
        for weight in (0.,.25,.5,.75,1.):
            score=(1-weight)*source_map[left][:,label_index]+weight*source_map[right][:,label_index]
            value=f1_score(target,top_k_prediction(score,support),zero_division=0)
            if value>best[0]: best=(value,left,right,weight)
    return best


def nested_ensemble(y, fold_id, sources, rules):
    ranked={name:rank_columns(value) for name,value in sources.items()}; ranked["rules"]=rank_columns(rules)
    result=np.zeros_like(next(iter(sources.values())),dtype=np.float32); choices=[]
    for fold in np.unique(fold_id):
        fit=fold_id!=fold; valid=~fit
        for j,label in enumerate(FACTOR_LABELS):
            allowed={k:v for k,v in ranked.items() if k!="rules" or label in RARE_RULES}
            fit_sources={k:v[fit] for k,v in allowed.items()}; _,left,right,weight=best_pair(y[fit,j],fit_sources,j)
            result[valid,j]=(1-weight)*allowed[left][valid,j]+weight*allowed[right][valid,j]
            choices.append({"fold":int(fold),"factor":label,"left":left,"right":right,"right_weight":weight})
    final=[]
    for j,label in enumerate(FACTOR_LABELS):
        allowed={k:v for k,v in ranked.items() if k!="rules" or label in RARE_RULES}
        score,left,right,weight=best_pair(y[:,j],allowed,j)
        final.append({"factor":label,"left":left,"right":right,"right_weight":weight,"selection_f1":score})
    return result,choices,final


def run_v5_oof(train_path="train.xlsx",cfg=None):
    cfg=cfg or V5Config(); seed_all(cfg.seed); output=Path(cfg.output_dir); output.mkdir(parents=True,exist_ok=True)
    frame,y,risk=load_frame(train_path); splits,fold_id=balanced_group_folds(frame,y,risk,cfg.folds,cfg.seed)
    oof=np.zeros_like(y,dtype=np.float32); histories=[]
    fold_balance=[]
    for fold,(fit_idx,valid_idx) in enumerate(splits):
        assert set(frame.iloc[fit_idx].anon_user_id).isdisjoint(set(frame.iloc[valid_idx].anon_user_id))
        fold_balance.append({"fold":fold,"rows":len(valid_idx),"users":int(frame.iloc[valid_idx].anon_user_id.nunique()),"positive_counts":y[valid_idx].sum(0).tolist()})
        out=output/f"fold_{fold}"; pf=out/"valid_probability.npy"; rf=out/"valid_row_ids.json"; hf=out/"history.json"
        ids=frame.iloc[valid_idx].row_id.astype(str).tolist()
        resume=cfg.resume_completed_folds and pf.exists() and rf.exists() and hf.exists() and json.loads(rf.read_text())==ids
        if resume:
            probability=np.load(pf); history=json.loads(hf.read_text()); print(f"V5 fold {fold}: using completed cached predictions")
        else: probability,history=train_fold(frame,y,risk,fit_idx,valid_idx,fold,cfg)
        if probability.shape!=(len(valid_idx),len(FACTOR_LABELS)): raise ValueError(f"Bad fold {fold} prediction shape")
        oof[valid_idx]=probability; histories.append(history)
    sources={"v5":oof}
    paths={"v2":Path(cfg.v2_dir)/"oof_predictions.npz","v3":Path(cfg.v3_dir)/"oof_predictions.npz","v4":Path(cfg.v4_dir)/"oof_predictions.npz"}
    keys={"v2":"blended","v3":"combined_probability","v4":"combined"}
    for name,path in paths.items():
        if path.exists():
            value=np.load(path)[keys[name]]
            if value.shape==y.shape: sources[name]=value
    rules=rule_scores(frame.post.tolist()); combined,nested_choices,final_choices=nested_ensemble(y,fold_id,sources,rules)
    fixed=np.zeros_like(y)
    for j in range(y.shape[1]): fixed[:,j]=top_k_prediction(combined[:,j],int(y[:,j].sum()))
    nested_fixed=f1_score(y,fixed,average="macro",zero_division=0)
    quota_cfg=V2Config(prevalence_cap_multiplier=cfg.prevalence_cap_multiplier,quota_shrink_to_gold=cfg.quota_shrink_to_gold)
    rates,prediction,_=tune_prevalence_quotas(y,combined,quota_cfg)
    calibrated=f1_score(y,prediction,average="macro",zero_division=0)
    v5_fixed=np.zeros_like(y)
    for j in range(y.shape[1]): v5_fixed[:,j]=top_k_prediction(oof[:,j],int(y[:,j].sum()))
    v5_macro=f1_score(y,v5_fixed,average="macro",zero_division=0)
    precision,recall,label_f1,support=precision_recall_fscore_support(y,prediction,average=None,zero_division=0)
    pd.DataFrame({"factor":FACTOR_LABELS,"support":support,"gold_rate":y.mean(0),"submission_rate":rates,"precision":precision,"recall":recall,"f1":label_f1}).to_csv(output/"oof_per_label.csv",index=False)
    np.savez_compressed(output/"oof_predictions.npz",y=y,v5=oof,combined=combined,prediction=prediction,fold_id=fold_id,rules=rules)
    metrics={"v5_only_fixed_quota_macro_f1":v5_macro,"nested_ensemble_fixed_quota_macro_f1":nested_fixed,"ensemble_calibrated_macro_f1":calibrated,"sources":list(sources),"fold_balance":fold_balance,"fold_history":histories}
    (output/"oof_metrics.json").write_text(json.dumps(metrics,indent=2)); (output/"config.json").write_text(json.dumps(asdict(cfg),indent=2)); (output/"nested_choices.json").write_text(json.dumps(nested_choices,indent=2)); (output/"final_ensemble_choices.json").write_text(json.dumps(final_choices,indent=2))
    return metrics
