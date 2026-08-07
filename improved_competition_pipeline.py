#!/usr/bin/env python3
"""Leakage-safe CV and a stronger CPU baseline for both competition subtasks."""
from __future__ import annotations

import argparse, ast, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MultiLabelBinarizer

RISK = ["Indicator", "Ideation", "Behavior", "Attempt"]
FACTORS = [
 "mental health issues","physical health/characteristic","substance use","hopelessness",
 "emotion dysregulation","low self-esteem","poor school performance","low socio-economic status",
 "interpersonal violence","prior self-harm or suicidal thought/attempt","poor social support",
 "interpersonal difficulty","dysfunctional family","exposure to others' suicide","stressful life event",
 "traumatic experience","cognitive deficits","suicide means (with access)",
 "sexual orientation related issues","social support","coping strategy","psychological capital",
 "sense of responsibility","meaning in life"]
NONE = {"", "none", "nan", "n/a", "na", "null"}
TRIGGER = re.compile(
 r"\\b(suicid\\w*|kill(?:ing|ed)? myself|die|dead|death|end my life|"
 r"overdos\\w*|hang\\w*|jump\\w*|gun|pills?|rope|knife|cut(?:ting)?|attempt\\w*)\\b", re.I)
TOK = re.compile(r"\\b[\\w']+\\b")

def parse_factors(x):
    if pd.isna(x): return []
    try: xs = ast.literal_eval(x) if isinstance(x, str) else x
    except (ValueError, SyntaxError): return []
    return sorted(set(str(v).strip() for v in xs if str(v).strip() in FACTORS))

def spans(x):
    if pd.isna(x): return []
    return [s.strip() for s in str(x).split(";") if s.strip().lower() not in NONE]

def make_vectors(train_text, other_text):
    # Word semantics + character morphology/typos is consistently stronger than word TF-IDF alone.
    w = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=.995, sublinear_tf=True,
                        strip_accents="unicode", max_features=60_000)
    c = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2, max_df=.995,
                        sublinear_tf=True, max_features=80_000)
    a=w.fit_transform(train_text); b=w.transform(other_text)
    ac=c.fit_transform(train_text); bc=c.transform(other_text)
    return hstack([a,ac]).tocsr(), hstack([b,bc]).tocsr()

def best_thresholds(y, p):
    out=[]
    for j in range(y.shape[1]):
        best=(0.5,-1)
        for t in np.arange(.05,.76,.025):
            z=f1_score(y[:,j], p[:,j]>=t, zero_division=0)
            if z>best[1]: best=(float(t),z)
        out.append(best[0])
    return np.array(out)

def official_phrase_f1(golds, preds):
    vals=[]
    for gs, ps in zip(golds, preds):
        used=set(); hit=0
        for p in ps:
            pn=" ".join(p.lower().split()); pw=len(TOK.findall(p))
            choices=[(i,g) for i,g in enumerate(gs) if i not in used]
            for i,g in choices:
                gn=" ".join(g.lower().split()); gw=max(1,len(TOK.findall(g)))
                if pw<=3*gw and (pn in gn or gn in pn):
                    used.add(i); hit+=1; break
        pr=hit/len(ps) if ps else (1.0 if not gs else 0.0)
        rc=hit/len(gs) if gs else (1.0 if not ps else 0.0)
        vals.append(2*pr*rc/(pr+rc) if pr+rc else 0.0)
    return float(np.mean(vals))

def evidence(post, risk):
    if risk=="Indicator": return []
    # Short verbatim windows centered on explicit risk expressions; 1 span minimizes FP cost.
    matches=list(TRIGGER.finditer(post))
    if not matches: return []
    ranked=[]
    for m in matches:
        left=max(post.rfind(".",0,m.start()),post.rfind("!",0,m.start()),post.rfind("?",0,m.start()))
        rights=[x for x in (post.find(".",m.end()),post.find("!",m.end()),post.find("?",m.end())) if x>=0]
        right=min(rights) if rights else len(post)
        clause=post[left+1:right+1].strip()
        words=list(TOK.finditer(clause))
        if len(words)>18:
            rel=m.start()-(left+1); k=min(range(len(words)),key=lambda i:abs(words[i].start()-rel))
            lo=max(0,k-7); hi=min(len(words),k+9)
            clause=clause[words[lo].start():words[hi-1].end()]
        score=3*bool(re.search(r"kill.*myself|suicid|attempt|overdos",clause,re.I))+bool(re.search(r"plan|tonight|tomorrow|tried|hospital",clause,re.I))
        ranked.append((score,-len(clause),clause))
    return [max(ranked)[2]]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--train",default="train.xlsx")
    ap.add_argument("--test",default="leaderboard.xlsx"); ap.add_argument("--output",default="outputs/ImprovedTeam.csv")
    ap.add_argument("--folds",type=int,default=5); args=ap.parse_args()
    tr=pd.read_excel(args.train); te=pd.read_excel(args.test)
    tr["post"]=tr.post.fillna("").astype(str); te["post"]=te.post.fillna("").astype(str)
    y=np.array([RISK.index(str(x).strip().title()) for x in tr["suicide risk"]])
    mlb=MultiLabelBinarizer(classes=FACTORS); yf=mlb.fit_transform(tr.factors.map(parse_factors))
    cv=StratifiedGroupKFold(args.folds,shuffle=True,random_state=42)
    oof=np.zeros((len(tr),4)); ooff=np.zeros_like(yf,dtype=float)
    fold_scores=[]
    for k,(a,b) in enumerate(cv.split(tr.post,y,tr.anon_user_id),1):
        xa,xb=make_vectors(tr.post.iloc[a],tr.post.iloc[b])
        r=LogisticRegression(C=4,max_iter=2500,class_weight="balanced",solver="lbfgs").fit(xa,y[a])
        oof[b]=r.predict_proba(xb); fold_scores.append(f1_score(y[b],oof[b].argmax(1),average="weighted"))
        for j in range(len(FACTORS)):
            if yf[a,j].min()==yf[a,j].max(): ooft=np.repeat(yf[a,j],len(b))
            else:
                m=LogisticRegression(C=2,max_iter=1500,class_weight="balanced",solver="liblinear").fit(xa,yf[a,j])
                ooft=m.predict_proba(xb)[:,1]
            ooff[b,j]=ooft
        print(f"fold {k}: risk={fold_scores[-1]:.4f}")
    th=best_thresholds(yf,ooof:=ooff)
    print("OOF risk weighted F1",f1_score(y,oof.argmax(1),average="weighted"))
    print("OOF factor macro F1",f1_score(yf,ooof>=th,average="macro",zero_division=0))
    pred_ev=[evidence(p,RISK[q]) for p,q in zip(tr.post,oof.argmax(1))]
    print("OOF heuristic phrase F1",official_phrase_f1(tr["evidence for suicide risk level"].map(spans),pred_ev))
    x,xt=make_vectors(tr.post,te.post)
    risk=LogisticRegression(C=4,max_iter=2500,class_weight="balanced",solver="lbfgs").fit(x,y).predict(xt)
    fp=np.zeros((len(te),len(FACTORS)))
    for j in range(len(FACTORS)):
        m=LogisticRegression(C=2,max_iter=1500,class_weight="balanced",solver="liblinear").fit(x,yf[:,j])
        fp[:,j]=m.predict_proba(xt)[:,1]
    out=pd.DataFrame({"row_id":te.row_id,"risk_level":[RISK[i] for i in risk]})
    out["evidence"]=["; ".join(evidence(p,r)) for p,r in zip(te.post,out.risk_level)]
    out["factors"]=[json.dumps([FACTORS[j] for j in np.flatnonzero(row>=th)]) for row in fp]
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(args.output,index=False)
    Path(args.output).with_suffix(".metrics.json").write_text(json.dumps({
      "risk_weighted_f1":f1_score(y,oof.argmax(1),average="weighted"),
      "factor_macro_f1":f1_score(yf,ooof>=th,average="macro",zero_division=0),
      "factor_thresholds":dict(zip(FACTORS,th.tolist()))},indent=2))
    print("saved",args.output)
if __name__=="__main__": main()
