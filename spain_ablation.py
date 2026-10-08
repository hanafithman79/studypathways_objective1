#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABLATION of the proposed dual-branch hybrid on the Spanish dropout data.

Requires spain_dropout_experiments.py, objective1_ablation.py and objective1_experiments.py in the same folder.
Each variant removes ONE thing from the full model (same folds, same hyper-parameters, 3 seeds per fold; the fold
value is the mean of the seeds). delta = variant - full; negative => the removed part helps.
Primary metric PR-AUC (threshold-free), plus ROC-AUC and share of dropouts captured in the top-10% highest risk.

Run:   HORIZON=H2 python spain_ablation.py            (QUICK_TEST=1 for a smoke test)
Env:   DATA_DIR / SPAIN_7Z (see spain_dropout_experiments.py), HORIZON (default H2), N_SEEDS (3), N_JOBS
"""
import os
import sys

os.environ.setdefault("EXTRA_HYBRIDS", "1")
os.environ["OUT_DIR"] = os.environ.get("OUT_DIR", "spain_outputs_ablation")
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    sys.path.insert(0, os.getcwd())

import time
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from joblib import Parallel, delayed

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

import objective1_experiments as base
import objective1_ablation as AB
import spain_dropout_experiments as S

warnings.filterwarnings("ignore")
QUICK = base.QUICK_TEST
H = os.environ.get("HORIZON", "H2")
SEEDS = [base.SEED + i for i in range(int(os.environ.get("N_SEEDS", 1 if QUICK else 3)))]
N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 2))
EPOCHS = 2 if QUICK else 15
BATCH = 512
OUT = base.OUT_DIR
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)

PRIOR = ["prior_1", "prior_2", "prior_3"]
CREDITS = ["cred_mat_total", "cred_mat_sem_a", "cred_mat_sem_b", "cred_sup_sem_a", "pass_rate_sem_a",
           "cred_sup_sem_b", "cred_sup_total", "pass_rate_total", "mean_grade", "share_failed", "n_subjects"]
ENTRY_NUM = ["nota10", "nota14", "preferencia_seleccion", "years_since_entry"]
ENTRY_CAT = ["tipo_ingreso"]
BACKGROUND_CAT = ["estudios_p_hash", "estudios_m_hash", "desplazado_hash", "campus_hash", "dedicacion"]

# key, label, group, overrides
VARIANTS = [
    ("full", "Full model (Proposed)", "reference", {}),
    ("no_static", "- static branch", "architecture", dict(use_static=False)),
    ("no_temporal", "- temporal branch", "architecture", dict(use_temporal=False)),
    ("no_tcn", "- causal TCN (MLP+LSTM)", "architecture", dict(tmode="lstm")),
    ("no_lstm", "- LSTM (MLP+TCN)", "architecture", dict(tmode="tcn_only")),
    ("no_residual", "- TCN residual connection", "architecture", dict(tmode="tcn_lstm_nores")),
    ("no_causal", "- causality (symmetric conv)", "architecture", dict(tmode="symconv_lstm")),
    ("no_bn", "- BatchNorm", "architecture", dict(batchnorm=False)),
    ("no_dropout", "- Dropout", "architecture", dict(dropout=0.0)),
    ("linear_fusion", "- fusion MLP (linear head)", "architecture", dict(fusion="linear")),
    ("no_cw", "- class-weighted loss", "training", dict(cw_power=0.0)),
    ("f_no_prior", "- prior-year pass rates", "features", dict(drop_num=PRIOR, zero_seq=(H == "H1"))),
    ("f_no_credits", "- credits / semester results", "features", dict(drop_num=CREDITS)),
    ("f_no_entry", "- entry route & admission marks", "features", dict(drop_num=ENTRY_NUM, drop_cat=ENTRY_CAT)),
    ("f_no_background", "- family / campus / displacement", "features", dict(drop_cat=BACKGROUND_CAT)),
    ("f_no_degree", "- degree", "features", dict(drop_cat=["tit_hash"])),
    ("f_no_lms", "- learning-platform activity (zeroed)", "features", dict(zero_seq=True)),
]
VKEYS = [v[0] for v in VARIANTS]
VLABEL = {v[0]: v[1] for v in VARIANTS}
VGROUP = {v[0]: v[2] for v in VARIANTS}
VSPEC = {v[0]: v[3] for v in VARIANTS}
COLORS = {"reference": "#d62728", "architecture": "#4c78a8", "training": "#54a24b", "features": "#f58518"}


def fold_data_abl(tr, va, sy, seq, spec):
    sp = S.HORIZON_SPEC[H]
    cat_cols = [c for c in S.STATIC_CAT if c not in spec.get("drop_cat", [])]
    num_cols = [c for c in S.STATIC_NUM + sp["extra"] if c not in spec.get("drop_num", [])]
    X = sy[cat_cols + num_cols]
    ct = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30, sparse_output=False), cat_cols),
        ("num", Pipeline([("i", SimpleImputer(strategy="median", add_indicator=True)), ("s", StandardScaler())]),
         num_cols)])
    xs_tr = ct.fit_transform(X.iloc[tr]).astype(np.float32)
    xs_va = ct.transform(X.iloc[va]).astype(np.float32)
    mu = seq[tr].mean(axis=(0, 1), keepdims=True)
    sd = seq[tr].std(axis=(0, 1), keepdims=True) + 1e-6
    sq_tr, sq_va = ((seq[tr] - mu) / sd).astype(np.float32), ((seq[va] - mu) / sd).astype(np.float32)
    if spec.get("zero_seq"):
        sq_tr, sq_va = np.zeros_like(sq_tr), np.zeros_like(sq_va)
    return xs_tr, xs_va, sq_tr, sq_va


def fit_predict_abl(spec, data, y_tr, seed):
    xs_tr, xs_va, xt_tr, xt_va = (torch.tensor(a) for a in data)
    base.set_seed(seed)
    model = AB.AblNet(xs_tr.shape[1], 2, xt_tr.shape[2], **spec)
    ytr = torch.tensor(y_tr, dtype=torch.long)
    counts = np.bincount(y_tr, minlength=2).astype(np.float64)
    cw = (len(y_tr) / (2 * np.maximum(counts, 1))) ** spec.get("cw_power", 1.0)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(seed)
    n = len(ytr)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss_fn(model(xs_tr[idx], xt_tr[idx]), ytr[idx]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), base.GRAD_CLIP)
            opt.step()
    model.eval()
    with torch.no_grad():
        return F.softmax(model(xs_va, xt_va), dim=1).numpy()[:, 1]


def run_job(vkey, fold, tr, va):
    torch.set_num_threads(1)
    path = os.path.join(CACHE, f"{H}_{vkey}_f{fold}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    sy, seq_all = S.build_student_year()
    seq, _ = S.horizon_sequences(H, seq_all, sy)
    y = sy["y"].values
    spec = VSPEC[vkey]
    data = fold_data_abl(tr, va, sy, seq, spec)
    model_spec = {k: v for k, v in spec.items() if k not in ("drop_num", "drop_cat", "zero_seq")}
    rows = []
    for s in SEEDS:
        p = fit_predict_abl(model_spec, data, y[tr], s + fold)
        top = p >= np.quantile(p, 0.90)
        rows.append(dict(variant=vkey, fold=fold + 1, seed=s, pr_auc=average_precision_score(y[va], p),
                         roc_auc=roc_auc_score(y[va], p), capture_top10=float(y[va][top].sum() / y[va].sum())))
    with open(path, "wb") as fh:
        pickle.dump(rows, fh)
    return rows


def analyse(runs):
    fold_df = runs.groupby(["variant", "fold"]).mean(numeric_only=True).reset_index()
    vec = lambda v, m: fold_df[fold_df.variant == v].sort_values("fold")[m].values
    rows = []
    full = vec("full", "pr_auc")
    for v in VKEYS:
        r = dict(variant=v, label=VLABEL[v], group=VGROUP[v])
        for m in ("pr_auc", "roc_auc", "capture_top10"):
            a = vec(v, m)
            r[f"{m}_mean"], r[f"{m}_std"] = a.mean(), a.std(ddof=1)
        if v != "full":
            diff = vec(v, "pr_auc") - full
            se = diff.std(ddof=1) / np.sqrt(len(diff))
            tc = stats.t.ppf(0.975, len(diff) - 1)
            r.update(d_pr_auc=diff.mean(), ci_lo=diff.mean() - tc * se, ci_hi=diff.mean() + tc * se,
                     p=base.paired_p(vec(v, "pr_auc"), full), d_roc_auc=(vec(v, "roc_auc") - vec("full", "roc_auc")).mean())
        rows.append(r)
    t = pd.DataFrame(rows)
    o = t.variant != "full"
    t.loc[o, "p_holm"] = base.holm_adjust(t.loc[o, "p"].values)
    return t, fold_df


def plot_delta(t):
    d = t[t.variant != "full"].sort_values("d_pr_auc")
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    y = np.arange(len(d))
    ax.barh(y, d.d_pr_auc, color=[COLORS[g] for g in d.group], alpha=0.85)
    ax.errorbar(d.d_pr_auc, y, xerr=[d.d_pr_auc - d.ci_lo, d.ci_hi - d.d_pr_auc], fmt="none", ecolor="k", capsize=3, lw=1)
    for yi, (_, r) in zip(y, d.iterrows()):
        star = "**" if r.p_holm < 0.05 else ("*" if r.p < 0.05 else "")
        ax.text(r.ci_hi if r.d_pr_auc >= 0 else r.ci_lo, yi, f" {star}", va="center",
                ha="left" if r.d_pr_auc >= 0 else "right")
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d.label)
    ax.set_xlabel("change in PR-AUC vs the full model (negative = the removed part helps)")
    ax.set_title(f"Ablation, {S.HORIZON_SPEC[H]['label']}: 95% CI over folds; * p<0.05, ** Holm-corrected p<0.05")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for k, c in COLORS.items() if k != "reference"]
    ax.legend(handles, [k for k in COLORS if k != "reference"], frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return base.save_fig(fig, f"fig_ablation_{H}_delta_pr_auc.png")


def main():
    t0 = time.time()
    print(f"[ablation] horizon={H} variants={len(VARIANTS)} seeds={SEEDS} epochs={EPOCHS} jobs={N_JOBS}")
    sy, _ = S.build_student_year()
    y = sy["y"].values
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=base.SEED).split(np.zeros(len(sy)), y,
                                                                                    sy["group"].values))
    res = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(run_job)(v, f, tr, va) for v in VKEYS
                                             for f, (tr, va) in enumerate(folds))
    runs = pd.DataFrame([r for rows in res for r in rows])
    runs.to_csv(os.path.join(OUT, f"ablation_{H}_all_runs.csv"), index=False)
    t, fold_df = analyse(runs)
    t.to_csv(os.path.join(OUT, f"ablation_{H}_numeric.csv"), index=False)
    pt = pd.DataFrame([{"Variant": r.label, "Type": r.group,
                        "PR-AUC": f"{r.pr_auc_mean:.3f} ± {r.pr_auc_std:.3f}",
                        "ROC-AUC": f"{r.roc_auc_mean:.3f} ± {r.roc_auc_std:.3f}",
                        "Δ PR-AUC [95% CI]": "-" if r.variant == "full" else f"{r.d_pr_auc:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]",
                        "p": "-" if r.variant == "full" else f"{r.p:.4f}",
                        "Holm p": "-" if r.variant == "full" else f"{r.p_holm:.4f}"} for _, r in t.iterrows()])
    pt.to_csv(os.path.join(OUT, f"ablation_{H}_table.csv"), index=False)
    print(pt.to_string(index=False))
    plot_delta(t)
    o = t[t.variant != "full"]
    hurt = list(o[(o.d_pr_auc < 0) & (o.p_holm < 0.05)].label)
    helped = list(o[(o.d_pr_auc > 0) & (o.p_holm < 0.05)].label)
    txt = (f"ABLATION {S.HORIZON_SPEC[H]['label']}: full-model PR-AUC {t.loc[t.variant == 'full', 'pr_auc_mean'].iloc[0]:.3f}\n"
           f"Removal significantly LOWERS PR-AUC (Holm p<0.05): {hurt or 'none'}\n"
           f"Removal significantly RAISES PR-AUC (Holm p<0.05): {helped or 'none'}\n"
           f"Largest drop: {o.sort_values('d_pr_auc').iloc[0].label} ({o.sort_values('d_pr_auc').iloc[0].d_pr_auc:+.4f})")
    print("\n" + txt)
    open(os.path.join(OUT, f"ablation_{H}_report.txt"), "w").write(txt)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
