#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
OBJECTIVE 1 (part 2) - Dedicated 4-class MACRO_TRACK head + hyper-parameter tuning
===============================================================================
Requires `objective1_experiments.py` in the same folder (data preparation,
fold-wise preprocessing and network building blocks are imported from it).

What it does
  * Task "track"   : every model is trained DIRECTLY on the 4 MACRO_TRACK classes
                     (a separate 4-way output head, not derived from 21 programmes).
  * Task "program" : the 21-programme task, now with tuning.
  * Hyper-parameter search for ALL 10 models with the same budget:
        trial 0 = the default configuration, trials 1..N-1 = random search.
    The search is NESTED: inside the training part of every outer fold the data
    are split 80/20 (stratified); configurations are ranked on that inner
    hold-out only, the winner is refitted on the full outer-training part and
    evaluated once on the outer validation fold -> no selection leakage.
  * Two selection criteria are reported: inner Macro-F1 (primary metric) and
    inner Top-1 accuracy (track task) / Top-3 accuracy (programme task).
  * Three result "modes" per task: default, tuned_macro_f1, tuned_<2nd metric>.
  * Paired t-tests (fold-wise Macro-F1) of the Proposed model vs the 9 others,
    Holm correction, majority-class reference, all figures / tables / CSVs.

Run:   python objective1_track_head_and_tuning.py
Quick: QUICK_TEST=1 python objective1_track_head_and_tuning.py
Env:   N_TRIALS (default 8), N_JOBS (default = CPU count), RESUME=0 to ignore cache,
       USE_SABER_PRO_FEATURES (default 0 here = leakage-safe; set 1 to include
       G_SC/PERCENTILE/2ND_DECILE/QUARTILE as in the original specification).
Outputs: ./objective1_outputs_tuning/
===============================================================================
"""
import os
import sys

os.environ.setdefault("USE_SABER_PRO_FEATURES", "0")
os.environ["OUT_DIR"] = os.environ.get("OUT_DIR", "objective1_outputs_tuning")
os.environ.setdefault("RESULTS_CSV", "objective1_tuning_results.csv")
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    sys.path.insert(0, os.getcwd())

import json
import time
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             top_k_accuracy_score)

import objective1_experiments as base

warnings.filterwarnings("ignore")

QUICK = base.QUICK_TEST
SEED = base.SEED
N_TRIALS = int(os.environ.get("N_TRIALS", 3 if QUICK else 8))
N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 2))
RESUME = os.environ.get("RESUME", "1") == "1"
OUT = base.OUT_DIR
CACHE_DIR = os.path.join(OUT, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
DEVICE = torch.device("cpu")          # tiny networks: process-level CPU parallelism beats a GPU here

KEYS, NAMES, CATEGORY = base.KEYS, base.NAMES, base.CATEGORY
DEEP_KEYS, PROPOSED = base.DEEP_KEYS, base.PROPOSED
SEL2 = {"track": "accuracy", "program": "top3"}      # 2nd selection criterion per task
COST = {"hgb": 5, "rf": 4, "logreg": 1}               # rough cost, used only to order the job queue


# =============================================================================
# 1. SEARCH SPACES
# =============================================================================
def default_cfg(key):
    if key == "logreg":
        return dict(C=1.0, cw_power=1.0)
    if key == "rf":
        return dict(n_estimators=100, max_depth=None, min_samples_leaf=1, max_features="sqrt", cw_power=1.0)
    if key == "hgb":
        return dict(learning_rate=0.1, max_iter=100, max_leaf_nodes=31, l2_regularization=0.0,
                    min_samples_leaf=20, cw_power=1.0)
    return dict(epochs=15, lr=1e-3, weight_decay=1e-2, batch_size=128, dropout=0.2,
                static_h=128, rnn_h=64, cw_power=1.0)


def pick(rng, options):
    return options[int(rng.integers(len(options)))]


def sample_cfg(key, rng):
    cw = pick(rng, [0.0, 0.5, 1.0])               # 0 = unweighted, 1 = fully inverse-frequency
    if key == "logreg":
        return dict(C=float(10 ** rng.uniform(-3, 1)), cw_power=cw)
    if key == "rf":
        return dict(n_estimators=pick(rng, [100, 200, 300]), max_depth=pick(rng, [None, 10, 20, 30]),
                    min_samples_leaf=pick(rng, [1, 2, 5, 10]), max_features=pick(rng, ["sqrt", 0.3, 0.5]),
                    cw_power=cw)
    if key == "hgb":
        return dict(learning_rate=float(10 ** rng.uniform(np.log10(0.03), np.log10(0.3))),
                    max_iter=pick(rng, [100, 200, 300]), max_leaf_nodes=pick(rng, [15, 31, 63]),
                    l2_regularization=float(10 ** rng.uniform(-3, 1)),
                    min_samples_leaf=pick(rng, [10, 20, 50]), cw_power=cw)
    return dict(epochs=pick(rng, [15, 25, 40]), lr=float(10 ** rng.uniform(-3.5, -2.5)),
                weight_decay=float(10 ** rng.uniform(-4, -1)), batch_size=pick(rng, [64, 128, 256]),
                dropout=pick(rng, [0.1, 0.2, 0.3, 0.5]), static_h=pick(rng, [64, 128, 256]),
                rnn_h=pick(rng, [32, 64, 128]), cw_power=cw)


def finalize_cfg(key, cfg):
    cfg = dict(cfg)
    if QUICK:                                      # shrink everything for the smoke test
        for k, v in (("epochs", 2), ("n_estimators", 20), ("max_iter", 20)):
            if k in cfg:
                cfg[k] = min(cfg[k], v)
    return cfg


# =============================================================================
# 2. MODELS (architectures identical to objective1_experiments.py at default cfg)
# =============================================================================
def make_temporal(key, f, h):
    if key == "lstm":
        return base.LSTMEncoder(f, h), h
    if key == "hyb1":
        return base.CNNLSTMEncoder(f, 32, h), h
    if key == "hyb2":
        return base.BiLSTMEncoder(f, h // 2), 2 * (h // 2)
    if key == "hyb3":
        return base.GRUEncoder(f, h), h
    if key == "hyb4":
        return base.TransformerTemporalEncoder(f, h), h
    if key == "proposed":
        return base.CausalTCNLSTMEncoder(f, h), h
    raise KeyError(key)


class StaticNet(nn.Module):
    def __init__(self, d_static, K, hidden, p):
        super().__init__()
        self.enc = base.StaticEncoder(d_static, 64, hidden, p)
        self.head = nn.Linear(64, K)

    def forward(self, xs, xt):
        return self.head(self.enc(xs))


class TemporalNet(nn.Module):
    def __init__(self, enc, dim, K):
        super().__init__()
        self.enc, self.head = enc, nn.Linear(dim, K)

    def forward(self, xs, xt):
        return self.head(self.enc(xt))


class DualNet(nn.Module):
    def __init__(self, d_static, enc, dim, K, hidden, p):
        super().__init__()
        self.static_enc = base.StaticEncoder(d_static, 64, hidden, p)
        self.temporal_enc = enc
        self.head = base.FusionHead(64 + dim, K, p)

    def forward(self, xs, xt):
        return self.head(torch.cat([self.static_enc(xs), self.temporal_enc(xt)], dim=1))


def build_net(key, d_static, K, cfg):
    if key == "static_mlp":
        return StaticNet(d_static, K, cfg["static_h"], cfg["dropout"])
    enc, dim = make_temporal(key, base.N_FEATS_STEP, cfg["rnn_h"])
    if key == "lstm":
        return TemporalNet(enc, dim, K)
    return DualNet(d_static, enc, dim, K, cfg["static_h"], cfg["dropout"])


def class_weights(y, K, power):
    counts = np.bincount(y, minlength=K).astype(np.float64)
    w = (len(y) / (K * np.maximum(counts, 1))) ** power
    return np.where(counts > 0, w, 0.0)


def fit_predict_deep(key, cfg, data, y_tr, K, seed):
    base.set_seed(seed)
    model = build_net(key, data["xs_tr"].shape[1], K, cfg).to(DEVICE)
    xs_tr, xt_tr = torch.tensor(data["xs_tr"]), torch.tensor(data["seq_tr"])
    ytr = torch.tensor(y_tr, dtype=torch.long)
    xs_va, xt_va = torch.tensor(data["xs_va"]), torch.tensor(data["seq_va"])
    cw = torch.tensor(class_weights(y_tr, K, cfg["cw_power"]), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    gen = torch.Generator().manual_seed(seed)
    n, bs = len(ytr), cfg["batch_size"]
    for _ in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss_fn(model(xs_tr[idx], xt_tr[idx]), ytr[idx]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), base.GRAD_CLIP)
            opt.step()
    model.eval()
    with torch.no_grad():
        return F.softmax(model(xs_va, xt_va), dim=1).numpy().astype(np.float32)


def fit_predict_sklearn(key, cfg, data, y_tr, K, seed):
    sw = class_weights(y_tr, K, cfg["cw_power"])[y_tr]
    if key == "logreg":
        m = LogisticRegression(C=cfg["C"], max_iter=1000)
    elif key == "rf":
        m = RandomForestClassifier(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                                   min_samples_leaf=cfg["min_samples_leaf"], max_features=cfg["max_features"],
                                   n_jobs=1, random_state=seed)
    else:
        m = HistGradientBoostingClassifier(
            learning_rate=cfg["learning_rate"], max_iter=cfg["max_iter"], max_leaf_nodes=cfg["max_leaf_nodes"],
            l2_regularization=cfg["l2_regularization"], min_samples_leaf=cfg["min_samples_leaf"],
            early_stopping=False, random_state=seed)
    m.fit(data["flat_tr"], y_tr, sample_weight=sw)
    return base.align_proba(m.predict_proba(data["flat_va"]), m.classes_, K)


def fit_predict(key, cfg, data, y_tr, K, seed):
    if key in DEEP_KEYS:
        return fit_predict_deep(key, cfg, data, y_tr, K, seed)
    return fit_predict_sklearn(key, cfg, data, y_tr, K, seed)


# =============================================================================
# 3. METRICS
# =============================================================================
def all_metrics(y_true, proba, K, prog_track_idx=None):
    pred = proba.argmax(1)
    m = {"accuracy": 100 * accuracy_score(y_true, pred),
         "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
         "bal_acc": 100 * balanced_accuracy_score(y_true, pred)}
    if K > 4:
        m["top3"] = 100 * top_k_accuracy_score(y_true, proba, k=3, labels=np.arange(K))
    if prog_track_idx is not None:       # programme model -> derived track accuracy
        tp = (proba @ np.eye(4)[prog_track_idx]).argmax(1)
        m["track_acc_derived"] = 100 * accuracy_score(prog_track_idx[y_true], tp)
    return m


# =============================================================================
# 4. ONE (task, model, outer-fold) JOB  - nested search + final evaluation
# =============================================================================
def run_task(task, key, fold, tr, va, df, y, y_track, K, prog_track_idx):
    torch.set_num_threads(1)
    path = os.path.join(CACHE_DIR, f"{task}_{key}_f{fold}.pkl")
    if RESUME and os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    t0 = time.time()
    rng = np.random.default_rng(SEED * 10007 + fold * 101 + KEYS.index(key) + (0 if task == "program" else 7919))
    cfgs = [default_cfg(key)] + [sample_cfg(key, rng) for _ in range(N_TRIALS - 1)]
    cfgs = [finalize_cfg(key, c) for c in cfgs]
    pti = prog_track_idx if task == "program" else None

    # ---- inner hold-out search (outer-validation data never touched) --------
    in_tr, in_va = train_test_split(tr, test_size=0.2, stratify=y_track[tr], random_state=SEED + fold)
    d_in = base.preprocess_fold(df.iloc[in_tr], df.iloc[in_va])
    trials = []
    for t, cfg in enumerate(cfgs):
        proba = fit_predict(key, cfg, d_in, y[in_tr], K, SEED + fold)
        m = all_metrics(y[in_va], proba, K)
        trials.append(dict(task=task, model=key, fold=fold + 1, trial=t, cfg=cfg,
                           **{f"inner_{k}": v for k, v in m.items()}))

    # ---- choose configs, refit on the full outer-training part, evaluate once ----
    d_out = base.preprocess_fold(df.iloc[tr], df.iloc[va])
    chosen = {"default": 0}
    for sel in ("macro_f1", SEL2[task]):
        best = max(trials, key=lambda t: (t[f"inner_{sel}"], -t["trial"]))
        chosen[f"tuned_{sel}"] = best["trial"]
    fitted = {}
    modes = {}
    for mode, t_idx in chosen.items():
        if t_idx not in fitted:
            fitted[t_idx] = fit_predict(key, cfgs[t_idx], d_out, y[tr], K, SEED + fold)
        modes[mode] = dict(trial=t_idx, cfg=cfgs[t_idx], proba=fitted[t_idx],
                           metrics=all_metrics(y[va], fitted[t_idx], K, pti))
    out = dict(task=task, model=key, fold=fold, va=va, trials=trials, modes=modes, seconds=time.time() - t0)
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    return out


# =============================================================================
# 5. AGGREGATION / STATISTICS
# =============================================================================
def aggregate(results, task, mode, y, tr_folds, K):
    rows = []
    for r in results:
        if r["task"] == task:
            rows.append(dict(model=r["model"], fold=r["fold"] + 1, **r["modes"][mode]["metrics"]))
    fold_df = pd.DataFrame(rows)
    # majority-class reference (fit on each training fold)
    ref = []
    for f, (tr, va) in enumerate(tr_folds):
        maj = np.bincount(y[tr], minlength=K).argmax()
        pred = np.full(len(va), maj)
        ref.append(dict(model="majority", fold=f + 1, accuracy=100 * accuracy_score(y[va], pred),
                        macro_f1=f1_score(y[va], pred, average="macro", zero_division=0),
                        bal_acc=100 * balanced_accuracy_score(y[va], pred)))
    fold_df = pd.concat([fold_df, pd.DataFrame(ref)], ignore_index=True)
    cols = [c for c in fold_df.columns if c not in ("model", "fold")]
    means = fold_df.groupby("model")[cols].mean()
    stds = fold_df.groupby("model")[cols].std(ddof=1)

    def vec(k, m):
        return fold_df[fold_df.model == k].sort_values("fold")[m].values

    pv = pd.DataFrame(index=KEYS, columns=["p_macro_f1", "p_accuracy", "p_macro_f1_holm"], dtype=float)
    for k in KEYS:
        if k != PROPOSED:
            pv.loc[k, "p_macro_f1"] = base.paired_p(vec(PROPOSED, "macro_f1"), vec(k, "macro_f1"))
            pv.loc[k, "p_accuracy"] = base.paired_p(vec(PROPOSED, "accuracy"), vec(k, "accuracy"))
    others = [k for k in KEYS if k != PROPOSED]
    pv.loc[others, "p_macro_f1_holm"] = base.holm_adjust(pv.loc[others, "p_macro_f1"].values)
    return fold_df, means, stds, pv


def make_table(means, stds, pv, task):
    metric_cols = [("accuracy", "Top-1 accuracy (%)", 2), ("macro_f1", "Macro-F1", 3),
                   ("bal_acc", "Balanced acc. (%)", 2)]
    if task == "program":
        metric_cols.insert(1, ("top3", "Top-3 accuracy (%)", 2))
    rows = []
    for k in KEYS + ["majority"]:
        row = {"Model": NAMES.get(k, "Majority class (reference)"),
               "Category": CATEGORY.get(k, "reference")}
        for c, label, nd in metric_cols:
            if c in means.columns and not np.isnan(means.loc[k, c]):
                row[label] = f"{means.loc[k, c]:.{nd}f} ± {stds.loc[k, c]:.{nd}f}"
            else:
                row[label] = "-"
        if k in (PROPOSED, "majority"):
            row["p vs Proposed (Macro-F1)"], row["Holm-adj. p"] = "-", "-"
        else:
            row["p vs Proposed (Macro-F1)"] = f"{pv.loc[k, 'p_macro_f1']:.4f}"
            row["Holm-adj. p"] = f"{pv.loc[k, 'p_macro_f1_holm']:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 6. FIGURES
# =============================================================================
def fig_default_vs_tuned(summary, task, modes):
    metrics = [("accuracy", "Top-1 accuracy (%)"), ("macro_f1", "Macro-F1")]
    if task == "program":
        metrics.insert(1, ("top3", "Top-3 accuracy (%)"))
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.2 * len(metrics), 6))
    w = 0.8 / len(modes)
    cols = ["#9ecae1", "#3182bd", "#e6550d"]
    for ax, (m, title) in zip(np.atleast_1d(axes), metrics):
        for i, mode in enumerate(modes):
            mu = [summary[(task, mode)][0].loc[k, m] for k in KEYS]
            sd = [summary[(task, mode)][1].loc[k, m] for k in KEYS]
            ax.barh(np.arange(len(KEYS)) + (i - (len(modes) - 1) / 2) * w, mu, w, xerr=sd, capsize=2,
                    color=cols[i], label=mode)
        ref = summary[(task, modes[0])][0].loc["majority", m] if m in ("accuracy", "macro_f1") else None
        if ref is not None:
            ax.axvline(ref, color="k", ls=":", lw=1)
            ax.text(ref, -0.75, "majority", fontsize=7, ha="center")
        if m == "accuracy" and task == "track":
            ax.axvline(80, color="r", ls="--", lw=1)
            ax.text(80, -0.75, "80% target", fontsize=7, ha="center", color="r")
        if m == "top3":
            ax.axvline(85, color="r", ls="--", lw=1)
            ax.text(85, -0.75, "85% target", fontsize=7, ha="center", color="r")
        ax.set_yticks(range(len(KEYS)))
        ax.set_yticklabels([NAMES[k] for k in KEYS])
        ax.invert_yaxis()
        ax.set_title(title)
    np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle(f"{task.upper()} task - default vs tuned hyper-parameters (outer 5-fold CV, mean ± std)")
    fig.tight_layout()
    return base.save_fig(fig, f"fig_t01_{task}_default_vs_tuned.png")


def fig_heatmap(means, task, mode):
    cols = [c for c in ["accuracy", "top3", "macro_f1", "bal_acc", "track_acc_derived"] if c in means.columns]
    labels = {"accuracy": "Top-1 %", "top3": "Top-3 %", "macro_f1": "Macro-F1", "bal_acc": "Bal-Acc %",
              "track_acc_derived": "Track acc %\n(derived)"}
    data = means.reindex(KEYS)[cols].values.astype(float)
    norm = (data - data.min(0)) / (data.max(0) - data.min(0) + 1e-12)
    fig, ax = plt.subplots(figsize=(1.6 * len(cols) + 4, 5.5))
    ax.imshow(norm, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([labels[c] for c in cols])
    ax.set_yticks(range(len(KEYS)))
    ax.set_yticklabels([NAMES[k] for k in KEYS])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.3f}" if cols[j] == "macro_f1" else f"{data[i, j]:.1f}",
                    ha="center", va="center", fontsize=8, color="white" if norm[i, j] > 0.6 else "black")
    ax.set_title(f"{task} / {mode} - mean performance (colour scaled per column)")
    fig.tight_layout()
    return base.save_fig(fig, f"fig_t02_{task}_{mode}_heatmap.png")


def fig_search(all_trials, task):
    t = all_trials[all_trials.task == task]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for i, k in enumerate(KEYS):
        d = t[t.model == k]
        ax.scatter(np.full(len(d), i) + np.random.uniform(-0.2, 0.2, len(d)), d["inner_macro_f1"],
                   s=14, alpha=0.5, color=base.COLORS[CATEGORY[k]])
        dd = d[d.trial == 0]
        ax.scatter(np.full(len(dd), i), dd["inner_macro_f1"], marker="D", s=40, color="k",
                   label="default config" if i == 0 else None, zorder=3)
        best = d.loc[d.groupby("fold")["inner_macro_f1"].idxmax()]
        ax.scatter(np.full(len(best), i), best["inner_macro_f1"], marker="*", s=90, color="gold",
                   edgecolor="k", label="best per fold" if i == 0 else None, zorder=4)
    ax.set_xticks(range(len(KEYS)))
    ax.set_xticklabels([NAMES[k] for k in KEYS], rotation=35, ha="right")
    ax.set_ylabel("inner hold-out Macro-F1")
    ax.set_title(f"{task.upper()} - random-search trials (each dot = one configuration in one outer fold)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return base.save_fig(fig, f"fig_t03_{task}_search_trials.png")


def fig_track_grid(oof, y_track, mode):
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    short_t = ["Ind&Mgmt", "Civil&Infra", "Mech/Elec/Tech", "Chem&Proc"]
    from sklearn.metrics import confusion_matrix
    for ax, k in zip(axes.ravel(), KEYS):
        p = oof[("track", mode, k)].argmax(1)
        cm = confusion_matrix(y_track, p, labels=np.arange(4))
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.55 else "black")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(short_t, rotation=40, ha="right", fontsize=7)
        ax.set_yticklabels(short_t, fontsize=7)
        ax.set_title(f"{NAMES[k]}\nacc {100 * np.mean(p == y_track):.1f}%", fontsize=8)
    fig.suptitle(f"4-class macro-track HEAD - pooled OOF confusion matrices ({mode}, row-normalised)")
    fig.tight_layout()
    return base.save_fig(fig, f"fig_t04_track_confusion_grid_{mode}.png")


def fig_pvalues(pv, task, mode):
    others = [k for k in KEYS if k != PROPOSED]
    data = pv.loc[others, ["p_macro_f1", "p_accuracy"]].values.astype(float)
    fig, ax = plt.subplots(figsize=(6, 5.2))
    base.draw_heatmap(ax, -np.log10(np.clip(data, 1e-6, 1)), ["Macro-F1", "Top-1 acc"],
                      [NAMES[k] for k in others], cmap="magma", annotate=False, rotation=0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.3f}" + ("*" if data[i, j] < 0.05 else ""), ha="center", va="center",
                    fontsize=8, color="white" if data[i, j] > 0.2 else "black")
    ax.set_title(f"{task}/{mode}: paired t-test p, Proposed vs model\n* p<0.05 (uncorrected)")
    fig.tight_layout()
    return base.save_fig(fig, f"fig_t05_{task}_{mode}_pvalues.png")


def fig_boxplot(fold_df, task, mode):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    data = [fold_df[fold_df.model == k].sort_values("fold")["macro_f1"].values for k in KEYS]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    for patch, k in zip(bp["boxes"], KEYS):
        patch.set_facecolor(base.COLORS[CATEGORY[k]])
        patch.set_alpha(0.6)
    for i, d in enumerate(data):
        ax.scatter(np.full(len(d), i + 1), d, color="k", s=10, zorder=3)
    ax.set_xticks(range(1, len(KEYS) + 1))
    ax.set_xticklabels([NAMES[k] for k in KEYS], rotation=40, ha="right")
    ax.set_title(f"{task}/{mode}: Macro-F1 per outer fold")
    fig.tight_layout()
    return base.save_fig(fig, f"fig_t06_{task}_{mode}_fold_boxplot.png")


# =============================================================================
# 7. MAIN
# =============================================================================
def main():
    t0 = time.time()
    print(f"[tuning] trials/model={N_TRIALS} jobs={N_JOBS} quick={QUICK} resume={RESUME} "
          f"saber_pro_features={base.USE_SABER_PRO_FEATURES}")
    df, y_prog, programs, prog_track_idx, M = base.prepare_data()
    y_track = prog_track_idx[y_prog]
    K = len(programs)
    N = len(df)
    skf = StratifiedKFold(n_splits=base.N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(N), y_prog))
    print(f"[data] N={N}  track majority floor {100 * np.bincount(y_track).max() / N:.1f}%  "
          f"programme majority floor {100 * np.bincount(y_prog).max() / N:.1f}%")

    jobs = []
    for task, yy, KK in (("program", y_prog, K), ("track", y_track, 4)):
        for fold, (tr, va) in enumerate(folds):
            for key in KEYS:
                jobs.append((COST.get(key, 2) * (2 if task == "program" else 1), task, key, fold, tr, va, yy, KK))
    jobs.sort(key=lambda j: -j[0])
    print(f"[jobs] {len(jobs)} (task x model x fold) jobs, each = {N_TRIALS} inner trials + final fits")
    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(run_task)(task, key, fold, tr, va, df, yy, y_track, KK, prog_track_idx)
        for _, task, key, fold, tr, va, yy, KK in jobs)

    # ---------------------------------------------------------------- per-trial / config logs
    trial_rows = [dict(t, cfg=json.dumps(t["cfg"])) for r in results for t in r["trials"]]
    all_trials = pd.DataFrame(trial_rows)
    all_trials.to_csv(os.path.join(OUT, "all_search_trials.csv"), index=False)
    sel_rows = []
    for r in results:
        for mode, d in r["modes"].items():
            sel_rows.append(dict(task=r["task"], model=r["model"], fold=r["fold"] + 1, mode=mode,
                                 trial=d["trial"], cfg=json.dumps(d["cfg"]), **d["metrics"]))
    pd.DataFrame(sel_rows).to_csv(os.path.join(OUT, "selected_hyperparameters_and_fold_metrics.csv"), index=False)

    # ---------------------------------------------------------------- OOF pooling
    oof = {}
    for r in results:
        for mode, d in r["modes"].items():
            kk = (r["task"], mode, r["model"])
            if kk not in oof:
                oof[kk] = np.zeros((N, K if r["task"] == "program" else 4), dtype=np.float32)
            oof[kk][r["va"]] = d["proba"]

    # ---------------------------------------------------------------- tables / figures
    summary, report = {}, ["OBJECTIVE 1 PART 2 - TRACK HEAD & TUNING", "=" * 70,
                           f"N={N} | trials/model={N_TRIALS} | nested inner hold-out 80/20 | "
                           f"saber_pro_features={base.USE_SABER_PRO_FEATURES}",
                           f"majority floors: track {100 * np.bincount(y_track).max() / N:.1f}%, "
                           f"programme {100 * np.bincount(y_prog).max() / N:.1f}%", ""]
    big = []
    for task, yy, KK in (("track", y_track, 4), ("program", y_prog, K)):
        modes = ["default", "tuned_macro_f1", f"tuned_{SEL2[task]}"]
        tr_folds = [(tr, va) for tr, va in folds]
        for mode in modes:
            fold_df, means, stds, pv = aggregate(results, task, mode, yy, tr_folds, KK)
            summary[(task, mode)] = (means, stds, pv, fold_df)
            table = make_table(means, stds, pv, task)
            table.to_csv(os.path.join(OUT, f"{task}_{mode}_results.csv"), index=False)
            fold_df.to_csv(os.path.join(OUT, f"{task}_{mode}_fold_metrics.csv"), index=False)
            pv.to_csv(os.path.join(OUT, f"{task}_{mode}_pvalues_vs_proposed.csv"))
            print("\n" + "=" * 120 + f"\n{task.upper()} / {mode}\n" + "=" * 120)
            print(table.to_string(index=False))
            for k in KEYS + ["majority"]:
                big.append(dict(task=task, mode=mode, model=k,
                                **{f"{c}_mean": means.loc[k, c] for c in means.columns},
                                **{f"{c}_std": stds.loc[k, c] for c in stds.columns}))
            fig_heatmap(means, task, mode)
            fig_pvalues(pv, task, mode)
            fig_boxplot(fold_df, task, mode)
            if task == "track":
                fig_track_grid(oof, yy, mode)
        fig_default_vs_tuned({(task, m): summary[(task, m)][:2] for m in modes}, task, modes)
        fig_search(all_trials, task)

    pd.DataFrame(big).to_csv(os.path.join(OUT, "objective1_tuning_summary_all.csv"), index=False)
    try:
        t = pd.read_csv(os.path.join(OUT, "track_tuned_macro_f1_results.csv"))
        t.to_csv(base.RESULTS_CSV, index=False)
    except Exception:
        pass

    # headline confusion matrices for the proposed model (track head)
    for mode in ("tuned_macro_f1", "tuned_accuracy"):
        p = oof[("track", mode, PROPOSED)].argmax(1)
        base.plot_confusion(y_track, p, base.TRACK_NAMES, base.TRACK_NAMES,
                            f"Proposed - 4-class track head ({mode}), pooled OOF",
                            f"fig_t07_proposed_track_confusion_{mode}.png", figsize=(7.5, 6.5), fontsize=7)
    np.savez_compressed(os.path.join(OUT, "oof_probabilities_tuning.npz"),
                        **{"|".join(k): v for k, v in oof.items()}, y_track=y_track, y_program=y_prog)

    # ---------------------------------------------------------------- narrative report
    for task in ("track", "program"):
        report.append(f"## {task.upper()} TASK")
        for mode in ["default", "tuned_macro_f1", f"tuned_{SEL2[task]}"]:
            means, stds, pv, _ = summary[(task, mode)]
            bf, ba = means.loc[KEYS, "macro_f1"].idxmax(), means.loc[KEYS, "accuracy"].idxmax()
            report.append(f"[{mode}] best Macro-F1: {NAMES[bf]} {means.loc[bf, 'macro_f1']:.3f} | "
                          f"best accuracy: {NAMES[ba]} {means.loc[ba, 'accuracy']:.1f}% | "
                          f"proposed: F1 {means.loc[PROPOSED, 'macro_f1']:.3f}, acc {means.loc[PROPOSED, 'accuracy']:.1f}%"
                          + (f", top3 {means.loc[PROPOSED, 'top3']:.1f}%" if task == "program" else "")
                          + f" | majority acc {means.loc['majority', 'accuracy']:.1f}%")
            sig = [k for k in KEYS if k != PROPOSED and pv.loc[k, "p_macro_f1"] < 0.05]
            worse = [NAMES[k] for k in sig if means.loc[PROPOSED, "macro_f1"] < means.loc[k, "macro_f1"]]
            better = [NAMES[k] for k in sig if means.loc[PROPOSED, "macro_f1"] > means.loc[k, "macro_f1"]]
            report.append(f"    significantly (p<0.05, uncorrected) better than: {better or 'none'}; "
                          f"significantly worse than: {worse or 'none'}")
        d0, d1 = summary[(task, "default")][0], summary[(task, "tuned_macro_f1")][0]
        report.append(f"    tuning gain for Proposed (Macro-F1): {d0.loc[PROPOSED, 'macro_f1']:.3f} -> "
                      f"{d1.loc[PROPOSED, 'macro_f1']:.3f}")
        if task == "track":
            best_acc = max(summary[("track", m)][0].loc[KEYS, "accuracy"].max()
                           for m in ["default", "tuned_macro_f1", "tuned_accuracy"])
            report.append(f"    80% track-accuracy target: best observed {best_acc:.1f}% -> "
                          f"{'MET' if best_acc > 80 else 'NOT MET'}")
        else:
            best_t3 = max(summary[("program", m)][0].loc[KEYS, "top3"].max()
                          for m in ["default", "tuned_macro_f1", "tuned_top3"])
            report.append(f"    85% Top-3 target: best observed {best_t3:.1f}% -> "
                          f"{'MET' if best_t3 > 85 else 'NOT MET'}")
        report.append("")
    report.append("Selection used ONLY the inner 20% hold-out of each outer-training set; outer folds were "
                  "evaluated once. Paired t-tests use 5 folds (4 d.o.f.) and have low power.")
    txt = "\n".join(report)
    print("\n" + txt)
    with open(os.path.join(OUT, "tuning_report.txt"), "w") as fh:
        fh.write(txt)
    print(f"\nAll outputs in ./{OUT}/  (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
