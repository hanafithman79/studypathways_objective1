#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MONTH-SHUFFLE TEST: does the temporal branch use the ORDER of the monthly activity (recurring / temporal patterns)
or only the overall activity level?

Each deep model is trained once per fold and seed (default hyper-parameters, same folds as the main study). The
test students' monthly activity sequence is then altered and the model is re-scored WITHOUT retraining:
  original        the real sequence
  shuffle_months  the months of each student in random order (same totals, order destroyed)   [mean of R draws]
  reverse_months  the months in reverse order
  swap_students   each student gets another student's sequence (association with the student destroyed) [R draws]
  zero            sequence set to the training mean (no activity information)
If shuffle_months / reverse_months leave PR-AUC unchanged, the model does not rely on temporal order.
Fold value = mean over seeds and draws; delta = variant - original; paired t-test over the 5 folds.

Run:  HORIZON=H2 python spain_shuffle_test.py        (QUICK_TEST=1 for a smoke test)
Env:  DATA_DIR / SPAIN_7Z (see spain_dropout_experiments.py), HORIZON, N_SEEDS (2), N_DRAWS (5), N_JOBS
"""
import os
import sys

os.environ.setdefault("EXTRA_HYBRIDS", "1")
os.environ["OUT_DIR"] = os.environ.get("OUT_DIR", "spain_outputs_shuffle")
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
from sklearn.metrics import roc_auc_score, average_precision_score

import objective1_experiments as base
import objective1_track_head_and_tuning as tune
import spain_dropout_experiments as S

warnings.filterwarnings("ignore")
QUICK = base.QUICK_TEST
H = os.environ.get("HORIZON", "H2")
SEEDS = [base.SEED + i for i in range(int(os.environ.get("N_SEEDS", 1 if QUICK else 2)))]
R = int(os.environ.get("N_DRAWS", 2 if QUICK else 5))
N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 2))
OUT = base.OUT_DIR
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)
MODELS = ["lstm", "hyb1", "hyb2", "hyb3", "hyb4", "hyb5", "hyb6", "hyb7", "proposed"]
VARIANTS = ["original", "shuffle_months", "reverse_months", "swap_students", "zero"]


def train_model(key, cfg, data, y_tr, seed):
    base.set_seed(seed)
    model = tune.build_net(key, data["xs_tr"].shape[1], 2, cfg)
    xs, xt = torch.tensor(data["xs_tr"]), torch.tensor(data["seq_tr"])
    ytr = torch.tensor(y_tr, dtype=torch.long)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(tune.class_weights(y_tr, 2, cfg["cw_power"]), dtype=torch.float32))
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
            loss_fn(model(xs[idx], xt[idx]), ytr[idx]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), base.GRAD_CLIP)
            opt.step()
    return model.eval()


def score(model, xs_va, seq_var):
    with torch.no_grad():
        return F.softmax(model(torch.tensor(xs_va), torch.tensor(np.ascontiguousarray(seq_var))), dim=1).numpy()[:, 1]


def make_variants(seq_va, rng):
    n, T, _ = seq_va.shape
    v = {"original": [seq_va]}
    v["shuffle_months"] = [np.take_along_axis(seq_va, np.argsort(rng.random((n, T)), axis=1)[:, :, None], axis=1)
                           for _ in range(R)]
    v["reverse_months"] = [seq_va[:, ::-1, :]]
    v["swap_students"] = [seq_va[rng.permutation(n)] for _ in range(R)]
    v["zero"] = [np.zeros_like(seq_va)]
    return v


def run_job(key, fold, tr, va):
    torch.set_num_threads(1)
    path = os.path.join(CACHE, f"{H}_{key}_f{fold}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    sy, seq_all = S.build_student_year()
    seq, _ = S.horizon_sequences(H, seq_all, sy)
    y = sy["y"].values
    base.N_FEATS_STEP = seq.shape[2]
    d = S.fold_data(H, tr, va, sy, seq)
    cfg = tune.finalize_cfg(key, dict(tune.default_cfg(key), batch_size=512, epochs=15))
    acc = {v: {"pr_auc": [], "roc_auc": []} for v in VARIANTS}
    for s in SEEDS:
        model = train_model(key, cfg, d, y[tr], s + fold)
        rng = np.random.default_rng(1000 * s + fold)
        for name, arrs in make_variants(d["seq_va"], rng).items():
            ps = [average_precision_score(y[va], score(model, d["xs_va"], a)) for a in arrs]
            rs = [roc_auc_score(y[va], score(model, d["xs_va"], a)) for a in arrs]
            acc[name]["pr_auc"].append(np.mean(ps))
            acc[name]["roc_auc"].append(np.mean(rs))
    rows = [dict(horizon=H, model=key, fold=fold + 1, variant=v, pr_auc=np.mean(acc[v]["pr_auc"]),
                 roc_auc=np.mean(acc[v]["roc_auc"])) for v in VARIANTS]
    with open(path, "wb") as fh:
        pickle.dump(rows, fh)
    return rows


def analyse(df):
    out = []
    for key in MODELS:
        d = df[df.model == key]
        orig = d[d.variant == "original"].sort_values("fold")["pr_auc"].values
        for v in VARIANTS[1:]:
            x = d[d.variant == v].sort_values("fold")["pr_auc"].values
            diff = x - orig
            se = diff.std(ddof=1) / np.sqrt(len(diff))
            tc = stats.t.ppf(0.975, len(diff) - 1)
            out.append(dict(model=key, name=base.NAMES[key], variant=v, pr_auc_original=orig.mean(),
                            pr_auc_variant=x.mean(), delta=diff.mean(), ci_lo=diff.mean() - tc * se,
                            ci_hi=diff.mean() + tc * se, rel_change_pct=100 * diff.mean() / orig.mean(),
                            p=base.paired_p(x, orig)))
    t = pd.DataFrame(out)
    t["p_holm"] = base.holm_adjust(t["p"].values)
    return t


def plot(t):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1, 1.5]})
    p = t[t.model == "proposed"].set_index("variant").loc[VARIANTS[1:]]
    y = np.arange(len(p))
    axes[0].barh(y, p["delta"], color="#d62728", alpha=0.85)
    axes[0].errorbar(p["delta"], y, xerr=[p["delta"] - p["ci_lo"], p["ci_hi"] - p["delta"]], fmt="none", ecolor="k",
                     capsize=3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(VARIANTS[1:])
    axes[0].axvline(0, color="k", lw=1)
    axes[0].set_title("Proposed model: change in PR-AUC")
    axes[0].set_xlabel("PR-AUC (altered sequence) - PR-AUC (real sequence); 95% CI over folds")
    sm = t[t.variant == "shuffle_months"].set_index("model").loc[MODELS]
    zr = t[t.variant == "zero"].set_index("model").loc[MODELS]
    x = np.arange(len(MODELS))
    axes[1].bar(x - 0.2, sm["delta"], 0.4, label="months shuffled", color="#4c78a8")
    axes[1].bar(x + 0.2, zr["delta"], 0.4, label="sequence zeroed", color="#9ecae1")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([base.NAMES[k].split(" (")[0] if k != "proposed" else "Proposed" for k in MODELS],
                            rotation=40, ha="right", fontsize=8)
    axes[1].axhline(0, color="k", lw=1)
    axes[1].legend(frameon=False)
    axes[1].set_title("All deep models: change in PR-AUC")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Month-shuffle test, {S.HORIZON_SPEC[H]['label']}")
    fig.tight_layout()
    return base.save_fig(fig, f"fig_shuffle_{H}.png")


def main():
    t0 = time.time()
    print(f"[shuffle] horizon={H} models={len(MODELS)} seeds={SEEDS} draws={R} jobs={N_JOBS}")
    sy, _ = S.build_student_year()
    y = sy["y"].values
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=base.SEED).split(np.zeros(len(sy)), y,
                                                                                    sy["group"].values))
    res = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(run_job)(k, f, tr, va) for k in MODELS
                                             for f, (tr, va) in enumerate(folds))
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(os.path.join(OUT, f"shuffle_{H}_all_runs.csv"), index=False)
    t = analyse(df)
    t.to_csv(os.path.join(OUT, f"shuffle_{H}_table.csv"), index=False)
    p = t[t.model == "proposed"]
    print(p[["variant", "pr_auc_original", "pr_auc_variant", "delta", "ci_lo", "ci_hi", "rel_change_pct", "p", "p_holm"]]
          .round(4).to_string(index=False))
    plot(t)
    sm = t[t.variant == "shuffle_months"]
    uses_order = sm[(sm.delta < 0) & (sm.p_holm < 0.05)].name.tolist()
    txt = (f"MONTH-SHUFFLE TEST {S.HORIZON_SPEC[H]['label']}\n"
           f"Proposed: original PR-AUC {p.iloc[0].pr_auc_original:.4f}; months shuffled "
           f"{p[p.variant == 'shuffle_months'].delta.iloc[0]:+.4f} (p={p[p.variant == 'shuffle_months'].p.iloc[0]:.4f}); "
           f"reversed {p[p.variant == 'reverse_months'].delta.iloc[0]:+.4f}; swapped students "
           f"{p[p.variant == 'swap_students'].delta.iloc[0]:+.4f}; zeroed {p[p.variant == 'zero'].delta.iloc[0]:+.4f}\n"
           f"Models whose PR-AUC drops significantly (Holm p<0.05) when months are shuffled: {uses_order or 'none'}")
    print("\n" + txt)
    open(os.path.join(OUT, f"shuffle_{H}_report.txt"), "w").write(txt)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
