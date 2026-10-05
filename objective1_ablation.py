#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
OBJECTIVE 1 (part 3) - ABLATION STUDY of the Proposed Dual-Branch Hybrid
===============================================================================
Requires `objective1_experiments.py` in the same folder.

Every variant is trained with the SAME hyper-parameters, the SAME 5 stratified
folds and 3 random seeds per fold (fold value = mean over seeds), on both tasks:
  * "program": 21 granular programmes   * "track": 4 macro-tracks (own 4-way head)

Variants (delta = variant - full model; negative delta => the component helps)
  Architecture : - static branch | - temporal branch | - causal TCN (MLP+LSTM) |
                 - LSTM (MLP+TCN) | - TCN residual | - causality (symmetric conv) |
                 - BatchNorm | - Dropout | - fusion MLP (linear head)
  Training     : - class-weighted loss
  Feature group: - family (EDU/OCC) | - household/economic | - school & gender |
                 - engineered STEM/HUM features | - English score
Statistics: paired t-test over the 5 folds, 95% CI of the fold-wise difference,
Holm correction across all comparisons.

Run:   python objective1_ablation.py          (QUICK_TEST=1 for a smoke test)
Env:   N_JOBS, USE_SABER_PRO_FEATURES (default 0 = leakage-safe), N_SEEDS (3)
Output: ./objective1_outputs_ablation/
===============================================================================
"""
import os
import sys

os.environ.setdefault("USE_SABER_PRO_FEATURES", "0")
os.environ["OUT_DIR"] = os.environ.get("OUT_DIR", "objective1_outputs_ablation")
os.environ.setdefault("RESULTS_CSV", "objective1_ablation_results.csv")
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

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, top_k_accuracy_score

import objective1_experiments as base

warnings.filterwarnings("ignore")

QUICK = base.QUICK_TEST
SEED = base.SEED
SEEDS = [SEED + i for i in range(int(os.environ.get("N_SEEDS", 1 if QUICK else 3)))]
N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 2))
RESUME = os.environ.get("RESUME", "1") == "1"
OUT = base.OUT_DIR
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)

TRAIN = dict(epochs=2 if QUICK else 15, lr=base.LR, weight_decay=base.WEIGHT_DECAY,
             batch_size=base.BATCH_SIZE, clip=base.GRAD_CLIP)

FAMILY = ["EDU_FATHER", "EDU_MOTHER", "OCC_FATHER", "OCC_MOTHER"]
ECONOMIC = ["STRATUM", "SISBEN", "REVENUE", "PEOPLE_HOUSE", "INTERNET", "COMPUTER", "CAR"]
SCHOOL_DEMO = ["GENDER", "SCHOOL_TYPE", "SCHOOL_NAT"]
ENGINEERED = ["STEM_AVG", "HUM_AVG", "STEM_HUM_RATIO"]

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
    ("f_no_family", "- family features (EDU/OCC)", "features", dict(drop_static=FAMILY)),
    ("f_no_economic", "- household/economic features", "features", dict(drop_static=ECONOMIC)),
    ("f_no_school", "- school & gender features", "features", dict(drop_static=SCHOOL_DEMO)),
    ("f_no_engineered", "- engineered STEM/HUM features", "features", dict(zero_temporal=ENGINEERED)),
    ("f_no_english", "- English score (ENG_S11)", "features", dict(zero_temporal=["ENG_S11"])),
]
VKEYS = [v[0] for v in VARIANTS]
VLABEL = {v[0]: v[1] for v in VARIANTS}
VGROUP = {v[0]: v[2] for v in VARIANTS}
VSPEC = {v[0]: v[3] for v in VARIANTS}
GROUP_COLOR = {"reference": "#d62728", "architecture": "#4c78a8", "training": "#54a24b", "features": "#f58518"}


# =============================================================================
# model with switchable components
# =============================================================================
class TemporalEnc(nn.Module):
    def __init__(self, f, mode, hidden=64):
        super().__init__()
        self.mode = mode
        if mode == "symconv_lstm":
            self.conv = nn.Conv1d(f, f, 3, padding=1)              # centred: sees the "future" step
        elif mode != "lstm":
            self.conv = base.CausalConv1d(f, 2, 1)                  # left-padded: causal
        if mode == "tcn_only":
            self.proj = nn.Linear(f, hidden)
        else:
            self.rnn = nn.LSTM(f, hidden, batch_first=True)

    def forward(self, x):                                           # x: (B, T, F)
        if self.mode != "lstm":
            c = x.transpose(1, 2)
            y = F.relu(self.conv(c))
            c = y if self.mode == "tcn_lstm_nores" else c + y       # residual unless ablated
            if self.mode == "tcn_only":
                return F.relu(self.proj(c[:, :, -1]))
            x = c.transpose(1, 2)
        _, (h, _) = self.rnn(x)
        return h[-1]


class AblNet(nn.Module):
    def __init__(self, d_static, K, f, use_static=True, use_temporal=True, tmode="tcn_lstm",
                 batchnorm=True, dropout=0.2, fusion="mlp", **_):
        super().__init__()
        self.use_static, self.use_temporal = use_static, use_temporal
        dim = 0
        if use_static:
            layers = [nn.Linear(d_static, 128)] + ([nn.BatchNorm1d(128)] if batchnorm else []) + \
                     [nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 64), nn.ReLU()]
            self.static = nn.Sequential(*layers)
            dim += 64
        if use_temporal:
            self.temporal = TemporalEnc(f, tmode)
            dim += 64
        if fusion == "mlp":
            self.head = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, K))
        else:
            self.head = nn.Linear(dim, K)

    def forward(self, xs, xt):
        z = []
        if self.use_static:
            z.append(self.static(xs))
        if self.use_temporal:
            z.append(self.temporal(xt))
        return self.head(torch.cat(z, dim=1))


# =============================================================================
# preprocessing (fold-wise) with feature-group ablation
# =============================================================================
def preprocess(df_tr, df_va, drop_static=(), zero_temporal=()):
    cols = [c for c in base.STATIC_COLS if c not in drop_static]
    ct = base.ColumnTransformer([("cat", base.make_onehot(), cols)])
    xs_tr = ct.fit_transform(df_tr[cols]).astype(np.float32)
    xs_va = ct.transform(df_va[cols]).astype(np.float32)
    sc = StandardScaler()
    xt_tr = sc.fit_transform(df_tr[base.TEMPORAL_COLS]).astype(np.float32)
    xt_va = sc.transform(df_va[base.TEMPORAL_COLS]).astype(np.float32)
    for c in zero_temporal:                       # neutralise = constant (train-mean) input
        j = base.TEMPORAL_COLS.index(c)
        xt_tr[:, j] = 0.0
        xt_va[:, j] = 0.0
    shp = (-1, base.N_STEPS, base.N_FEATS_STEP)
    return xs_tr, xs_va, xt_tr.reshape(shp), xt_va.reshape(shp)


def fit_predict(spec, data, y_tr, K, seed):
    xs_tr, xs_va, xt_tr, xt_va = (torch.tensor(a) for a in data)
    base.set_seed(seed)
    model = AblNet(xs_tr.shape[1], K, base.N_FEATS_STEP, **spec)
    ytr = torch.tensor(y_tr, dtype=torch.long)
    counts = np.bincount(y_tr, minlength=K).astype(np.float64)
    power = spec.get("cw_power", 1.0)
    cw = np.where(counts > 0, (len(y_tr) / (K * np.maximum(counts, 1))) ** power, 0.0)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])
    gen = torch.Generator().manual_seed(seed)
    n, bs = len(ytr), TRAIN["batch_size"]
    for _ in range(TRAIN["epochs"]):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss_fn(model(xs_tr[idx], xt_tr[idx]), ytr[idx]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), TRAIN["clip"])
            opt.step()
    model.eval()
    with torch.no_grad():
        return F.softmax(model(xs_va, xt_va), dim=1).numpy()


def metrics(y, proba, K):
    pred = proba.argmax(1)
    m = {"accuracy": 100 * accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
         "bal_acc": 100 * balanced_accuracy_score(y, pred)}
    if K > 4:
        m["top3"] = 100 * top_k_accuracy_score(y, proba, k=3, labels=np.arange(K))
    return m


def run_job(task, vkey, fold, tr, va, df, y, K):
    torch.set_num_threads(1)
    path = os.path.join(CACHE, f"{task}_{vkey}_f{fold}.pkl")
    if RESUME and os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    spec = VSPEC[vkey]
    data = preprocess(df.iloc[tr], df.iloc[va], spec.get("drop_static", ()), spec.get("zero_temporal", ()))
    rows = []
    for s in SEEDS:
        m = metrics(y[va], fit_predict(spec, data, y[tr], K, s + fold), K)
        rows.append(dict(task=task, variant=vkey, fold=fold + 1, seed=s, **m))
    with open(path, "wb") as fh:
        pickle.dump(rows, fh)
    return rows


# =============================================================================
# statistics & plots
# =============================================================================
def analyse(runs, task):
    d = runs[runs.task == task]
    fold_df = d.groupby(["variant", "fold"]).mean(numeric_only=True).reset_index()
    metric_cols = [c for c in ["accuracy", "macro_f1", "bal_acc", "top3"] if c in fold_df.columns]

    def vec(v, m):
        return fold_df[fold_df.variant == v].sort_values("fold")[m].values

    rows = []
    full_f1, full_acc = vec("full", "macro_f1"), vec("full", "accuracy")
    for v in VKEYS:
        r = dict(variant=v, label=VLABEL[v], group=VGROUP[v])
        for m in metric_cols:
            a = vec(v, m)
            r[f"{m}_mean"], r[f"{m}_std"] = a.mean(), a.std(ddof=1)
        if v != "full":
            diff = vec(v, "macro_f1") - full_f1
            se = diff.std(ddof=1) / np.sqrt(len(diff))
            tcrit = stats.t.ppf(0.975, len(diff) - 1)
            r["d_macro_f1"], r["ci_lo"], r["ci_hi"] = diff.mean(), diff.mean() - tcrit * se, diff.mean() + tcrit * se
            r["p_macro_f1"] = base.paired_p(vec(v, "macro_f1"), full_f1)
            r["d_accuracy"] = (vec(v, "accuracy") - full_acc).mean()
            r["p_accuracy"] = base.paired_p(vec(v, "accuracy"), full_acc)
            # secondary: all fold x seed runs, paired
            a = d[d.variant == v].sort_values(["fold", "seed"])["macro_f1"].values
            b = d[d.variant == "full"].sort_values(["fold", "seed"])["macro_f1"].values
            r["p_macro_f1_runs"] = base.paired_p(a, b)
        rows.append(r)
    t = pd.DataFrame(rows)
    others = t.variant != "full"
    t.loc[others, "p_holm"] = base.holm_adjust(t.loc[others, "p_macro_f1"].values)
    return t, fold_df, metric_cols


def pretty_table(t, task):
    out = []
    for _, r in t.iterrows():
        row = {"Variant": r.label, "Type": r.group,
               "Macro-F1": f"{r.macro_f1_mean:.3f} ± {r.macro_f1_std:.3f}",
               "Top-1 acc (%)": f"{r.accuracy_mean:.2f} ± {r.accuracy_std:.2f}"}
        if task == "program":
            row["Top-3 acc (%)"] = f"{r.top3_mean:.2f} ± {r.top3_std:.2f}"
        if r.variant == "full":
            row.update({"Δ Macro-F1 vs full [95% CI]": "-", "p (paired t)": "-", "Holm p": "-"})
        else:
            row["Δ Macro-F1 vs full [95% CI]"] = f"{r.d_macro_f1:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]"
            row["p (paired t)"] = f"{r.p_macro_f1:.4f}"
            row["Holm p"] = f"{r.p_holm:.4f}"
        out.append(row)
    return pd.DataFrame(out)


def plot_delta(t, task):
    d = t[t.variant != "full"].sort_values("d_macro_f1")
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    y = np.arange(len(d))
    ax.barh(y, d.d_macro_f1, color=[GROUP_COLOR[g] for g in d.group], alpha=0.85)
    ax.errorbar(d.d_macro_f1, y, xerr=[d.d_macro_f1 - d.ci_lo, d.ci_hi - d.d_macro_f1], fmt="none", ecolor="k",
                capsize=3, lw=1)
    for yi, (_, r) in zip(y, d.iterrows()):
        star = "**" if r.p_holm < 0.05 else ("*" if r.p_macro_f1 < 0.05 else "")
        ax.text(r.ci_hi if r.d_macro_f1 >= 0 else r.ci_lo, yi, f" {star}", va="center", ha="left" if r.d_macro_f1 >= 0 else "right")
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d.label)
    ax.set_xlabel("Δ Macro-F1 relative to the full model (negative = the removed component helps)")
    ax.set_title(f"Ablation ({task} task): change in Macro-F1 with 95% CI over folds\n"
                 "* p<0.05 uncorrected, ** p<0.05 Holm-corrected")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for k, c in GROUP_COLOR.items() if k != "reference"]
    ax.legend(handles, [k for k in GROUP_COLOR if k != "reference"], frameon=False, loc="lower right")
    fig.tight_layout()
    return base.save_fig(fig, f"fig_a01_{task}_ablation_delta_macro_f1.png")


def plot_heatmap(t, task, metric_cols):
    cols = metric_cols
    lab = {"accuracy": "Top-1 %", "macro_f1": "Macro-F1", "bal_acc": "Bal-Acc %", "top3": "Top-3 %"}
    data = np.array([[r[f"{c}_mean"] for c in cols] for _, r in t.iterrows()])
    norm = (data - data.min(0)) / (data.max(0) - data.min(0) + 1e-12)
    fig, ax = plt.subplots(figsize=(1.7 * len(cols) + 4, 6.5))
    ax.imshow(norm, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([lab[c] for c in cols])
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t.label)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.3f}" if cols[j] == "macro_f1" else f"{data[i, j]:.1f}", ha="center",
                    va="center", fontsize=8, color="white" if norm[i, j] > 0.6 else "black")
    ax.set_title(f"Ablation ({task} task) - mean metrics (colour scaled per column)")
    fig.tight_layout()
    return base.save_fig(fig, f"fig_a02_{task}_ablation_heatmap.png")


def plot_box(fold_df, task):
    fig, ax = plt.subplots(figsize=(11, 5))
    data = [fold_df[fold_df.variant == v].sort_values("fold")["macro_f1"].values for v in VKEYS]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    for patch, v in zip(bp["boxes"], VKEYS):
        patch.set_facecolor(GROUP_COLOR[VGROUP[v]])
        patch.set_alpha(0.6)
    for i, a in enumerate(data):
        ax.scatter(np.full(len(a), i + 1), a, color="k", s=9, zorder=3)
    ax.axhline(np.mean(data[0]), color="#d62728", ls="--", lw=1)
    ax.set_xticks(range(1, len(VKEYS) + 1))
    ax.set_xticklabels([VLABEL[v] for v in VKEYS], rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Macro-F1 per fold (mean of seeds)")
    ax.set_title(f"Ablation ({task} task) - fold-wise Macro-F1 distribution")
    fig.tight_layout()
    return base.save_fig(fig, f"fig_a03_{task}_ablation_boxplot.png")


def plot_both_tasks(tabs):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    order = [v for v in VKEYS if v != "full"]
    for ax, task in zip(axes, ["program", "track"]):
        t = tabs[task].set_index("variant").loc[order]
        y = np.arange(len(order))
        ax.barh(y, t.d_macro_f1, color=[GROUP_COLOR[g] for g in t.group], alpha=0.85)
        ax.errorbar(t.d_macro_f1, y, xerr=[t.d_macro_f1 - t.ci_lo, t.ci_hi - t.d_macro_f1], fmt="none", ecolor="k",
                    capsize=2, lw=0.8)
        ax.axvline(0, color="k", lw=1)
        ax.set_title(f"{task} task")
        ax.set_xlabel("Δ Macro-F1 vs full model")
    axes[0].set_yticks(np.arange(len(order)))
    axes[0].set_yticklabels([VLABEL[v] for v in order])
    axes[0].invert_yaxis()
    fig.suptitle("Ablation summary - both tasks (95% CI over folds)")
    fig.tight_layout()
    return base.save_fig(fig, "fig_a04_ablation_both_tasks.png")


# =============================================================================
def main():
    t0 = time.time()
    print(f"[ablation] variants={len(VARIANTS)} seeds={SEEDS} epochs={TRAIN['epochs']} jobs={N_JOBS} "
          f"saber_pro_features={base.USE_SABER_PRO_FEATURES} temporal_steps={base.N_STEPS}")
    df, y_prog, programs, prog_track_idx, M = base.prepare_data()
    y_track = prog_track_idx[y_prog]
    K = len(programs)
    folds = list(StratifiedKFold(base.N_SPLITS, shuffle=True, random_state=SEED).split(np.zeros(len(df)), y_prog))
    jobs = [(task, v, f, tr, va, yy, KK) for task, yy, KK in (("program", y_prog, K), ("track", y_track, 4))
            for v in VKEYS for f, (tr, va) in enumerate(folds)]
    res = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(run_job)(task, v, f, tr, va, df, yy, KK) for task, v, f, tr, va, yy, KK in jobs)
    runs = pd.DataFrame([r for rows in res for r in rows])
    runs.to_csv(os.path.join(OUT, "ablation_all_runs.csv"), index=False)

    tabs, report = {}, ["OBJECTIVE 1 PART 3 - ABLATION STUDY", "=" * 70,
                        f"folds={base.N_SPLITS} seeds={len(SEEDS)} epochs={TRAIN['epochs']} | identical hyper-parameters | "
                        f"saber_pro_features={base.USE_SABER_PRO_FEATURES} | temporal steps T={base.N_STEPS}", ""]
    for task in ("program", "track"):
        t, fold_df, mc = analyse(runs, task)
        tabs[task] = t
        pt = pretty_table(t, task)
        pt.to_csv(os.path.join(OUT, f"ablation_{task}_table.csv"), index=False)
        t.to_csv(os.path.join(OUT, f"ablation_{task}_numeric.csv"), index=False)
        fold_df.to_csv(os.path.join(OUT, f"ablation_{task}_fold_metrics.csv"), index=False)
        try:
            with open(os.path.join(OUT, f"ablation_{task}_table.tex"), "w") as fh:
                fh.write(pt.to_latex(index=False, escape=True))
        except Exception:
            pass
        print("\n" + "=" * 130 + f"\nABLATION - {task.upper()} TASK\n" + "=" * 130)
        print(pt.to_string(index=False))
        plot_delta(t, task)
        plot_heatmap(t, task, mc)
        plot_box(fold_df, task)

        o = t[t.variant != "full"]
        hurt = o[(o.d_macro_f1 < 0) & (o.p_macro_f1 < 0.05)]
        hurt_h = o[(o.d_macro_f1 < 0) & (o.p_holm < 0.05)]
        helped = o[(o.d_macro_f1 > 0) & (o.p_macro_f1 < 0.05)]
        report += [f"## {task.upper()} TASK  (full model Macro-F1 {t.loc[t.variant == 'full', 'macro_f1_mean'].iloc[0]:.3f})",
                   f"Components whose removal significantly LOWERS Macro-F1 (p<0.05 uncorrected): "
                   f"{list(hurt.label) or 'none'}",
                   f"... after Holm correction: {list(hurt_h.label) or 'none'}",
                   f"Removals that significantly RAISE Macro-F1 (component not helping): {list(helped.label) or 'none'}",
                   f"Largest drop: {o.sort_values('d_macro_f1').iloc[0].label} "
                   f"({o.sort_values('d_macro_f1').iloc[0].d_macro_f1:+.4f}); "
                   f"largest gain: {o.sort_values('d_macro_f1').iloc[-1].label} "
                   f"({o.sort_values('d_macro_f1').iloc[-1].d_macro_f1:+.4f})", ""]
    plot_both_tasks(tabs)
    report.append("Interpretation guide: delta = variant - full; paired t-tests use 5 folds (4 d.o.f.) so power is "
                  "low; a non-significant delta means 'no evidence this component matters', not 'proven useless'. "
                  "With T=2 sequence steps the temporal branch sees only a very short pseudo-trajectory.")
    txt = "\n".join(report)
    print("\n" + txt)
    with open(os.path.join(OUT, "ablation_report.txt"), "w") as fh:
        fh.write(txt)
    print(f"\nAll outputs in ./{OUT}/ (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
