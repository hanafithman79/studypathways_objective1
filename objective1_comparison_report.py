#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-model comparison report: is the Proposed model better than the 7 hybrid baselines (and the rest)?

Reads fold-wise metrics written by objective1_track_head_and_tuning.py (EXTRA_HYBRIDS=1) and reports, per
scenario / tuning mode / metric:
  * mean rank of every model across the 5 folds (1 = best) and a Friedman test over all models,
  * paired t-test of Proposed vs every other model (Holm-corrected) and the effect size (Cohen's dz),
  * a verdict stating whether Proposed is best, tied, or worse - computed, not assumed,
  * trainable-parameter counts of the deep models.

Usage:  EXTRA_HYBRIDS=1 python objective1_comparison_report.py
Env:    DIR_PRE / DIR_UNIV  (default objective1_outputs_hybrids7_preuni / _univ), OUT_DIR
"""
import os
import sys

os.environ["EXTRA_HYBRIDS"] = "1"
os.environ.setdefault("USE_SABER_PRO_FEATURES", "0")
os.environ["OUT_DIR"] = os.environ.get("OUT_DIR", "objective1_outputs_comparison")
os.environ.setdefault("FEATURE_SET", "extended")
os.environ.setdefault("RESULTS_CSV", "objective1_comparison_results.csv")
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import objective1_experiments as base
import objective1_track_head_and_tuning as tune

OUT = os.environ["OUT_DIR"]
os.makedirs(OUT, exist_ok=True)
SCEN = {"Pre-university features": os.environ.get("DIR_PRE", "objective1_outputs_hybrids7_preuni"),
        "With university (recommendation)": os.environ.get("DIR_UNIV", "objective1_outputs_hybrids7_univ")}
MODES = ["tuned_macro_f1", "tuned_top3"]
METRICS = [("macro_f1", "Macro-F1"), ("top3", "Top-3 accuracy"), ("accuracy", "Top-1 accuracy")]
NAMES, PROPOSED = base.NAMES, base.PROPOSED

# ---- parameter counts (default configuration, 21 classes, extended static width of the pre-university set) ----
df, y, programs, pti, M = base.prepare_data()
d = base.preprocess_fold(df.iloc[:10000], df.iloc[10000:], y[:10000])
d_static = d["xs_tr"].shape[1]
params = {}
for k in base.DEEP_KEYS:
    net = tune.build_net(k, d_static, len(programs), tune.default_cfg(k))
    params[k] = sum(p.numel() for p in net.parameters() if p.requires_grad)
pd.Series({NAMES[k]: v for k, v in params.items()}, name="trainable_parameters").to_csv(
    os.path.join(OUT, "parameter_counts.csv"))

rows, verdicts = [], []
for scen, dname in SCEN.items():
    for mode in MODES:
        f = pd.read_csv(os.path.join(dname, f"program_{mode}_fold_metrics.csv"))
        f = f[f.model != "majority"]
        for m, mlabel in METRICS:
            piv = f.pivot(index="fold", columns="model", values=m)[base.KEYS]          # folds x models
            ranks = piv.rank(axis=1, ascending=False)                                  # 1 = best per fold
            fried = stats.friedmanchisquare(*[piv[c].values for c in piv.columns])
            others = [k for k in base.KEYS if k != PROPOSED]
            pvals, dz = {}, {}
            for k in others:
                diff = piv[PROPOSED].values - piv[k].values
                dz[k] = diff.mean() / (diff.std(ddof=1) + 1e-12)
                pvals[k] = base.paired_p(piv[PROPOSED].values, piv[k].values)
            holm = dict(zip(others, base.holm_adjust([pvals[k] for k in others])))
            for k in base.KEYS:
                rows.append(dict(scenario=scen, mode=mode, metric=mlabel, model=NAMES[k], mean=piv[k].mean(),
                                 std=piv[k].std(ddof=1), mean_rank=ranks[k].mean(),
                                 diff_vs_proposed=(piv[k].mean() - piv[PROPOSED].mean()) if k != PROPOSED else np.nan,
                                 p_vs_proposed=pvals.get(k, np.nan), p_holm=holm.get(k, np.nan),
                                 cohens_dz=dz.get(k, np.nan), friedman_p=fried.pvalue))
            best = piv.mean().idxmax()
            sig_better = [NAMES[k] for k in others if holm[k] < 0.05 and piv[PROPOSED].mean() > piv[k].mean()]
            sig_worse = [NAMES[k] for k in others if holm[k] < 0.05 and piv[PROPOSED].mean() < piv[k].mean()]
            rank_of_prop = int(ranks.mean().rank().loc[PROPOSED])
            verdicts.append(
                f"[{scen} | {mode} | {mlabel}] best mean = {NAMES[best]} ({piv[best].mean():.3f}); Proposed = "
                f"{piv[PROPOSED].mean():.3f}, mean-rank position {rank_of_prop} of {len(base.KEYS)}; Friedman p = "
                f"{fried.pvalue:.3f}; Holm-significantly better than: {sig_better or 'none'}; "
                f"Holm-significantly worse than: {sig_worse or 'none'}")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "comparison_all.csv"), index=False)
txt = "\n".join(["MODEL COMPARISON VERDICTS (computed from the fold-wise results)", "=" * 72, *verdicts, "",
                 "Note: 5 folds => Friedman and paired tests have low power; 'tied' means no detectable difference."])
open(os.path.join(OUT, "comparison_verdicts.txt"), "w").write(txt)
print(txt)

# ---- compact table: Macro-F1 and Top-3, tuned for Macro-F1 (primary) ----
for scen in SCEN:
    sub = res[(res.scenario == scen) & (res["mode"] == "tuned_macro_f1")]
    tab = sub.pivot_table(index="model", columns="metric", values=["mean", "mean_rank"], aggfunc="first")
    tab.to_csv(os.path.join(OUT, f"table_{scen.split()[0].lower()}_tuned_macro_f1.csv"))

# ---- figure: mean rank per model (Macro-F1 and Top-3), both scenarios ----
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
for i, scen in enumerate(SCEN):
    for j, (m, mlabel) in enumerate([("macro_f1", "Macro-F1"), ("top3", "Top-3 accuracy")]):
        ax = axes[i, j]
        sub = res[(res.scenario == scen) & (res["mode"] == ("tuned_macro_f1" if m == "macro_f1" else "tuned_top3"))
                  & (res.metric == mlabel)].set_index("model").loc[[NAMES[k] for k in base.KEYS]]
        cols = ["#d62728" if n == NAMES[PROPOSED] else "#9ecae1" for n in sub.index]
        ax.barh(sub.index, sub.mean_rank, color=cols)
        ax.invert_yaxis()
        ax.set_xlabel("mean rank over 5 folds (1 = best)")
        ax.set_title(f"{scen} - {mlabel}", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Mean rank of each model across folds (red = Proposed)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_mean_ranks.png"), dpi=200, bbox_inches="tight")

# ---- figure: performance vs model size ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, scen in zip(axes, SCEN):
    sub = res[(res.scenario == scen) & (res["mode"] == "tuned_macro_f1") & (res.metric == "Macro-F1")]
    for k in base.DEEP_KEYS:
        r = sub[sub.model == NAMES[k]].iloc[0]
        ax.scatter(params[k], r["mean"], s=70 if k == PROPOSED else 35, color="#d62728" if k == PROPOSED else "#4c78a8")
        ax.annotate(k, (params[k], r["mean"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("trainable parameters (default configuration)")
    ax.set_ylabel("Macro-F1 (tuned, 5-fold mean)")
    ax.set_title(scen, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Performance vs model size (deep models)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_performance_vs_size.png"), dpi=200, bbox_inches="tight")
print("\nparameters:", {k: params[k] for k in base.DEEP_KEYS})
print("saved to", OUT)
