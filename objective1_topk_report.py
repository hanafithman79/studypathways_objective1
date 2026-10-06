#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-k recommendation report for the 21-programme task (reads the saved out-of-fold probabilities).

For every model: Top-1..Top-5 accuracy per fold -> mean +- std, 95% CI (t, 4 d.o.f.) and the paired
difference to the "recommend the k most common programmes" floor. Produces tables and a figure.

Usage:  python objective1_topk_report.py [MODE]        (MODE default: tuned_top3)
Needs:  objective1_outputs_top3_preuni/  and  objective1_outputs_top3_univ/  (from the tuning script,
        FEATURE_SET=extended / extended_univ, TASKS=program, PROGRAM_SEL2=top3)
Output: ./objective1_outputs_topk/
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import top_k_accuracy_score

MODE = sys.argv[1] if len(sys.argv) > 1 else "tuned_top3"
SCEN = {"Pre-university features only": os.environ.get("DIR_PRE", "objective1_outputs_top3_preuni"),
        "With university (recommendation task)": os.environ.get("DIR_UNIV", "objective1_outputs_top3_univ")}
OUT = os.environ.get("OUT_DIR", "objective1_outputs_topk")
os.makedirs(OUT, exist_ok=True)
KS = [1, 2, 3, 4, 5]
NAMES = {"logreg": "Logistic Regression", "rf": "Random Forest", "hgb": "HistGradientBoosting",
         "static_mlp": "Static MLP", "lstm": "Standalone LSTM", "hyb1": "Hybrid-1 (MLP+CNN+LSTM)",
         "hyb2": "Hybrid-2 (MLP+BiLSTM)", "hyb3": "Hybrid-3 (MLP+GRU)", "hyb4": "Hybrid-4 (MLP+Transformer)",
         "proposed": "Proposed (MLP+CausalTCN+LSTM)"}
rows, curves = [], {}
for scen, d in SCEN.items():
    z = np.load(os.path.join(d, "oof_probabilities_tuning.npz"))
    y = z["y_program"]
    K = int(max(v.shape[1] for k, v in z.items() if k.startswith("program|")))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    per = {}
    for key in NAMES:
        P = z[f"program|{MODE}|{key}"]
        per[key] = np.array([[100 * top_k_accuracy_score(y[va], P[va], k=k, labels=np.arange(K)) for k in KS]
                             for _, va in folds])                  # (folds, ks)
    floor = []
    for tr, va in folds:                                           # k most common training programmes
        freq = np.bincount(y[tr], minlength=K) / len(tr)
        floor.append([100 * top_k_accuracy_score(y[va], np.tile(freq, (len(va), 1)), k=k, labels=np.arange(K))
                      for k in KS])
    per["floor"] = np.array(floor)
    curves[scen] = per
    for key, a in per.items():
        for j, k in enumerate(KS):
            v = a[:, j]
            se = v.std(ddof=1) / np.sqrt(len(v))
            t = stats.t.ppf(0.975, len(v) - 1)
            diff = v - per["floor"][:, j]
            rows.append(dict(scenario=scen, model=NAMES.get(key, "Floor: recommend the k most common programmes"),
                             k=k, mean=v.mean(), std=v.std(ddof=1), ci_lo=v.mean() - t * se, ci_hi=v.mean() + t * se,
                             gain_over_floor=diff.mean() if key != "floor" else np.nan,
                             p_vs_floor=(stats.ttest_rel(v, per["floor"][:, j]).pvalue if key != "floor" else np.nan)))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, f"topk_all_models_{MODE}.csv"), index=False)

# compact table: k = 1, 3, 5 for every model, both scenarios
t = df[df.k.isin([1, 3, 5])].copy()
t["cell"] = t.apply(lambda r: f"{r['mean']:.1f} ± {r['std']:.1f}", axis=1)
wide = t.pivot_table(index=["scenario", "model"], columns="k", values="cell", aggfunc="first")
wide.columns = [f"Top-{c} (%)" for c in wide.columns]
g3 = df[df.k == 3].set_index(["scenario", "model"])[["gain_over_floor", "p_vs_floor"]]
wide = wide.join(g3)
wide.to_csv(os.path.join(OUT, f"topk_summary_{MODE}.csv"))
print(wide.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
for ax, (scen, per) in zip(axes, curves.items()):
    for key, a in per.items():
        if key == "floor":
            continue
        ax.plot(KS, a.mean(0), marker="o", lw=2.6 if key == "proposed" else 1.1, alpha=1 if key == "proposed" else 0.7,
                color="#d62728" if key == "proposed" else None, label=NAMES[key])
    ax.plot(KS, per["floor"].mean(0), "k--", lw=1.8, label="Floor: k most common programmes")
    ax.axhline(80, color="grey", ls=":", lw=1)
    ax.axhline(85, color="grey", ls=":", lw=1)
    ax.text(1.02, 80.5, "80%", fontsize=7)
    ax.text(1.02, 85.5, "85%", fontsize=7)
    ax.set_xticks(KS)
    ax.set_xlabel("k (programmes recommended)")
    ax.set_title(scen)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Top-k accuracy (%)")
axes[1].legend(fontsize=6.5, frameon=False, loc="lower right")
fig.suptitle(f"Top-k programme recommendation accuracy vs the 'most common programmes' floor ({MODE}, 5-fold mean)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, f"fig_topk_curves_{MODE}.png"), dpi=200, bbox_inches="tight")
print("saved to", OUT)
