#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIAGNOSTIC: is class imbalance the reason the Colombian Saber 11 programme task gives weak results?

Two checks on the pre-university features (the 14 static variables + Saber 11 scores, flattened), no deep models:
  1. Class-size table of the 21 programmes (how severe is the imbalance?).
  2. The five largest programmes (each >= 849 students) analysed twice:
       natural   the real class mix (accuracy is compared with the majority share),
       balanced  every programme down-sampled to the same size (chance = 1/5 = 20%).
     If the balanced accuracy is only slightly above chance, removing the imbalance does not rescue the task:
     the features carry little information about the programme, and imbalance mostly distorts how the results look.
5-fold stratified CV, 3 down-sampling seeds, preprocessing fitted on the training part only.

Run:   python colombian_balance_check.py            (from the repository folder, dataset.csv present)
Env:   OUT_DIR (default objective1_outputs_balance_check), N_SEEDS (3)
"""
import os
import sys
import warnings

os.environ["OUT_DIR"] = os.environ.get("OUT_DIR", "objective1_outputs_balance_check")
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

import objective1_experiments as base

warnings.filterwarnings("ignore")
OUT = base.OUT_DIR
os.makedirs(OUT, exist_ok=True)
N_SEEDS = int(os.environ.get("N_SEEDS", 3))
N_TOP = 5

MODELS = {
    "Logistic regression": lambda s: LogisticRegression(max_iter=2000),
    "Random forest": lambda s: RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=s),
    "Gradient boosting": lambda s: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, random_state=s),
}


def cv_scores(d, yy, seed, rows, setting):
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(d, yy):
        P = base.preprocess_fold(d.iloc[tr], d.iloc[va], yy[tr])
        for name, make in MODELS.items():
            pred = make(seed).fit(P["flat_tr"], yy[tr]).predict(P["flat_va"])
            rows.append(dict(setting=setting, model=name, accuracy=accuracy_score(yy[va], pred),
                             balanced_accuracy=balanced_accuracy_score(yy[va], pred),
                             macro_f1=f1_score(yy[va], pred, average="macro")))


def main():
    df, y, programs, _, _ = base.prepare_data()
    counts = np.bincount(y)
    order = np.argsort(-counts)
    table = pd.DataFrame({"programme": [programs[i] for i in order], "students": counts[order]})
    table["share_pct"] = (100 * table.students / table.students.sum()).round(1)
    table["cumulative_pct"] = table.share_pct.cumsum().round(1)
    table.to_csv(os.path.join(OUT, "class_sizes.csv"), index=False)
    print(table.to_string(index=False))

    top = order[:N_TOP]
    n_min = int(counts[top].min())
    keep = np.isin(y, top)
    d5, y5 = df[keep].reset_index(drop=True), y[keep]
    majority = float(counts[top].max() / counts[top].sum())

    rows = []
    cv_scores(d5, y5, 0, rows, "natural")
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        idx = np.concatenate([rng.choice(np.where(y5 == c)[0], n_min, replace=False) for c in top])
        cv_scores(d5.iloc[idx].reset_index(drop=True), y5[idx], seed, rows, "balanced")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "balance_check_all_folds.csv"), index=False)
    summary = res.groupby(["setting", "model"]).mean(numeric_only=True).round(3)
    summary.to_csv(os.path.join(OUT, "balance_check_summary.csv"))
    print("\n", summary)

    bal = summary.loc["balanced", "accuracy"]
    nat = summary.loc["natural", "accuracy"]
    txt = (f"CLASS-IMBALANCE DIAGNOSTIC (Colombian Saber 11, pre-university features)\n"
           f"Programmes: {len(counts)}; students: {int(counts.sum())}; largest/smallest = {counts.max() / counts.min():.0f}; "
           f"programmes with <100 students: {int((counts < 100).sum())}; <50: {int((counts < 50).sum())}; "
           f"single-student programmes: {int((counts == 1).sum())}\n"
           f"Five largest programmes: {[programs[i] for i in top]}\n"
           f"Natural mix (majority share {majority:.3f}): accuracy "
           f"{', '.join(f'{m} {nat[m]:.3f}' for m in nat.index)}\n"
           f"Balanced, {n_min} per programme (chance 0.200): accuracy "
           f"{', '.join(f'{m} {bal[m]:.3f}' for m in bal.index)}\n"
           f"Reading: the natural-mix accuracy sits at the majority share, and the perfectly balanced accuracy is "
           f"only {100 * (bal.min() - 0.2):.0f}-{100 * (bal.max() - 0.2):.0f} points above chance, so removing the "
           f"imbalance does not make the task easy: the features carry little information about the programme.")
    print("\n" + txt)
    open(os.path.join(OUT, "balance_check_report.txt"), "w").write(txt)


if __name__ == "__main__":
    main()
