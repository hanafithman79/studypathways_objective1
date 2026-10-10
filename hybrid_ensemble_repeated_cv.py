#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Addendum A1 of HYBRID_ENSEMBLE_PROTOCOL.md: repeated cross-validation for the Spanish new-entrants setting.
Reads repeated_cv_entrants/seed*/ensemble_fold_metrics.csv (one run of hybrid_ensemble_eval.py per fold seed) and tests
TNE minus each component with the Nadeau-Bengio corrected resampled t-test (variance factor 1/J + n_test/n_train,
J-1 degrees of freedom), Holm over the 3 components. Primary metric PR-AUC; secondary ROC-AUC. Also reports the same
test on the 20 differences that include the original seed-2024 run.
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

OUT = "repeated_cv_entrants"
COMP = ["rf", "hgb", "proposed"]
RATIO = 0.25                                     # n_test / n_train for 5-fold CV


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def load(paths):
    parts = []
    for f in paths:
        d = pd.read_csv(f)
        d["seed"] = f.split(os.sep)[-2] if "seed" in f else "seed2024"
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def analyse(fm, metric, label):
    wide = fm.pivot_table(index=["seed", "fold"], columns="method", values=metric)
    J = len(wide)
    rows = []
    for c in COMP:
        d = (wide["tne"] - wide[c]).values
        t = d.mean() / np.sqrt((1 / J + RATIO) * d.var(ddof=1))
        p = 2 * stats.t.sf(abs(t), J - 1)
        half = stats.t.ppf(0.975, J - 1) * np.sqrt((1 / J + RATIO) * d.var(ddof=1))
        rows.append(dict(analysis=label, metric=metric, J=J, contrast=f"tne - {c}", mean_tne=wide["tne"].mean(),
                         mean_other=wide[c].mean(), diff=d.mean(), ci_lo=d.mean() - half, ci_hi=d.mean() + half,
                         units_better=int((d > 0).sum()), p=p))
    r = pd.DataFrame(rows)
    r["p_holm"] = holm(r.p.values)
    return r


def main():
    new = sorted(glob.glob(f"{OUT}/seed*/ensemble_fold_metrics.csv"))
    fm = load(new)
    old = "hybrid_ensemble_spanish_entrants/ensemble_fold_metrics.csv"
    fm_all = pd.concat([fm, load([old])], ignore_index=True) if os.path.exists(old) else fm
    res = [analyse(fm, "pr_auc", "new seeds (primary)"), analyse(fm, "roc_auc", "new seeds (secondary)"),
           analyse(fm_all, "pr_auc", "new seeds + seed 2024 (secondary)")]
    res = pd.concat(res, ignore_index=True)
    res.to_csv(f"{OUT}/repeated_cv_contrasts.csv", index=False)
    by_seed = fm.pivot_table(index="seed", columns="method", values="pr_auc").round(4)
    by_seed.to_csv(f"{OUT}/pr_auc_by_seed.csv")
    prim = res[res.analysis == "new seeds (primary)"]
    best = prim.loc[prim.mean_other.idxmax()]
    wins = bool((prim["diff"] > 0).all())
    sup = bool(wins and best.p_holm < 0.05)
    worse = prim[(prim["diff"] < 0) & (prim.p_holm < 0.05)]
    txt = ["ADDENDUM A1: REPEATED CV, SPANISH NEW ENTRANTS AFTER SEMESTER 1 (post-protocol addition)", "=" * 78,
           f"seeds: {sorted(fm.seed.unique())}; units = {prim.J.iloc[0]} paired fold differences; "
           "Nadeau-Bengio corrected resampled t-test, Holm over 3 components", "",
           "PR-AUC per seed (mean over 5 folds):", by_seed.to_string(), ""]
    for a, g in res.groupby("analysis", sort=False):
        txt.append(a + ":")
        txt += [f"  {r.contrast}: TNE {r.mean_tne:.4f} vs {r.mean_other:.4f}, diff {r.diff:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}], "
                f"better in {r.units_better}/{r.J}, p {r.p:.4f}, Holm {r.p_holm:.4f}" for r in g.itertuples()]
    txt += ["", "VERDICT (primary): TNE beats all three components in mean: " + str(wins) + "; best component "
            + best.contrast.split(" - ")[1] + f" diff {best['diff']:+.4f}, Holm p {best.p_holm:.4f}; significantly worse than: "
            + str(list(worse.contrast) or "none") + " => " + ("SUPPORTED" if sup else "NOT SUPPORTED"),
            "This addendum does not change the main verdict (2 of 5 settings supported)."]
    open(f"{OUT}/repeated_cv_summary.txt", "w").write("\n".join(txt))
    print("\n".join(txt))


if __name__ == "__main__":
    main()
