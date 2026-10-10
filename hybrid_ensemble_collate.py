#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joins the per-setting results of hybrid_ensemble_eval.py and applies the decision rule of HYBRID_ENSEMBLE_PROTOCOL.md:
  * a setting is SUPPORTED when TNE beats all three components on the primary metric and the within-setting
    Holm-adjusted p against the best component is < 0.05;
  * the overall claim needs >= 3 of 5 settings supported and no setting where TNE is significantly worse than a component.
A conservative Holm over all primary component contrasts is also reported.

Run:  python hybrid_ensemble_collate.py          (reads hybrid_ensemble_*/ensemble_contrasts.csv)
"""
import glob
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.environ.get("OUT_DIR", "hybrid_ensemble_summary")
os.makedirs(OUT, exist_ok=True)
EXPECTED = ["colombian_pre", "colombian_univ", "spanish_pooled_H2", "spanish_pooled_H3", "spanish_entrants_H2"]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    files = sorted(f for f in glob.glob("hybrid_ensemble_*/ensemble_contrasts.csv") if "summary" not in f)
    ct = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    prim = ct[(ct.tier == "primary") & (ct.family == "components")].copy()
    prim["p_holm_global"] = holm(prim.p.values)
    prim.to_csv(os.path.join(OUT, "primary_contrasts.csv"), index=False)
    rows, worse_any = [], False
    for s, g in prim.groupby("setting", sort=False):
        best = g.loc[g.mean_other.idxmax()]
        wins = bool((g["diff"] > 0).all())
        sup = bool(wins and best.p_holm < 0.05)
        worse = g[(g["diff"] < 0) & (g.p_holm < 0.05)]
        worse_any = worse_any or len(worse) > 0
        r = dict(setting=s, metric=g.metric.iloc[0], tne=g.mean_tne.iloc[0])
        for c in g.itertuples():
            r[c.contrast.split(" - ")[1]] = c.mean_other
        r.update(best_component=best.contrast.split(" - ")[1], diff_vs_best=best["diff"], ci_lo=best.ci_lo,
                 ci_hi=best.ci_hi, folds_better_than_best=int(best.folds_better), p_holm_within=best.p_holm,
                 p_holm_global=float(g.loc[g.mean_other.idxmax(), "p_holm_global"]), tne_beats_all=wins,
                 verdict="SUPPORTED" if sup else "NOT SUPPORTED",
                 significantly_worse_than=", ".join(worse.contrast.str.split(" - ").str[1]) or "none")
        rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    n_sup = int((summ.verdict == "SUPPORTED").sum())
    missing = [s for s in EXPECTED if s not in set(summ.setting)]
    overall = (n_sup >= 3) and not worse_any and not missing
    base_rows = ct[(ct.family == "baseline") & (ct.tier == "primary")]
    txt = ["CONFIRMATORY TEST OF THE TREE-NEURAL ENSEMBLE (decision rule from HYBRID_ENSEMBLE_PROTOCOL.md)", "=" * 78]
    txt += [f"{r.setting}: {r.metric} TNE {r.tne:.4f} | best component {r.best_component} {getattr(r, r.best_component):.4f} | "
            f"diff {r.diff_vs_best:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}], better in {r.folds_better_than_best}/5 folds, "
            f"Holm p within setting {r.p_holm_within:.4f} (global {r.p_holm_global:.4f}) -> {r.verdict}; "
            f"significantly worse than: {r.significantly_worse_than}" for r in summ.itertuples()]
    if len(base_rows):
        txt += ["", "TNE against the university-lookup baseline (Colombian, with university):"]
        txt += [f"  {r.setting} {r.metric} ({r.mode}): TNE {r.mean_tne:.4f} vs lookup {r.mean_other:.4f}, diff {r.diff:+.4f} "
                f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}], p {r.p:.4f}" for r in ct[ct.family == "baseline"].itertuples()]
    txt += ["", f"Settings supported: {n_sup} of {len(summ)}; settings missing: {missing or 'none'}; TNE significantly worse "
                f"than a component somewhere: {worse_any}",
            "OVERALL: " + ("the tree-neural ensemble is supported as the more suitable hybrid."
                           if overall else "the protocol's overall claim is NOT met; report the per-setting results as they are.")]
    open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(txt))
    print("\n".join(txt))

    fig, ax = plt.subplots(figsize=(9, 0.9 * len(prim) + 1.5))
    y = np.arange(len(prim))[::-1]
    ax.errorbar(prim["diff"], y, xerr=[prim["diff"] - prim.ci_lo, prim.ci_hi - prim["diff"]], fmt="o", color="#d62728",
                ecolor="k", capsize=3)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s}: vs {c.split(' - ')[1]}" for s, c in zip(prim.setting, prim.contrast)], fontsize=7)
    ax.set_xlabel("TNE minus component, primary metric (95% CI over 5 folds)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_primary_contrasts.png"), dpi=200)


if __name__ == "__main__":
    main()
