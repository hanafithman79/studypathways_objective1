#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Confirmatory test of the TREE-NEURAL ENSEMBLE (TNE) = unweighted mean of the predicted probabilities of
Random Forest, HistGradientBoosting and the Proposed model (MLP + causal TCN + LSTM).
The design, metrics, contrasts and decision rule are fixed in HYBRID_ENSEMBLE_PROTOCOL.md (committed before any run).

One setting per run (fresh random seed 2024 by default; the exploratory look used 42):
  DATASET=colombian FEATURE_SET=extended       SETTING=colombian_pre   python hybrid_ensemble_eval.py
  DATASET=colombian FEATURE_SET=extended_univ  SETTING=colombian_univ  python hybrid_ensemble_eval.py
  DATASET=spanish   HORIZONS=H2,H3             SETTING=spanish_pooled  python hybrid_ensemble_eval.py
  DATASET=spanish   HORIZONS=H2 ENTRANTS_ONLY=1 SEARCH_FRAC=1.0 SETTING=spanish_entrants python hybrid_ensemble_eval.py
then:  python hybrid_ensemble_collate.py
Env: SEED (2024), N_JOBS, N_TRIALS, QUICK_TEST=1 (smoke test), DATA_DIR / SPAIN_7Z (Spanish data, see spain_dropout_experiments.py)
"""
import os
import sys

os.environ.setdefault("SEED", "2024")
DATASET = os.environ.get("DATASET", "colombian")
SETTING = os.environ.get("SETTING", DATASET)
os.environ["OUT_DIR"] = os.environ.get("OUT_DIR", f"hybrid_ensemble_{SETTING}")
os.environ.setdefault("TASKS", "program")
os.environ.setdefault("EXTRA_HYBRIDS", "1")
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    sys.path.insert(0, os.getcwd())

import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, top_k_accuracy_score

import objective1_experiments as base

warnings.filterwarnings("ignore")
COMPONENTS = ["rf", "hgb", "proposed"]
N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 2))
OUT = base.OUT_DIR
os.makedirs(OUT, exist_ok=True)
COST = {"rf": 5, "hgb": 4, "proposed": 3}


def contrast_rows(setting, mode, metric, tier, fold_df, comps=COMPONENTS, extra=()):
    """TNE minus each component (Holm within the 3) and minus extra baselines (reported separately)."""
    def vec(m):
        return fold_df[fold_df.method == m].sort_values("fold")[metric].values
    rows = []
    for other in list(comps) + list(extra):
        d = vec("tne") - vec(other)
        se = d.std(ddof=1) / np.sqrt(len(d))
        tc = stats.t.ppf(0.975, len(d) - 1)
        rows.append(dict(setting=setting, mode=mode, metric=metric, tier=tier, contrast=f"tne - {other}",
                         family="components" if other in comps else "baseline", mean_tne=vec("tne").mean(),
                         mean_other=vec(other).mean(), diff=d.mean(), ci_lo=d.mean() - tc * se,
                         ci_hi=d.mean() + tc * se, folds_better=int((d > 0).sum()),
                         p=base.paired_p(vec("tne"), vec(other))))
    out = pd.DataFrame(rows)
    comp = out.family == "components"
    out.loc[comp, "p_holm"] = base.holm_adjust(out.loc[comp, "p"].values)
    return out


def verdict_line(setting, ct, mode, metric):
    c = ct[(ct["mode"] == mode) & (ct.metric == metric) & (ct.family == "components")]
    best = c.loc[c.mean_other.idxmax()]
    wins_all = bool((c["diff"] > 0).all())
    supported = bool(wins_all and best.p_holm < 0.05)
    worse = c[(c["diff"] < 0) & (c.p_holm < 0.05)]
    comps = ", ".join("{} {:.4f}".format(r.contrast.split(" - ")[1], r.mean_other) for r in c.itertuples())
    line = ("[{}] primary {} ({}): TNE {:.4f}; components {}; best component {}: diff {:+.4f} [{:+.4f}, {:+.4f}], "
            "Holm p {:.4f}; TNE better than all three: {}; significantly worse than: {} => {}").format(
        setting, metric, mode, c.mean_tne.iloc[0], comps, best.contrast.split(" - ")[1], best["diff"], best.ci_lo,
        best.ci_hi, best.p_holm, wins_all, list(worse.contrast) or "none", "SUPPORTED" if supported else "NOT SUPPORTED")
    return line, supported, bool(len(worse))


# =============================================================================
def run_colombian():
    import objective1_track_head_and_tuning as tune
    df, y, programs, pti, M = base.prepare_data()
    y_track = pti[y]
    K, N = len(programs), len(df)
    folds = list(StratifiedKFold(base.N_SPLITS, shuffle=True, random_state=base.SEED).split(np.zeros(N), y))
    univ = os.environ.get("FEATURE_SET", "spec") == "extended_univ"
    jobs = sorted([(COST[k], k, f, tr, va) for f, (tr, va) in enumerate(folds) for k in COMPONENTS], key=lambda j: -j[0])
    print(f"[ens] {SETTING}: N={N} K={K} seed={base.SEED} univ_feature={univ} jobs={len(jobs)} trials={tune.N_TRIALS}")
    res = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(tune.run_task)("program", k, f, tr, va, df, y, y_track, K, pti) for _, k, f, tr, va in jobs)
    by = {(r["model"], r["fold"]): r for r in res}
    modes = ["tuned_macro_f1", f"tuned_{tune.SEL2['program']}"]
    u = df["UNIVERSITY"].fillna("Missing").astype(str).values if univ else None
    rows = []
    for f, (tr, va) in enumerate(folds):
        freq = np.bincount(y[tr], minlength=K) / len(tr)
        majority_p = np.tile(freq + 1e-9 * np.arange(K)[::-1], (len(va), 1))      # frequency ranking for Top-3
        look = None
        if univ:
            ct = pd.crosstab(u[tr], y[tr]).reindex(columns=range(K), fill_value=0)
            look = np.array([(ct.loc[u[j]].values + 1e-3 * freq) / (ct.loc[u[j]].values.sum() + 1e-3) if u[j] in ct.index
                             else freq for j in va])
        for mode in modes:
            P = {k: by[(k, f)]["modes"][mode]["proba"].astype(np.float64) for k in COMPONENTS}
            P["tne"] = np.mean([P[k] for k in COMPONENTS], axis=0)
            P["majority"] = majority_p
            if univ:
                P["lookup"] = look
            for m, p in P.items():
                rows.append(dict(setting=SETTING, mode=mode, fold=f + 1, method=m, **tune.all_metrics(y[va], p, K, pti)))
    fm = pd.DataFrame(rows)
    fm.to_csv(os.path.join(OUT, "ensemble_fold_metrics.csv"), index=False)
    cts, lines, sup, worse_any = [], [], None, False
    plan = [("tuned_macro_f1", "macro_f1", "primary"), ("tuned_macro_f1", "bal_acc", "secondary"),
            (modes[1], "top3", "secondary"), (modes[1], "accuracy", "secondary")]
    for mode, metric, tier in plan:
        sub = fm[fm["mode"] == mode]
        cts.append(contrast_rows(SETTING, mode, metric, tier, sub, extra=(["lookup"] if univ else [])))
    ct = pd.concat(cts, ignore_index=True)
    ct.to_csv(os.path.join(OUT, "ensemble_contrasts.csv"), index=False)
    line, sup, worse_any = verdict_line(SETTING, ct, "tuned_macro_f1", "macro_f1")
    return fm, ct, [line], sup, worse_any


def run_spanish():
    import spain_dropout_experiments as S
    sy, seq = S.build_student_year()
    y = sy["y"].values
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=S.SEED).split(np.zeros(len(sy)), y, sy["group"].values))
    for tr, va in folds:
        assert not (set(sy["group"].values[tr]) & set(sy["group"].values[va])), "student leaked across folds"
    horizons = S.HORIZONS
    jobs = sorted([(COST[k], h, k, f, tr, va) for h in horizons for k in COMPONENTS
                   for f, (tr, va) in enumerate(folds)], key=lambda j: -j[0])
    print(f"[ens] {SETTING}: records={len(sy)} students={sy['group'].nunique()} dropout={100 * y.mean():.2f}% "
          f"seed={base.SEED} horizons={horizons} jobs={len(jobs)} trials={S.N_TRIALS} search_frac={S.SEARCH_FRAC}")
    res = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(S.run_job)(h, k, f, tr, va) for _, h, k, f, tr, va in jobs)
    by = {(r["horizon"], r["model"], r["fold"]): r for r in res}
    rows = []
    for h in horizons:
        for f, (tr, va) in enumerate(folds):
            sc = {k: by[(h, k, f)]["modes"]["tuned"]["score"].astype(np.float64) for k in COMPONENTS}
            sc["tne"] = np.mean([sc[k] for k in COMPONENTS], axis=0)
            for m, p in sc.items():
                top = p >= np.quantile(p, 0.90)
                rows.append(dict(setting=f"{SETTING}_{h}", mode="tuned", fold=f + 1, method=m,
                                 pr_auc=average_precision_score(y[va], p), roc_auc=roc_auc_score(y[va], p),
                                 capture_top10=float(y[va][top].sum() / y[va].sum())))
            rows.append(dict(setting=f"{SETTING}_{h}", mode="tuned", fold=f + 1, method="prevalence",
                             pr_auc=float(y[va].mean()), roc_auc=0.5, capture_top10=0.10))
    fm = pd.DataFrame(rows)
    fm.to_csv(os.path.join(OUT, "ensemble_fold_metrics.csv"), index=False)
    cts, lines, sups, worse = [], [], [], False
    for h in horizons:
        name = f"{SETTING}_{h}"
        sub = fm[fm.setting == name]
        for metric, tier in (("pr_auc", "primary"), ("roc_auc", "secondary"), ("capture_top10", "secondary")):
            cts.append(contrast_rows(name, "tuned", metric, tier, sub))
    ct = pd.concat(cts, ignore_index=True)
    ct.to_csv(os.path.join(OUT, "ensemble_contrasts.csv"), index=False)
    for h in horizons:
        name = f"{SETTING}_{h}"
        line, s, w = verdict_line(name, ct[ct.setting == name], "tuned", "pr_auc")
        lines.append(line); sups.append(s); worse = worse or w
    return fm, ct, lines, sups, worse


def main():
    t0 = time.time()
    fm, ct, lines, sup, worse = run_colombian() if DATASET == "colombian" else run_spanish()
    show = ct[ct.tier == "primary"][["setting", "metric", "contrast", "mean_tne", "mean_other", "diff", "ci_lo", "ci_hi",
                                     "folds_better", "p", "p_holm"]]
    print("\n" + show.round(4).to_string(index=False))
    txt = "CONFIRMATORY TNE TEST (protocol: HYBRID_ENSEMBLE_PROTOCOL.md), seed " + str(base.SEED) + "\n" + "\n".join(lines)
    print("\n" + txt)
    open(os.path.join(OUT, "ensemble_report.txt"), "w").write(txt)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
