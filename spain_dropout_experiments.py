#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
SPANISH UNIVERSITY DROPOUT STUDY - dual-branch hybrid vs 12 baselines
===============================================================================
Data: three CSV files (academic years 2018/19, 2021/22, 2022/23; ';'-separated, decimal comma), one row per
student x subject. Source: "University Student Dropout: A Longitudinal Dataset of Demographic, Socioeconomic,
and Academic Indicators" (Data 2025, doi:10.3390/data10100162).

Unit of analysis : one record per student x degree x academic year (60,924 records, 39,364 students).
Target           : abandono_hash == "A"  (7% of records). The label is hashed in the release; "A" is read as
                   "dropout" because those students re-appear in the next year's file far less often
                   (16% vs 66%) and pass 37% vs 77% of their credits. CONFIRM against the paper's codebook.
Validation       : StratifiedGroupKFold(5) - a student is never in training and test at the same time.

Three prediction points (what is known when the prediction is made):
  H1  enrolment time : static background + the pass rates of the previous 1-3 years (pre-year trajectory)
  H2  after semester 1: H1 + learning-platform activity Sep-Jan (5 monthly steps) + semester-1 credits
  H3  full year      : H2 + activity Sep-Aug (12 steps) + semester-2 / total credits + subject grades

Excluded on purpose (they can encode the outcome): matricula_activa, baja_fecha, cumulative degree-level credits
(cred_sup_tit, cred_pend_sup_tit - the files were extracted in June 2023 for every year), all-null columns.

13 models: 3 classical, 2 single-branch deep, 7 hybrid baselines, the proposed dual-branch hybrid.
Nested tuning: configurations ranked on an inner group-disjoint hold-out by PR-AUC; the share of students to
flag is chosen on the same hold-out (maximising Macro-F1) and applied to the outer fold as a rate.

Run:   python spain_dropout_experiments.py            (QUICK_TEST=1 for a smoke test)
Env:   DATA_DIR (folder with the 3 CSVs, or SPAIN_7Z=path to spanishdatasets.7z -> needs `pip install py7zr`),
       HORIZONS (default "H2,H3,H1"), N_TRIALS (default 6), N_JOBS, SEARCH_FRAC (default 0.5), OUT_DIR
===============================================================================
"""
import os
import sys

os.environ.setdefault("EXTRA_HYBRIDS", "1")
os.environ.setdefault("USE_SABER_PRO_FEATURES", "0")
os.environ["OUT_DIR"] = os.environ.get("OUT_DIR", "spain_outputs")
os.environ.setdefault("RESULTS_CSV", "spain_unused.csv")
os.environ.setdefault("FEATURE_SET", "spec")
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

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
                             balanced_accuracy_score, accuracy_score, precision_recall_curve, roc_curve,
                             confusion_matrix)

import objective1_experiments as base
import objective1_track_head_and_tuning as tune

warnings.filterwarnings("ignore")

QUICK = base.QUICK_TEST
SEED = base.SEED
OUT = base.OUT_DIR
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)
N_TRIALS = int(os.environ.get("N_TRIALS", 2 if QUICK else 6))
N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 2))
SEARCH_FRAC = float(os.environ.get("SEARCH_FRAC", 0.3 if QUICK else 0.5))
RESUME = os.environ.get("RESUME", "1") == "1"
HORIZONS = os.environ.get("HORIZONS", "H2,H3,H1").split(",")
DATA_DIR = os.environ.get("DATA_DIR", "spain_data")
KEYS, NAMES, CATEGORY = base.KEYS, base.NAMES, base.CATEGORY
DEEP_KEYS, PROPOSED = set(base.DEEP_KEYS), base.PROPOSED
YEARS = (2018, 2021, 2022)
LMS_METRICS = ["pft_events", "pft_total_minutes", "resource_events", "n_resource_days"]  # populated in all years
N_MONTHS = 12
LMS_FEATS = LMS_METRICS + ["active_subjects"]

# =============================================================================
# 1. DATA: one record per student x degree x year
# =============================================================================
FIRST_COLS = ["tipo_ingreso", "nota10_hash", "nota14_hash", "campus_hash", "estudios_p_hash", "estudios_m_hash",
              "dedicacion", "desplazado_hash", "abandono_hash", "preferencia_seleccion", "anyo_ingreso",
              "anyo_inicio_estudios", "curso_mas_bajo", "curso_mas_alto", "cred_mat_total", "cred_mat_sem_a",
              "cred_mat_sem_b", "cred_sup_total", "cred_sup_sem_a", "cred_sup_sem_b", "rend_total_ultimo",
              "rend_total_penultimo", "rend_total_antepenultimo", "es_retitulado", "es_adaptado"]
STATIC_CAT = ["tit_hash", "tipo_ingreso", "campus_hash", "estudios_p_hash", "estudios_m_hash", "dedicacion",
              "desplazado_hash", "acad_year", "retitulado", "adaptado"]
STATIC_NUM = ["nota10", "nota14", "preferencia_seleccion", "years_since_entry", "years_since_start",
              "curso_mas_bajo", "curso_mas_alto", "cred_mat_total", "cred_mat_sem_a", "cred_mat_sem_b",
              "n_subjects", "prior_1", "prior_2", "prior_3"]
HORIZON_SPEC = {
    "H1": dict(label="H1 enrolment time", extra=[], seq="prior", months=0),
    "H2": dict(label="H2 after semester 1", extra=["cred_sup_sem_a", "pass_rate_sem_a", "has_lms"], seq="lms", months=5),
    "H3": dict(label="H3 full year", extra=["cred_sup_sem_a", "pass_rate_sem_a", "cred_sup_sem_b", "cred_sup_total",
                                            "pass_rate_total", "mean_grade", "share_failed", "has_lms"],
               seq="lms", months=12),
}


def find_csvs():
    if all(os.path.exists(os.path.join(DATA_DIR, f"dataset_{y}_hash.csv")) for y in YEARS):
        return DATA_DIR
    z = os.environ.get("SPAIN_7Z")
    if z and os.path.exists(z):
        import py7zr
        os.makedirs(DATA_DIR, exist_ok=True)
        with py7zr.SevenZipFile(z) as arc:
            arc.extractall(path=DATA_DIR)
        return DATA_DIR
    raise FileNotFoundError(f"Put dataset_2018_hash.csv, dataset_2021_hash.csv, dataset_2022_hash.csv in "
                            f"'{DATA_DIR}' (or set SPAIN_7Z to the .7z archive).")


def load_year(path, Y):
    hdr = pd.read_csv(path, sep=";", nrows=0).columns
    months = [(Y, m) for m in (9, 10, 11, 12)] + [(Y + 1, m) for m in range(1, 9)]
    lms_cols = {(mi, met): f"{met}_{yy}_{mm}" for mi, (yy, mm) in enumerate(months) for met in LMS_METRICS
                if f"{met}_{yy}_{mm}" in hdr}
    use = ["dni_hash", "tit_hash", "asi_hash", "nota_asig_hash"] + FIRST_COLS + list(lms_cols.values())
    d = pd.read_csv(path, sep=";", decimal=",", usecols=lambda c: c in set(use), low_memory=False)
    key = [d["dni_hash"], d["tit_hash"]]
    g = d.groupby(["dni_hash", "tit_hash"], sort=False)
    sy = g[FIRST_COLS].first()
    sy["n_subjects"] = g.size()
    grade = pd.to_numeric(d["nota_asig_hash"], errors="coerce")
    sy["mean_grade"] = grade.groupby(key, sort=False).mean()
    sy["share_failed"] = (grade < 5).where(grade.notna()).groupby(key, sort=False).mean()
    sy["has_lms"] = d[list(lms_cols.values())].notna().any(axis=1).groupby(key, sort=False).max().astype(float)
    seq = np.zeros((len(sy), N_MONTHS, len(LMS_FEATS)), dtype=np.float32)
    for (mi, met), col in lms_cols.items():
        v = pd.to_numeric(d[col], errors="coerce").fillna(0.0)
        seq[:, mi, LMS_METRICS.index(met)] = v.groupby(key, sort=False).sum().reindex(sy.index).values
        if met == "pft_events":
            seq[:, mi, len(LMS_METRICS)] = (v > 0).groupby(key, sort=False).sum().reindex(sy.index).values
    sy = sy.reset_index()
    sy["acad_year"] = str(Y)
    sy["year_num"] = Y
    return sy, seq


ENTRANTS_ONLY = os.environ.get("ENTRANTS_ONLY", "0") == "1"   # keep only records in the student's entry year


def build_student_year():
    sy, seq = _build_full()
    if ENTRANTS_ONLY:                                  # secondary-school leavers in their entry year
        mask = (sy["years_since_entry"] == 0).values
        sy, seq = sy[mask].reset_index(drop=True), seq[mask]
    return sy, seq


def _build_full():
    cache = os.path.join(CACHE, "spain_student_year.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    folder = find_csvs()
    parts, seqs = [], []
    for Y in YEARS:
        sy, seq = load_year(os.path.join(folder, f"dataset_{Y}_hash.csv"), Y)
        parts.append(sy)
        seqs.append(seq)
        print(f"[data] {Y}: {len(sy)} student-degree records, dropout(A) {100 * (sy.abandono_hash == 'A').mean():.1f}%")
    sy = pd.concat(parts, ignore_index=True)
    seq = np.concatenate(seqs)
    sy["y"] = (sy["abandono_hash"] == "A").astype(int)
    sy["nota10"], sy["nota14"] = sy["nota10_hash"], sy["nota14_hash"]
    sy["years_since_entry"] = sy["year_num"] - sy["anyo_ingreso"]
    sy["years_since_start"] = sy["year_num"] - sy["anyo_inicio_estudios"]
    sy["prior_1"], sy["prior_2"], sy["prior_3"] = (sy["rend_total_ultimo"], sy["rend_total_penultimo"],
                                                   sy["rend_total_antepenultimo"])
    sy["pass_rate_sem_a"] = sy["cred_sup_sem_a"] / sy["cred_mat_sem_a"].replace(0, np.nan) * 100
    sy["pass_rate_total"] = sy["cred_sup_total"] / sy["cred_mat_total"].replace(0, np.nan) * 100
    sy["retitulado"] = sy["es_retitulado"].notna().astype(str)
    sy["adaptado"] = sy["es_adaptado"].notna().astype(str)
    for c in STATIC_CAT:
        sy[c] = sy[c].astype(str).fillna("missing").replace({"nan": "missing"})
    sy["group"] = sy["dni_hash"].astype(str)
    out = (sy, seq)
    with open(cache, "wb") as fh:
        pickle.dump(out, fh)
    return out


# =============================================================================
# 2. FOLD-WISE FEATURES (fitted on the training part only)
# =============================================================================
def horizon_sequences(h, seq_all, sy):
    spec = HORIZON_SPEC[h]
    if spec["seq"] == "prior":
        rates = np.stack([sy["prior_3"].values, sy["prior_2"].values, sy["prior_1"].values], axis=1)
        obs = ~np.isnan(rates)
        s = np.stack([np.nan_to_num(rates, nan=0.0) / 100.0, obs.astype(float)], axis=2).astype(np.float32)
        return s, False
    return np.log1p(np.clip(seq_all[:, :spec["months"], :], 0, None)).astype(np.float32), True


def fold_data(h, tr, va, sy, seq):
    spec = HORIZON_SPEC[h]
    num_cols = STATIC_NUM + spec["extra"]
    X = sy[STATIC_CAT + num_cols]
    ct = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30, sparse_output=False),
         STATIC_CAT),
        ("num", Pipeline([("i", SimpleImputer(strategy="median", add_indicator=True)), ("s", StandardScaler())]),
         num_cols)])
    xs_tr = ct.fit_transform(X.iloc[tr]).astype(np.float32)
    xs_va = ct.transform(X.iloc[va]).astype(np.float32)
    mu = seq[tr].mean(axis=(0, 1), keepdims=True)
    sd = seq[tr].std(axis=(0, 1), keepdims=True) + 1e-6
    sq_tr, sq_va = ((seq[tr] - mu) / sd).astype(np.float32), ((seq[va] - mu) / sd).astype(np.float32)
    return dict(xs_tr=xs_tr, xs_va=xs_va, seq_tr=sq_tr, seq_va=sq_va,
                flat_tr=np.hstack([xs_tr, sq_tr.reshape(len(tr), -1)]),
                flat_va=np.hstack([xs_va, sq_va.reshape(len(va), -1)]))


# =============================================================================
# 3. SCORES
# =============================================================================
Q_GRID = np.linspace(0.02, 0.50, 49)


def best_rate(y, p):
    best, bq = -1, 0.1
    for q in Q_GRID:
        pred = (p >= np.quantile(p, 1 - q)).astype(int)
        f = f1_score(y, pred, average="macro", zero_division=0)
        if f > best:
            best, bq = f, q
    return bq


def score_all(y, p, q):
    pred = (p >= np.quantile(p, 1 - q)).astype(int)
    top = p >= np.quantile(p, 0.90)
    return dict(roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                macro_f1=f1_score(y, pred, average="macro", zero_division=0),
                f1_dropout=f1_score(y, pred, zero_division=0),
                precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred, zero_division=0),
                bal_acc=balanced_accuracy_score(y, pred), accuracy=accuracy_score(y, pred),
                capture_top10=float(y[top].sum() / max(y.sum(), 1)), flag_rate=q)


def make_cfgs(key, rng):
    cfgs = [tune.default_cfg(key)] + [tune.sample_cfg(key, rng) for _ in range(N_TRIALS - 1)]
    out = []
    for i, c in enumerate(cfgs):
        c = dict(c)
        if key in DEEP_KEYS:
            c["batch_size"] = 512 if i == 0 else int(rng.choice([512, 1024]))
            c["epochs"] = 15 if i == 0 else int(rng.choice([10, 15, 20]))
        out.append(tune.finalize_cfg(key, c))
    return out


# =============================================================================
# 4. ONE (horizon, model, fold) JOB
# =============================================================================
_STATE = {}


def state():
    if not _STATE:
        sy, seq = build_student_year()
        _STATE["sy"], _STATE["seq"] = sy, seq
    return _STATE["sy"], _STATE["seq"]


def run_job(h, key, fold, tr, va):
    import torch
    torch.set_num_threads(1)
    path = os.path.join(CACHE, f"{h}_{key}_f{fold}.pkl")
    if RESUME and os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    t0 = time.time()
    sy, seq_all = state()
    y = sy["y"].values
    groups = sy["group"].values
    seq, _ = horizon_sequences(h, seq_all, sy)
    base.N_FEATS_STEP = seq.shape[2]                       # networks read the per-step width from here
    rng = np.random.default_rng(SEED * 7919 + fold * 131 + KEYS.index(key) + 17 * {"H1": 0, "H2": 1, "H3": 2}[h])
    cfgs = make_cfgs(key, rng)

    # ---- inner, student-disjoint hold-out (outer-validation students never touched) ----
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + fold)
    i_tr, i_va = next(sgk.split(tr, y[tr], groups[tr]))
    in_tr, in_va = tr[i_tr], tr[i_va]
    sub = np.random.default_rng(SEED + fold).random(len(in_tr)) < SEARCH_FRAC
    d_in = fold_data(h, in_tr[sub], in_va, sy, seq)
    trials, in_scores = [], []
    for t, cfg in enumerate(cfgs):
        pr = tune.fit_predict(key, cfg, d_in, y[in_tr[sub]], 2, SEED + fold)[:, 1]
        in_scores.append(pr)
        q = best_rate(y[in_va], pr)
        trials.append(dict(horizon=h, model=key, fold=fold + 1, trial=t, cfg=cfg, q=q,
                           inner_pr_auc=average_precision_score(y[in_va], pr),
                           inner_macro_f1=score_all(y[in_va], pr, q)["macro_f1"]))
    best = max(trials, key=lambda r: (r["inner_pr_auc"], -r["trial"]))["trial"]

    # ---- refit on the whole outer-training part, evaluate once on the outer fold ----
    d_out = fold_data(h, tr, va, sy, seq)
    modes, fitted = {}, {}
    for mode, t in (("default", 0), ("tuned", best)):
        if t not in fitted:
            fitted[t] = tune.fit_predict(key, cfgs[t], d_out, y[tr], 2, SEED + fold)[:, 1]
        q = trials[t]["q"]
        modes[mode] = dict(trial=t, cfg=cfgs[t], q=q, score=fitted[t].astype(np.float32),
                           metrics=score_all(y[va], fitted[t], q))
    out = dict(horizon=h, model=key, fold=fold, va=va, trials=trials, modes=modes, seconds=time.time() - t0)
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    return out


# =============================================================================
# 5. AGGREGATION, STATISTICS, FIGURES
# =============================================================================
METRIC_LABELS = {"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "macro_f1": "Macro-F1", "f1_dropout": "Dropout F1",
                 "recall": "Dropout recall", "precision": "Dropout precision", "capture_top10": "Captured in top 10%",
                 "bal_acc": "Balanced accuracy"}


def aggregate(results, h, mode, y_all, folds):
    rows = [dict(model=r["model"], fold=r["fold"] + 1, **r["modes"][mode]["metrics"])
            for r in results if r["horizon"] == h]
    return pd.DataFrame(rows), None


def paired_tests(f, metric):
    piv = f.pivot(index="fold", columns="model", values=metric)[KEYS]
    others = [k for k in KEYS if k != PROPOSED]
    p = {k: base.paired_p(piv[PROPOSED].values, piv[k].values) for k in others}
    holm = dict(zip(others, base.holm_adjust([p[k] for k in others])))
    dz = {k: (lambda d: d.mean() / (d.std(ddof=1) + 1e-12))(piv[PROPOSED].values - piv[k].values) for k in others}
    ranks = piv.rank(axis=1, ascending=False).mean()
    fried = stats.friedmanchisquare(*[piv[c].values for c in piv.columns]).pvalue
    return piv, p, holm, dz, ranks, fried


def build_report(results, folds, y_all, sy):
    summary_rows, verdicts, tables = [], [], {}
    for h in HORIZONS:
        for mode in ("default", "tuned"):
            f, _ = aggregate(results, h, mode, y_all, folds)
            f.to_csv(os.path.join(OUT, f"{h}_{mode}_fold_metrics.csv"), index=False)
            piv_pr, p_pr, holm_pr, dz_pr, ranks_pr, fried_pr = paired_tests(f, "pr_auc")
            _, p_f1, holm_f1, dz_f1, ranks_f1, fried_f1 = paired_tests(f, "macro_f1")
            prev = float(np.mean([y_all[va].mean() for _, va in folds]))
            tab = []
            for k in KEYS:
                row = {"Model": NAMES[k], "Category": CATEGORY[k]}
                for m in ("pr_auc", "roc_auc", "macro_f1", "recall", "capture_top10"):
                    v = f[f.model == k][m]
                    row[METRIC_LABELS[m]] = f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"
                row["Mean rank PR-AUC"] = f"{ranks_pr[k]:.1f}"
                row["p vs Proposed (PR-AUC)"] = "-" if k == PROPOSED else f"{p_pr[k]:.4f}"
                row["Holm p"] = "-" if k == PROPOSED else f"{holm_pr[k]:.4f}"
                row["p vs Proposed (Macro-F1)"] = "-" if k == PROPOSED else f"{p_f1[k]:.4f}"
                tab.append(row)
            tab.append({"Model": "Reference (no skill)", "Category": "reference", "PR-AUC": f"{prev:.3f}",
                        "ROC-AUC": "0.500", "Macro-F1": f"{(1 - prev) / (2 - prev):.3f}", "Dropout recall": "-",
                        "Captured in top 10%": "0.100"})
            tab = pd.DataFrame(tab)
            tab.to_csv(os.path.join(OUT, f"{h}_{mode}_results.csv"), index=False)
            tables[(h, mode)] = tab
            means = f.groupby("model")[list(METRIC_LABELS)].mean()
            best = means["pr_auc"].idxmax()
            sig_b = [NAMES[k] for k in KEYS if k != PROPOSED and holm_pr[k] < 0.05 and
                     means.loc[PROPOSED, "pr_auc"] > means.loc[k, "pr_auc"]]
            sig_w = [NAMES[k] for k in KEYS if k != PROPOSED and holm_pr[k] < 0.05 and
                     means.loc[PROPOSED, "pr_auc"] < means.loc[k, "pr_auc"]]
            verdicts.append(
                f"[{HORIZON_SPEC[h]['label']} | {mode}] prevalence {prev:.3f}. Best PR-AUC: {NAMES[best]} "
                f"{means.loc[best, 'pr_auc']:.3f} (ROC-AUC {means.loc[best, 'roc_auc']:.3f}). Proposed: PR-AUC "
                f"{means.loc[PROPOSED, 'pr_auc']:.3f}, ROC-AUC {means.loc[PROPOSED, 'roc_auc']:.3f}, Macro-F1 "
                f"{means.loc[PROPOSED, 'macro_f1']:.3f}, mean-rank {ranks_pr[PROPOSED]:.1f}/{len(KEYS)}; Friedman "
                f"p={fried_pr:.4f}; Holm-significantly better than {sig_b or 'none'}; worse than {sig_w or 'none'}.")
            for k in KEYS:
                summary_rows.append(dict(horizon=h, mode=mode, model=NAMES[k],
                                         **{m: means.loc[k, m] for m in METRIC_LABELS}))
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUT, "summary_all.csv"), index=False)
    return verdicts, tables


def fig_bars(results, folds, y_all, h, mode):
    f, _ = aggregate(results, h, mode, y_all, folds)
    metrics = ["pr_auc", "roc_auc", "macro_f1", "capture_top10"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)
    for ax, m in zip(axes, metrics):
        g = f.groupby("model")[m].agg(["mean", "std"]).reindex(KEYS)
        ax.barh([NAMES[k] for k in KEYS], g["mean"], xerr=g["std"], capsize=2,
                color=[base.COLORS[CATEGORY[k]] for k in KEYS])
        ax.invert_yaxis()
        ax.set_title(METRIC_LABELS[m])
        if m == "pr_auc":
            ax.axvline(float(np.mean([y_all[va].mean() for _, va in folds])), color="k", ls=":", lw=1)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Dropout prediction - {HORIZON_SPEC[h]['label']} ({mode}); 5-fold grouped CV, mean ± std")
    fig.tight_layout()
    return base.save_fig(fig, f"fig_{h}_{mode}_metric_bars.png")


def pooled_scores(results, h, mode, n):
    out = {}
    for r in results:
        if r["horizon"] == h:
            out.setdefault(r["model"], np.zeros(n))[r["va"]] = r["modes"][mode]["score"]
    return out


def fig_curves(results, y_all, h, mode):
    sc = pooled_scores(results, h, mode, len(y_all))
    f_means = {k: average_precision_score(y_all, v) for k, v in sc.items()}
    top = sorted(f_means, key=f_means.get, reverse=True)[:3]
    show = list(dict.fromkeys(top + [PROPOSED]))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for k in show:
        pr, rc, _ = precision_recall_curve(y_all, sc[k])
        axes[0].plot(rc, pr, lw=2.4 if k == PROPOSED else 1.3, color="#d62728" if k == PROPOSED else None, label=NAMES[k])
        fpr, tpr, _ = roc_curve(y_all, sc[k])
        axes[1].plot(fpr, tpr, lw=2.4 if k == PROPOSED else 1.3, color="#d62728" if k == PROPOSED else None)
        order = np.argsort(-sc[k])
        cum = np.cumsum(y_all[order]) / y_all.sum()
        axes[2].plot(np.arange(1, len(cum) + 1) / len(cum), cum, lw=2.4 if k == PROPOSED else 1.3,
                     color="#d62728" if k == PROPOSED else None)
    axes[0].axhline(y_all.mean(), color="k", ls=":")
    axes[0].set_xlabel("recall")
    axes[0].set_ylabel("precision")
    axes[0].set_title("Precision-recall (pooled out-of-fold)")
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].plot([0, 1], [0, 1], "k:")
    axes[1].set_xlabel("false-positive rate")
    axes[1].set_ylabel("true-positive rate")
    axes[1].set_title("ROC")
    axes[2].plot([0, 1], [0, 1], "k:")
    axes[2].set_xlabel("share of students flagged (highest risk first)")
    axes[2].set_ylabel("share of dropouts captured")
    axes[2].set_title("Capture (gain) curve")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{HORIZON_SPEC[h]['label']} ({mode}) - proposed model in red vs the three best by PR-AUC")
    fig.tight_layout()
    return base.save_fig(fig, f"fig_{h}_{mode}_curves.png")


def fig_confusion(results, y_all, h, mode):
    q = np.mean([r["modes"][mode]["q"] for r in results if r["horizon"] == h and r["model"] == PROPOSED])
    s = pooled_scores(results, h, mode, len(y_all))[PROPOSED]
    pred = (s >= np.quantile(s, 1 - q)).astype(int)
    cm = confusion_matrix(y_all, pred)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.55 else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["stays", "dropout"])
    ax.set_yticklabels(["stays", "dropout"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(f"Proposed, {HORIZON_SPEC[h]['label']} ({mode})\nflagging top {100 * q:.0f}% by risk")
    fig.tight_layout()
    return base.save_fig(fig, f"fig_{h}_{mode}_confusion_proposed.png")


def fig_horizons(results, folds, y_all):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, m in zip(axes, ["pr_auc", "roc_auc"]):
        for k in [PROPOSED, "hgb", "rf", "logreg", "hyb3"]:
            vals = []
            for h in HORIZONS:
                f, _ = aggregate(results, h, "tuned", y_all, folds)
                vals.append(f[f.model == k][m].mean())
            ax.plot([HORIZON_SPEC[h]["label"] for h in HORIZONS], vals, marker="o",
                    lw=2.6 if k == PROPOSED else 1.3, color="#d62728" if k == PROPOSED else None, label=NAMES[k])
        ax.set_title(METRIC_LABELS[m] + " by prediction point (tuned)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return base.save_fig(fig, "fig_horizons.png")


def main():
    t0 = time.time()
    print(f"[spain] horizons={HORIZONS} trials={N_TRIALS} jobs={N_JOBS} search_frac={SEARCH_FRAC} quick={QUICK}")
    sy, seq = build_student_year()
    y = sy["y"].values
    print(f"[spain] {len(sy)} student-year records, {sy['group'].nunique()} students, dropout {100 * y.mean():.2f}%")
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(sgk.split(np.zeros(len(sy)), y, sy["group"].values))
    for i, (tr, va) in enumerate(folds):
        assert not (set(sy["group"].values[tr]) & set(sy["group"].values[va])), "student leaked across folds"
        print(f"  fold {i + 1}: train {len(tr)} test {len(va)} dropout {100 * y[va].mean():.2f}%")
    cost = {"rf": 5, "hgb": 3, "logreg": 1}
    jobs = sorted([(cost.get(k, 4), h, k, f, tr, va) for h in HORIZONS for k in KEYS for f, (tr, va) in enumerate(folds)],
                  key=lambda j: -j[0])
    print(f"[spain] {len(jobs)} jobs")
    results = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(run_job)(h, k, f, tr, va) for _, h, k, f, tr, va in jobs)

    pd.DataFrame([dict(t, cfg=str(t["cfg"])) for r in results for t in r["trials"]]).to_csv(
        os.path.join(OUT, "all_search_trials.csv"), index=False)
    pd.DataFrame([dict(horizon=r["horizon"], model=r["model"], fold=r["fold"] + 1, mode=m, trial=d["trial"],
                       cfg=str(d["cfg"]), q=d["q"], **d["metrics"]) for r in results
                  for m, d in r["modes"].items()]).to_csv(os.path.join(OUT, "selected_configs_and_fold_metrics.csv"),
                                                          index=False)
    verdicts, tables = build_report(results, folds, y, sy)
    for h in HORIZONS:
        for mode in ("default", "tuned"):
            fig_bars(results, folds, y, h, mode)
            fig_curves(results, y, h, mode)
            fig_confusion(results, y, h, mode)
    if len(HORIZONS) > 1:
        fig_horizons(results, folds, y)
    for (h, mode), tab in tables.items():
        print("\n" + "=" * 140 + f"\n{HORIZON_SPEC[h]['label']} / {mode}\n" + "=" * 140)
        print(tab.drop(columns=["Category"]).to_string(index=False))
    txt = "\n".join(["SPANISH DROPOUT STUDY - VERDICTS (computed)", "=" * 70, *verdicts, "",
                     "5 folds => low power; 'tied' means no detectable difference. Dropout label A is inferred."])
    print("\n" + txt)
    open(os.path.join(OUT, "verdicts.txt"), "w").write(txt)
    print(f"\nAll outputs in {OUT}/ (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
