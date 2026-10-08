#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COURSE-LEVEL MODEL on the Spanish dropout data, tested against the current after-semester-1 model (H2).

The raw files hold one row per student x degree x COURSE (asi_hash) with the course grade and the course's own
monthly learning-platform activity. The main study collapsed these rows to totals (number of courses, mean grade,
credits passed, number of active courses). Here every course becomes a token and a third network branch reads the
tokens, so the question "does course identity / course-level performance add anything?" can be tested directly.

Token of one enrolled course (all information available after semester 1):
  * course id            embedding learned from training students only (rare ids -> UNK)
  * inferred semester    S1 / S2 / mixed / unknown, from WHEN the course's learning-platform activity happens
                         (median share of the course's events in Sep-Jan over TRAINING students; no outcome is used)
  * activity             4 platform metrics x 5 months (Sep-Jan), log1p, standardised on training tokens
  * grade                the final course grade, shown ONLY for courses inferred to be semester 1 (a semester-2 grade
                         is not known after semester 1); flags: pass/fail, grade recorded, grade missing
Variants (all: same 5 student-disjoint folds as the main study, 3 seeds, default hyper-parameters, 15 epochs):
  current          static MLP + causal-TCN/LSTM over the monthly platform totals (the model of the main study)
  courses_set      current + course branch (token MLP, masked mean+max pooling over the student's courses)
  courses_lstm     current + course branch (courses ordered by first activity month, LSTM)
  static_courses   static MLP + course branch (no monthly-totals temporal branch)
  courses_noid / _nograde / _noact / _idonly / _gradeonly   courses_set with parts of the token removed
  hgb_current, rf_current, hgb_courses, rf_courses   tree models without / with hand-made course features
Paired t-test over the 5 folds, Holm-corrected over the planned contrasts.

Run:  ENTRANTS_ONLY=1 OUT_DIR=spain_entrants_courses python spain_course_model.py     (QUICK_TEST=1 for a smoke test)
Env:  DATA_DIR / SPAIN_7Z / ENTRANTS_ONLY (see spain_dropout_experiments.py), N_SEEDS (3), N_JOBS, MIN_COUNT (10),
      K_MAX (24), EPOCHS (15)
"""
import os
import sys

os.environ.setdefault("EXTRA_HYBRIDS", "1")
os.environ["OUT_DIR"] = os.environ.get("OUT_DIR", "spain_entrants_courses")
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
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

import objective1_experiments as base
import objective1_ablation as AB
import objective1_track_head_and_tuning as tune
import spain_dropout_experiments as S

warnings.filterwarnings("ignore")
QUICK = base.QUICK_TEST
H = "H2"
SEEDS = [base.SEED + i for i in range(int(os.environ.get("N_SEEDS", 1 if QUICK else 3)))]
N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 2))
EPOCHS = 2 if QUICK else int(os.environ.get("EPOCHS", 15))
BATCH = 512
K_MAX = int(os.environ.get("K_MAX", 24))
MIN_COUNT = int(os.environ.get("MIN_COUNT", 10))
EMB_DIM = 16
N_MONTHS_TOK = 5                                  # Sep-Jan = the semester-1 window used at H2
OUT = base.OUT_DIR
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)
MET = S.LMS_METRICS                               # pft_events, pft_total_minutes, resource_events, n_resource_days
SEM_NAMES = {0: "unknown", 1: "S1", 2: "S2", 3: "mixed"}

# key, label, kind, spec
VARIANTS = [
    ("current", "Current model (static + monthly totals)", "deep", dict(course=False)),
    ("courses_set", "+ course branch (set pooling)", "deep", dict(course=True, ckind="set")),
    ("courses_lstm", "+ course branch (ordered, LSTM)", "deep", dict(course=True, ckind="lstm")),
    ("static_courses", "static + course branch (no monthly totals)", "deep",
     dict(course=True, ckind="set", temporal=False)),
    ("courses_noid", "course branch without course ids", "deep", dict(course=True, ckind="set", ids=False)),
    ("courses_nograde", "course branch without grades", "deep", dict(course=True, ckind="set", grade=False)),
    ("courses_noact", "course branch without course activity", "deep", dict(course=True, ckind="set", act=False)),
    ("courses_idonly", "course branch: ids and semester only", "deep",
     dict(course=True, ckind="set", grade=False, act=False)),
    ("courses_gradeonly", "course branch: grades and semester only", "deep",
     dict(course=True, ckind="set", ids=False, act=False)),
    ("hgb_current", "Gradient boosting, no course features", "tree", dict(model="hgb", course=False)),
    ("hgb_courses", "Gradient boosting + course features", "tree", dict(model="hgb", course=True)),
    ("rf_current", "Random Forest, no course features", "tree", dict(model="rf", course=False)),
    ("rf_courses", "Random Forest + course features", "tree", dict(model="rf", course=True)),
]
VKEYS = [v[0] for v in VARIANTS]
VLABEL = {v[0]: v[1] for v in VARIANTS}
VKIND = {v[0]: v[2] for v in VARIANTS}
VSPEC = {v[0]: v[3] for v in VARIANTS}
# planned contrasts (a - b), Holm-corrected together
CONTRASTS = [
    ("courses_set", "current", "course branch (set) vs current"),
    ("courses_lstm", "current", "course branch (ordered LSTM) vs current"),
    ("static_courses", "current", "course branch instead of monthly totals vs current"),
    ("courses_idonly", "current", "which courses only (ids) vs current"),
    ("courses_gradeonly", "current", "course grades only vs current"),
    ("hgb_courses", "hgb_current", "gradient boosting: course features vs none"),
    ("rf_courses", "rf_current", "random forest: course features vs none"),
    ("courses_set", "courses_noid", "identity effect: with vs without course ids"),
    ("courses_set", "courses_nograde", "grade effect: with vs without course grades"),
    ("courses_set", "courses_noact", "activity effect: with vs without course activity"),
]


# =============================================================================
# 1. COURSE TOKENS (one row of the raw files = one course of one student-year)
# =============================================================================
def build_course_data(sy):
    path = os.path.join(CACHE, f"courses_k{K_MAX}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    N = len(sy)
    keys = (sy["group"].astype(str) + "|" + sy["tit_hash"].astype(str) + "|" + sy["year_num"].astype(int).astype(str))
    pos = pd.Series(np.arange(N), index=keys.values)
    pos = pos[~pos.index.duplicated()]
    cid = np.zeros((N, K_MAX), np.int32)
    grade = np.full((N, K_MAX), np.nan, np.float32)
    act = np.zeros((N, K_MAX, N_MONTHS_TOK, len(MET)), np.float32)
    tot = np.zeros((N, K_MAX), np.float32)
    first5 = np.zeros((N, K_MAX), np.float32)
    mask = np.zeros((N, K_MAX), bool)
    code, found, rows_all, dropped, same_act = {}, 0, 0, 0, []
    folder = S.find_csvs()
    for Y in S.YEARS:
        p = os.path.join(folder, f"dataset_{Y}_hash.csv")
        hdr = pd.read_csv(p, sep=";", nrows=0).columns
        months = [(Y, m) for m in (9, 10, 11, 12)] + [(Y + 1, m) for m in range(1, 9)]
        cols = {(mi, mt): f"{met}_{yy}_{mm}" for mi, (yy, mm) in enumerate(months) for mt, met in enumerate(MET)
                if f"{met}_{yy}_{mm}" in hdr}
        use = ["dni_hash", "tit_hash", "asi_hash", "nota_asig_hash"] + list(cols.values())
        d = pd.read_csv(p, sep=";", decimal=",", usecols=use, low_memory=False,
                        dtype={"dni_hash": str, "tit_hash": str, "asi_hash": str})
        d["pos"] = (d["dni_hash"] + "|" + d["tit_hash"] + "|" + str(Y)).map(pos)
        d = d[d["pos"].notna()].reset_index(drop=True)
        rows_all += len(d)
        for a in pd.unique(d["asi_hash"]):
            if a not in code:
                code[a] = len(code) + 1
        A = np.zeros((len(d), 12, len(MET)), np.float32)
        for (mi, mt), c in cols.items():
            A[:, mi, mt] = np.clip(pd.to_numeric(d[c], errors="coerce").fillna(0.0).values, 0, None)
        ev = A[:, :, 0]
        onset = np.where((ev[:, :N_MONTHS_TOK] > 0).any(axis=1), (ev[:, :N_MONTHS_TOK] > 0).argmax(axis=1), N_MONTHS_TOK)
        t5 = ev[:, :N_MONTHS_TOK].sum(axis=1)
        pp = d["pos"].astype(int).values
        cc = d["asi_hash"].map(code).values
        order = np.lexsort((cc, -t5, onset, pp))              # per student-year: first activity first
        pp, cc = pp[order], cc[order]
        slot = pd.Series(pp).groupby(pp).cumcount().values
        keep = slot < K_MAX
        dropped += int((~keep).sum())
        r, s = order[keep], slot[keep]
        P = pp[keep]
        cid[P, s] = cc[keep]
        grade[P, s] = pd.to_numeric(d["nota_asig_hash"], errors="coerce").values[r]
        act[P, s] = np.log1p(A[r][:, :N_MONTHS_TOK, :])
        tot[P, s] = ev[r].sum(axis=1)
        first5[P, s] = t5[r]
        mask[P, s] = True
        found += len(np.unique(pp))
        g = pd.DataFrame({"p": pp, "v": A[order][:, :, 0].sum(axis=1)})
        g = g[g.v > 0].groupby("p").v.agg(["nunique", "size"])
        same_act.append(float(((g["nunique"] == 1) & (g["size"] > 1)).sum() / max((g["size"] > 1).sum(), 1)))
    out = dict(cid=cid, grade=grade, act=act, tot=tot, first5=first5, mask=mask, n_codes=len(code),
               info=dict(records=N, records_with_courses=int(mask.any(axis=1).sum()), course_rows=rows_all,
                         dropped_by_cap=dropped, k_max=K_MAX, courses_per_record=float(mask.sum(axis=1).mean()),
                         share_records_identical_activity_across_courses=float(np.mean(same_act))))
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    return out


def course_semester(D, tr):
    """S1/S2/mixed/unknown per course code, from the TRAINING students' activity timing only."""
    m = D["mask"][tr] & (D["tot"][tr] >= 10)
    share = (D["first5"][tr] / np.maximum(D["tot"][tr], 1))[m]
    g = pd.DataFrame({"c": D["cid"][tr][m], "s": share}).groupby("c").s.agg(["median", "size"])
    g = g[g["size"] >= 5]
    lab = np.zeros(D["n_codes"] + 1, np.int64)
    lab[g.index[g["median"] >= 0.75]] = 1
    lab[g.index[g["median"] <= 0.25]] = 2
    lab[g.index[(g["median"] > 0.25) & (g["median"] < 0.75)]] = 3
    return lab


def token_arrays(D, idx, lab, vocab):
    cid, mask = D["cid"][idx], D["mask"][idx]
    ids = vocab[cid]
    sem = np.where(mask, lab[cid], 0)
    act = D["act"][idx]
    gr = D["grade"][idx]
    s1 = (sem == 1) & mask
    has_g = s1 & ~np.isnan(gr)
    gfeat = np.stack([np.where(has_g, np.nan_to_num(gr) / 10.0, 0.0), (has_g & (gr < 5)).astype(np.float32),
                      has_g.astype(np.float32), (s1 & np.isnan(gr)).astype(np.float32)], axis=-1)
    has_act = (act[..., 0].sum(-1) > 0).astype(np.float32)[..., None]
    return dict(ids=ids, sem=sem, mask=mask, act=act.reshape(*cid.shape, -1), gfeat=gfeat.astype(np.float32),
                has_act=has_act, raw_grade=gr, s1=s1, has_g=has_g)


def course_fold(D, tr, va):
    lab = course_semester(D, tr)
    cnt = np.bincount(D["cid"][tr][D["mask"][tr]], minlength=D["n_codes"] + 1)
    keep = np.where(cnt >= MIN_COUNT)[0]
    vocab = np.zeros(D["n_codes"] + 1, np.int64)
    vocab[keep] = np.arange(1, len(keep) + 1)
    T_tr, T_va = token_arrays(D, tr, lab, vocab), token_arrays(D, va, lab, vocab)
    m = T_tr["mask"]
    mu = T_tr["act"][m].mean(axis=0)
    sd = T_tr["act"][m].std(axis=0) + 1e-6
    for T in (T_tr, T_va):
        T["act"] = (((T["act"] - mu) / sd) * T["mask"][..., None]).astype(np.float32)
    cov = {SEM_NAMES[k]: float(((T_tr["sem"] == k) & m).sum() / m.sum()) for k in range(4)}
    return T_tr, T_va, len(keep), cov


def course_summary(T, D_top):
    """Hand-made course features for the tree models (same information as the token branch)."""
    m, s1, hg = T["mask"], T["s1"], T["has_g"]
    gr = T["raw_grade"]
    failed = hg & (gr < 5)
    n_s1 = s1.sum(1)
    f = np.column_stack([
        m.sum(1), n_s1, (T["sem"] == 2).sum(1), (T["sem"] == 3).sum(1), (T["sem"] == 0).sum(1) - (~m).sum(1),
        hg.sum(1), failed.sum(1), failed.sum(1) / np.maximum(hg.sum(1), 1),
        np.where(hg.any(1), np.where(hg, gr, np.inf).min(1), np.nan), np.where(hg.any(1), np.nansum(np.where(hg, gr, 0), 1) / np.maximum(hg.sum(1), 1), np.nan),
        (s1 & np.isnan(gr)).sum(1), (s1 & (T["has_act"][..., 0] == 0)).sum(1), (m & (T["has_act"][..., 0] > 0)).sum(1)])
    f[~np.isfinite(f)] = np.nan
    # multi-hot "failed this course" flags for the courses failed most often in training
    flags = np.zeros((m.shape[0], len(D_top)), np.float32)
    for j, c in enumerate(D_top):
        flags[:, j] = ((T["cid_raw"] == c) & failed).any(1)
    return np.hstack([np.nan_to_num(f, nan=-1.0), flags]).astype(np.float32)


# =============================================================================
# 2. NETWORK
# =============================================================================
class CourseBranch(nn.Module):
    def __init__(self, V, f_tok, kind, hid=64, dropout=0.2):
        super().__init__()
        self.kind = kind
        self.emb_id = nn.Embedding(V + 1, EMB_DIM)
        self.emb_sem = nn.Embedding(4, 4)
        self.tok = nn.Sequential(nn.Linear(EMB_DIM + 4 + f_tok, hid), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hid, hid), nn.ReLU())
        if kind == "set":
            self.out = nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU())
        else:
            self.rnn = nn.LSTM(hid, hid, batch_first=True)

    def forward(self, ids, sem, feats, mask):
        h = self.tok(torch.cat([self.emb_id(ids), self.emb_sem(sem), feats], dim=-1))
        mk = mask.unsqueeze(-1)
        if self.kind == "set":
            mean = (h * mk).sum(1) / mk.sum(1).clamp(min=1)
            mx = h.masked_fill(~mk, -1e9).max(1).values
            mx = torch.where(mask.any(1, keepdim=True), mx, torch.zeros_like(mx))
            return self.out(torch.cat([mean, mx], dim=1))
        lengths = mask.sum(1).clamp(min=1).cpu()
        _, (hn, _) = self.rnn(pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False))
        return hn[-1]


class CourseNet(nn.Module):
    def __init__(self, d_static, f_seq, V, f_tok, course=False, ckind="set", temporal=True, dropout=0.2):
        super().__init__()
        self.temporal_on, self.course_on = temporal, course
        self.static = nn.Sequential(nn.Linear(d_static, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
                                    nn.Linear(128, 64), nn.ReLU())
        dim = 64
        if temporal:
            self.temporal = AB.TemporalEnc(f_seq, "tcn_lstm")
            dim += 64
        if course:
            self.course = CourseBranch(V, f_tok, ckind, dropout=dropout)
            dim += 64
        self.head = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2))

    def forward(self, xs, xt, ids, sem, ft, mask):
        z = [self.static(xs)]
        if self.temporal_on:
            z.append(self.temporal(xt))
        if self.course_on:
            z.append(self.course(ids, sem, ft, mask))
        return self.head(torch.cat(z, dim=1))


def to_tensors(T, spec):
    ids = T["ids"] if spec.get("ids", True) else np.zeros_like(T["ids"])
    parts = []
    parts.append(T["act"] if spec.get("act", True) else np.zeros_like(T["act"]))
    parts.append(T["gfeat"] if spec.get("grade", True) else np.zeros_like(T["gfeat"]))
    parts.append(T["has_act"] if spec.get("act", True) else np.zeros_like(T["has_act"]))
    ft = np.concatenate(parts, axis=-1).astype(np.float32)
    return (torch.tensor(ids, dtype=torch.long), torch.tensor(T["sem"], dtype=torch.long), torch.tensor(ft),
            torch.tensor(T["mask"]))


def fit_predict_deep(spec, d, T_tr, T_va, V, y_tr, seed):
    base.set_seed(seed)
    xs_tr, xs_va = torch.tensor(d["xs_tr"]), torch.tensor(d["xs_va"])
    xt_tr, xt_va = torch.tensor(d["seq_tr"]), torch.tensor(d["seq_va"])
    c_tr, c_va = to_tensors(T_tr, spec), to_tensors(T_va, spec)
    model = CourseNet(xs_tr.shape[1], xt_tr.shape[2], V, c_tr[2].shape[-1], course=spec.get("course", False),
                      ckind=spec.get("ckind", "set"), temporal=spec.get("temporal", True))
    ytr = torch.tensor(y_tr, dtype=torch.long)
    counts = np.bincount(y_tr, minlength=2).astype(np.float64)
    cw = len(y_tr) / (2 * np.maximum(counts, 1))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(seed)
    n = len(ytr)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            out = model(xs_tr[idx], xt_tr[idx], *[t[idx] for t in c_tr])
            loss_fn(out, ytr[idx]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), base.GRAD_CLIP)
            opt.step()
    model.eval()
    with torch.no_grad():
        return F.softmax(model(xs_va, xt_va, *c_va), dim=1).numpy()[:, 1]


# =============================================================================
# 3. ONE (variant, fold) JOB
# =============================================================================
def run_job(vkey, fold, tr, va):
    torch.set_num_threads(1)
    path = os.path.join(CACHE, f"{H}_{vkey}_f{fold}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    sy, seq_all = S.build_student_year()
    seq, _ = S.horizon_sequences(H, seq_all, sy)
    y = sy["y"].values
    D = build_course_data(sy)
    d = S.fold_data(H, tr, va, sy, seq)
    T_tr, T_va, V, cov = course_fold(D, tr, va)
    spec = VSPEC[vkey]
    rows = []
    if VKIND[vkey] == "deep":
        for s in SEEDS:
            p = fit_predict_deep(spec, d, T_tr, T_va, V, y[tr], s + fold)
            rows.append(p)
    else:
        flat_tr, flat_va = d["flat_tr"], d["flat_va"]
        if spec["course"]:
            for T, idx in ((T_tr, tr), (T_va, va)):
                T["cid_raw"] = D["cid"][idx]
            failed_tr = T_tr["s1"] & T_tr["has_g"] & (T_tr["raw_grade"] < 5)
            cnt = np.bincount(T_tr["cid_raw"][failed_tr], minlength=D["n_codes"] + 1)
            top = np.where(cnt >= 15)[0][np.argsort(-cnt[cnt >= 15])][:150]
            flat_tr = np.hstack([flat_tr, course_summary(T_tr, top)])
            flat_va = np.hstack([flat_va, course_summary(T_va, top)])
        dd = dict(flat_tr=flat_tr, flat_va=flat_va)
        cfg = tune.finalize_cfg(spec["model"], tune.default_cfg(spec["model"]))
        for s in SEEDS:
            rows.append(tune.fit_predict(spec["model"], cfg, dd, y[tr], 2, s + fold)[:, 1])
    out = []
    for s, p in zip(SEEDS, rows):
        top10 = p >= np.quantile(p, 0.90)
        out.append(dict(variant=vkey, fold=fold + 1, seed=s, pr_auc=average_precision_score(y[va], p),
                        roc_auc=roc_auc_score(y[va], p), capture_top10=float(y[va][top10].sum() / y[va].sum()),
                        n_vocab=V, **{f"sem_{k}": v for k, v in cov.items()}))
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    return out


# =============================================================================
# 4. ANALYSIS
# =============================================================================
def analyse(runs):
    fold_df = runs.groupby(["variant", "fold"]).mean(numeric_only=True).reset_index()
    vec = lambda v, m="pr_auc": fold_df[fold_df.variant == v].sort_values("fold")[m].values
    rows = []
    for v in VKEYS:
        r = dict(variant=v, label=VLABEL[v])
        for m in ("pr_auc", "roc_auc", "capture_top10"):
            a = vec(v, m)
            r[f"{m}_mean"], r[f"{m}_std"] = a.mean(), a.std(ddof=1)
        rows.append(r)
    t = pd.DataFrame(rows)
    ct = []
    for a, b, desc in CONTRASTS:
        diff = vec(a) - vec(b)
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        tc = stats.t.ppf(0.975, len(diff) - 1)
        ct.append(dict(contrast=desc, a=a, b=b, delta_pr_auc=diff.mean(), ci_lo=diff.mean() - tc * se,
                       ci_hi=diff.mean() + tc * se, delta_roc_auc=(vec(a, "roc_auc") - vec(b, "roc_auc")).mean(),
                       delta_capture=(vec(a, "capture_top10") - vec(b, "capture_top10")).mean(),
                       folds_a_better=int((diff > 0).sum()), p=base.paired_p(vec(a), vec(b))))
    ct = pd.DataFrame(ct)
    ct["p_holm"] = base.holm_adjust(ct["p"].values)
    return t, ct, fold_df


def plot(t, ct):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), gridspec_kw={"width_ratios": [1.1, 1]})
    d = t.set_index("variant").loc[VKEYS]
    y = np.arange(len(d))[::-1]
    cols = ["#d62728" if k == "current" else ("#4c78a8" if VKIND[k] == "deep" else "#f58518") for k in d.index]
    axes[0].barh(y, d.pr_auc_mean, xerr=d.pr_auc_std, color=cols, alpha=0.85, capsize=2)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(d.label, fontsize=8)
    axes[0].set_xlim(max(0, d.pr_auc_mean.min() - 0.08), d.pr_auc_mean.max() + 0.05)
    axes[0].set_xlabel("PR-AUC, mean ± std over 5 folds")
    axes[0].set_title("Variants (red = current model, blue = deep, orange = trees)")
    c = ct.iloc[::-1]
    yy = np.arange(len(c))
    axes[1].barh(yy, c.delta_pr_auc, color="#4c78a8", alpha=0.85)
    axes[1].errorbar(c.delta_pr_auc, yy, xerr=[c.delta_pr_auc - c.ci_lo, c.ci_hi - c.delta_pr_auc], fmt="none",
                     ecolor="k", capsize=3)
    for yi, (_, r) in zip(yy, c.iterrows()):
        star = "**" if r.p_holm < 0.05 else ("*" if r.p < 0.05 else "")
        axes[1].text(r.ci_hi if r.delta_pr_auc >= 0 else r.ci_lo, yi, f" {star}", va="center",
                     ha="left" if r.delta_pr_auc >= 0 else "right")
    axes[1].axvline(0, color="k", lw=1)
    axes[1].set_yticks(yy)
    axes[1].set_yticklabels(c.contrast, fontsize=8)
    axes[1].set_xlabel("change in PR-AUC (first minus second); 95% CI over folds; * p<0.05, ** Holm p<0.05")
    axes[1].set_title("Planned contrasts")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Course-level model, {S.HORIZON_SPEC[H]['label']}" + (" - new entrants" if S.ENTRANTS_ONLY else ""))
    fig.tight_layout()
    return base.save_fig(fig, "fig_course_model_H2.png")


def main():
    t0 = time.time()
    sy, _ = S.build_student_year()
    y = sy["y"].values
    D = build_course_data(sy)
    info = D["info"]
    print(f"[courses] {len(sy)} records, {sy['group'].nunique()} students, dropout {100 * y.mean():.2f}%, "
          f"variants={len(VARIANTS)} seeds={SEEDS} epochs={EPOCHS} jobs={N_JOBS} min_count={MIN_COUNT}")
    print(f"[courses] {info}")
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=base.SEED).split(np.zeros(len(sy)), y,
                                                                                    sy["group"].values))
    for tr, va in folds:
        assert not set(sy["group"].values[tr]) & set(sy["group"].values[va]), "student overlap between folds"
    res = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(run_job)(v, f, tr, va) for v in VKEYS
                                             for f, (tr, va) in enumerate(folds))
    runs = pd.DataFrame([r for rows in res for r in rows])
    runs.to_csv(os.path.join(OUT, "course_model_H2_all_runs.csv"), index=False)
    t, ct, fold_df = analyse(runs)
    t.to_csv(os.path.join(OUT, "course_model_H2_variants.csv"), index=False)
    ct.to_csv(os.path.join(OUT, "course_model_H2_contrasts.csv"), index=False)
    cov = runs[[c for c in runs.columns if c.startswith("sem_")]].mean()
    print("\nmean share of training course tokens by inferred semester:", cov.round(3).to_dict())
    print("\n", t[["label", "pr_auc_mean", "pr_auc_std", "roc_auc_mean", "capture_top10_mean"]].round(4).to_string(index=False))
    print("\n", ct[["contrast", "delta_pr_auc", "ci_lo", "ci_hi", "folds_a_better", "p", "p_holm"]].round(4).to_string(index=False))
    plot(t, ct)
    prev = float(y.mean())
    sig = ct[(ct.p_holm < 0.05)]
    txt = (f"COURSE-LEVEL MODEL {S.HORIZON_SPEC[H]['label']}" + (" (new entrants)" if S.ENTRANTS_ONLY else "") +
           f"\nprevalence {prev:.3f}; current model PR-AUC {t.loc[t.variant == 'current', 'pr_auc_mean'].iloc[0]:.3f}; "
           f"course branch (set) {t.loc[t.variant == 'courses_set', 'pr_auc_mean'].iloc[0]:.3f}\n"
           f"Contrasts with Holm p<0.05: {list(sig.contrast) or 'none'}\n" + f"data: {info}")
    print("\n" + txt)
    open(os.path.join(OUT, "course_model_H2_report.txt"), "w").write(txt)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
