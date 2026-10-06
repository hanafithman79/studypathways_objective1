#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
OBJECTIVE 1 - Dual-Branch Hybrid Deep Learning for Study-Pathway Prediction
===============================================================================
Self-contained experiment script (Google Colab / local, CPU or GPU).

Predicts post-secondary study pathways by fusing static socioeconomic features
with a short academic-trajectory sequence.

Targets
  * MACRO_TRACK       : 4 broad tracks mapped from the 21 granular majors
  * ACADEMIC_PROGRAM  : 21 granular engineering programmes (Top-1, Top-3, Macro-F1)

Models (10 in total: 3 classical + 2 single-branch deep + 4 comparative hybrids
        + 1 proposed dual-branch hybrid)

Protocol: StratifiedKFold(5, shuffle, seed=42) on ACADEMIC_PROGRAM; all
scaling / encoding is fitted on the training part of each fold only; paired
t-tests (fold-wise Macro-F1) of the Proposed model against every other model.

Usage (Colab):  upload this file, run `!python objective1_experiments.py`
                (dataset.csv is looked up in the working directory, then
                downloaded from GitHub, then - in Colab - requested as upload).
Quick check:    QUICK_TEST=1 python objective1_experiments.py
Outputs:        ./objective1_outputs/  (+ ./objective1_rigorous_results.csv)
===============================================================================
"""

import os
import sys
import json
import time
import random
import warnings
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, f1_score, top_k_accuracy_score,
                             balanced_accuracy_score, confusion_matrix,
                             classification_report)

warnings.filterwarnings("ignore")

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
QUICK_TEST = os.environ.get("QUICK_TEST", "0") == "1"   # tiny run to smoke-test the pipeline

DATA_PATH = "dataset.csv"
DATA_URL = "https://raw.githubusercontent.com/hanafithman79/dataset/main/dataset.csv"
OUT_DIR = os.environ.get("OUT_DIR", "objective1_outputs")
RESULTS_CSV = os.environ.get("RESULTS_CSV", "objective1_rigorous_results.csv")

SEED = 42
N_SPLITS = 5
EPOCHS = 2 if QUICK_TEST else 15
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 5.0
RF_TREES = 20 if QUICK_TEST else 100
HGB_ITER = 20 if QUICK_TEST else 100

# The specification lists G_SC / PERCENTILE / 2ND_DECILE / QUARTILE as part of the
# temporal sequence. These are Saber-Pro (end-of-degree) quantities; set this to
# False for a leakage-safe ablation that uses only Saber-11 information.
USE_SABER_PRO_FEATURES = os.environ.get("USE_SABER_PRO_FEATURES", "1") == "1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from IPython import get_ipython
    IN_NOTEBOOK = get_ipython() is not None
except Exception:
    IN_NOTEBOOK = False
IN_COLAB = "google.colab" in sys.modules

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "confusion_matrices"), exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

# =============================================================================
# 1. DATA LOADING, TARGET MAPPING, FEATURE ENGINEERING
# =============================================================================
TRACK_MAP = {
    "Industrial & Management": [
        "INDUSTRIAL ENGINEERING", "PRODUCTION ENGINEERING",
        "PRODUCTIVITY AND QUALITY ENGINEERING"],
    "Civil & Infrastructure": [
        "CIVIL ENGINEERING", "CATASTRAL ENGINEERING AND GEODESY",
        "TOPOGRAPHIC ENGINEERY", "CIVIL CONSTRUCTIONS",
        "TRANSPORTATION AND ROAD ENGINEERING"],
    "Mechanical, Electrical & Tech": [
        "MECHANICAL ENGINEERING", "ELECTRONIC ENGINEERING", "ELECTRIC ENGINEERING",
        "MECHATRONICS ENGINEERING", "ELECTRIC ENGINEERING AND TELECOMMUNICATIONS",
        "AERONAUTICAL ENGINEERING", "ELECTROMECHANICAL ENGINEERING",
        "INDUSTRIAL AUTOMATIC ENGINEERING", "CONTROL ENGINEERING",
        "AUTOMATION ENGINEERING", "INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING"],
    "Chemical & Process": ["CHEMICAL ENGINEERING", "TEXTILE ENGINEERING"],
}
PROGRAM_TO_TRACK = {p: t for t, ps in TRACK_MAP.items() for p in ps}
TRACK_NAMES = list(TRACK_MAP.keys())

STATIC_COLS = ["GENDER", "STRATUM", "SISBEN", "SCHOOL_TYPE", "SCHOOL_NAT", "EDU_FATHER",
               "EDU_MOTHER", "OCC_FATHER", "OCC_MOTHER", "REVENUE", "PEOPLE_HOUSE",
               "INTERNET", "COMPUTER", "CAR"]

# FEATURE_SET: "spec" (default, the original 14 static variables) | "extended" (+ extra household
# assets / job + school name, target-encoded) | "extended_univ" (extended + UNIVERSITY, which is
# chosen together with the programme -> NOT a pre-decision predictor; use only as a labelled experiment)
FEATURE_SET = os.environ.get("FEATURE_SET", "spec")
TE_COLS = []                      # high-cardinality columns -> fold-wise target encoding
if FEATURE_SET in ("extended", "extended_univ"):
    STATIC_COLS = STATIC_COLS + ["TV", "WASHING_MCH", "MIC_OVEN", "DVD", "FRESH", "PHONE", "MOBILE", "JOB"]
    TE_COLS = ["SCHOOL_NAME"]
if FEATURE_SET == "extended_univ":
    TE_COLS = TE_COLS + ["UNIVERSITY"]

# The 12 temporal variables are arranged as a 3-step trajectory (4 features/step):
#   step 1: Saber-11 subject scores        step 2: Saber-11 English + STEM/HUM profile
#   step 3: aggregate standing (global score, percentile, decile, quartile)
TEMPORAL_STEPS = [
    ["MAT_S11", "CR_S11", "CC_S11", "BIO_S11"],
    ["ENG_S11", "STEM_AVG", "HUM_AVG", "STEM_HUM_RATIO"],
    ["G_SC", "PERCENTILE", "2ND_DECILE", "QUARTILE"],
]
if not USE_SABER_PRO_FEATURES:
    TEMPORAL_STEPS = TEMPORAL_STEPS[:2]
TEMPORAL_COLS = [c for step in TEMPORAL_STEPS for c in step]
N_STEPS = len(TEMPORAL_STEPS)
N_FEATS_STEP = len(TEMPORAL_STEPS[0])


def read_csv_robust(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def load_dataframe():
    if not os.path.exists(DATA_PATH):
        try:
            print(f"[data] {DATA_PATH} not found - downloading from {DATA_URL}")
            urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        except Exception as e:
            if IN_COLAB:
                print(f"[data] download failed ({e}); please upload dataset.csv")
                from google.colab import files
                up = files.upload()
                name = list(up.keys())[0]
                os.replace(name, DATA_PATH)
            else:
                raise FileNotFoundError(
                    "dataset.csv not found and could not be downloaded.") from e
    df = read_csv_robust(DATA_PATH)
    df.columns = df.columns.astype(str).str.strip()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    return df


def prepare_data():
    df = load_dataframe()
    print(f"[data] loaded {df.shape[0]} rows x {df.shape[1]} columns")
    df = df.dropna(subset=["ACADEMIC_PROGRAM"]).copy()
    df["ACADEMIC_PROGRAM"] = (df["ACADEMIC_PROGRAM"].astype(str).str.upper()
                              .str.replace(r"\s+", " ", regex=True).str.strip())
    unmapped = sorted(set(df["ACADEMIC_PROGRAM"]) - set(PROGRAM_TO_TRACK))
    if unmapped:
        raise ValueError(f"Programmes without a MACRO_TRACK mapping: {unmapped}")
    df["MACRO_TRACK"] = df["ACADEMIC_PROGRAM"].map(PROGRAM_TO_TRACK)
    # PROGRAM_MIN_COUNT>0: a DIFFERENT, clearly labelled task that keeps only programmes with enough students
    min_count = int(os.environ.get("PROGRAM_MIN_COUNT", "0"))
    if min_count > 0:
        vc = df["ACADEMIC_PROGRAM"].value_counts()
        keep = vc[vc >= min_count].index
        print(f"[subset] keeping {len(keep)} programmes with >= {min_count} students "
              f"({100 * vc[keep].sum() / len(df):.1f}% of students); dropped: {sorted(set(vc.index) - set(keep))}")
        df = df[df["ACADEMIC_PROGRAM"].isin(keep)].copy()

    # feature engineering
    df["STEM_AVG"] = (df["MAT_S11"] + df["BIO_S11"]) / 2.0
    df["HUM_AVG"] = (df["CR_S11"] + df["CC_S11"]) / 2.0
    df["STEM_HUM_RATIO"] = df["STEM_AVG"] / (df["HUM_AVG"] + 1e-5)

    df = df.reset_index(drop=True)
    for c in STATIC_COLS:                       # categorical gaps -> explicit level
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna("Missing")
    for c in TE_COLS:
        df[c] = df[c].fillna("Missing")
    for c in TEMPORAL_COLS:                     # numeric gaps (none expected) -> column median
        df[c] = df[c].fillna(df[c].median())

    programs = sorted(df["ACADEMIC_PROGRAM"].unique())
    prog_to_idx = {p: i for i, p in enumerate(programs)}
    y = df["ACADEMIC_PROGRAM"].map(prog_to_idx).values.astype(int)
    prog_track_idx = np.array([TRACK_NAMES.index(PROGRAM_TO_TRACK[p]) for p in programs])
    M = np.zeros((len(programs), len(TRACK_NAMES)), dtype=np.float32)   # program -> track
    M[np.arange(len(programs)), prog_track_idx] = 1.0
    return df, y, programs, prog_track_idx, M


# =============================================================================
# 2. FOLD-WISE PREPROCESSING (fit on train only -> no leakage)
# =============================================================================
def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:                       # older scikit-learn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def preprocess_fold(df_tr, df_va, y_tr=None):
    """Returns dict with flat / static / sequence arrays for train and validation."""
    cat_cols = [c for c in STATIC_COLS if not pd.api.types.is_numeric_dtype(df_tr[c])]
    num_cols = [c for c in STATIC_COLS if pd.api.types.is_numeric_dtype(df_tr[c])]
    parts = []
    if cat_cols:
        parts.append(("cat", make_onehot(), cat_cols))
    if num_cols:
        parts.append(("num", StandardScaler(), num_cols))
    static_ct = ColumnTransformer(parts)
    xs_tr = static_ct.fit_transform(df_tr[STATIC_COLS]).astype(np.float32)
    xs_va = static_ct.transform(df_va[STATIC_COLS]).astype(np.float32)
    if TE_COLS and y_tr is not None:            # leakage-safe: encoder fitted on the training part only
        from sklearn.preprocessing import TargetEncoder
        te = TargetEncoder(target_type="multiclass", random_state=0)
        te_tr = te.fit_transform(df_tr[TE_COLS], y_tr).astype(np.float32)
        te_va = te.transform(df_va[TE_COLS]).astype(np.float32)
        xs_tr, xs_va = np.hstack([xs_tr, te_tr]), np.hstack([xs_va, te_va])

    sc = StandardScaler()
    xt_tr = sc.fit_transform(df_tr[TEMPORAL_COLS]).astype(np.float32)
    xt_va = sc.transform(df_va[TEMPORAL_COLS]).astype(np.float32)
    seq_tr = xt_tr.reshape(-1, N_STEPS, N_FEATS_STEP)       # (B, T, F)
    seq_va = xt_va.reshape(-1, N_STEPS, N_FEATS_STEP)

    return dict(
        flat_tr=np.hstack([xs_tr, xt_tr]), flat_va=np.hstack([xs_va, xt_va]),
        xs_tr=xs_tr, xs_va=xs_va, seq_tr=seq_tr, seq_va=seq_va)


# =============================================================================
# 3. MODEL ARCHITECTURES
# =============================================================================
class StaticEncoder(nn.Module):
    """Dense(128) -> BN -> ReLU -> Dropout(0.2) -> Dense(64) -> ReLU."""

    def __init__(self, d_in, out_dim=64, hidden=128, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, out_dim), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class LSTMEncoder(nn.Module):                      # uni-directional LSTM, 64-dim
    def __init__(self, f_in, hidden=64):
        super().__init__()
        self.rnn = nn.LSTM(f_in, hidden, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.rnn(x)
        return h[-1]


class CNNLSTMEncoder(nn.Module):                   # standard Conv1D(32) -> uni-LSTM(64)
    def __init__(self, f_in, conv_ch=32, hidden=64):
        super().__init__()
        self.conv = nn.Conv1d(f_in, conv_ch, kernel_size=3, padding=1)
        self.rnn = nn.LSTM(conv_ch, hidden, batch_first=True)

    def forward(self, x):
        z = F.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        _, (h, _) = self.rnn(z)
        return h[-1]


class BiLSTMEncoder(nn.Module):                    # 2 x 32 = 64-dim
    def __init__(self, f_in, hidden=32):
        super().__init__()
        self.rnn = nn.LSTM(f_in, hidden, batch_first=True, bidirectional=True)

    def forward(self, x):
        _, (h, _) = self.rnn(x)
        return torch.cat([h[0], h[1]], dim=1)


class GRUEncoder(nn.Module):                       # GRU, 64-dim
    def __init__(self, f_in, hidden=64):
        super().__init__()
        self.rnn = nn.GRU(f_in, hidden, batch_first=True)

    def forward(self, x):
        _, h = self.rnn(x)
        return h[-1]


class TransformerTemporalEncoder(nn.Module):       # TransformerEncoderLayer(d_model=feats, nhead=1) -> 64
    def __init__(self, f_in, out_dim=64):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model=f_in, nhead=1, dim_feedforward=64,
                                                dropout=0.1, batch_first=True)
        self.proj = nn.Linear(f_in, out_dim)

    def forward(self, x):
        z = self.layer(x).mean(dim=1)
        return F.relu(self.proj(z))


class CausalConv1d(nn.Module):
    """Dilated 1-D convolution padded on the left only -> no information from the future."""

    def __init__(self, ch, kernel_size=2, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(ch, ch, kernel_size, dilation=dilation)

    def forward(self, x):                           # x: (B, C, T)
        return self.conv(F.pad(x, (self.pad, 0)))


class CausalTCNLSTMEncoder(nn.Module):             # causal dilated conv (+residual) -> uni-LSTM(64)
    def __init__(self, f_in, hidden=64, kernel_size=2, dilation=1):
        super().__init__()
        self.tcn = CausalConv1d(f_in, kernel_size, dilation)
        self.rnn = nn.LSTM(f_in, hidden, batch_first=True)

    def forward(self, x):                           # x: (B, T, F)
        c = x.transpose(1, 2)
        c = c + F.relu(self.tcn(c))                 # residual connection
        _, (h, _) = self.rnn(c.transpose(1, 2))
        return h[-1]


class FusionHead(nn.Module):                       # [z_s || z_t] -> Dense(64) -> ReLU -> Dropout -> logits
    def __init__(self, d_in, n_classes, p=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(64, n_classes))

    def forward(self, x):
        return self.net(x)


class StaticOnlyNet(nn.Module):
    def __init__(self, d_static, n_classes):
        super().__init__()
        self.enc = StaticEncoder(d_static)
        self.head = nn.Linear(64, n_classes)

    def forward(self, xs, xt):
        return self.head(self.enc(xs))


class TemporalOnlyNet(nn.Module):
    def __init__(self, f_in, n_classes):
        super().__init__()
        self.enc = LSTMEncoder(f_in)
        self.head = nn.Linear(64, n_classes)

    def forward(self, xs, xt):
        return self.head(self.enc(xt))


class DualBranchNet(nn.Module):
    """Static MLP branch + temporal branch, fused by concatenation."""

    def __init__(self, d_static, temporal_encoder, n_classes):
        super().__init__()
        self.static_enc = StaticEncoder(d_static)
        self.temporal_enc = temporal_encoder
        self.head = FusionHead(128, n_classes)

    def forward(self, xs, xt):
        return self.head(torch.cat([self.static_enc(xs), self.temporal_enc(xt)], dim=1))


# (key, display name, category)
MODEL_SPECS = [
    ("logreg", "Logistic Regression", "A: Traditional ML"),
    ("rf", "Random Forest", "A: Traditional ML"),
    ("hgb", "HistGradientBoosting", "A: Traditional ML"),
    ("static_mlp", "Static MLP", "B: Single-branch deep"),
    ("lstm", "Standalone LSTM", "B: Single-branch deep"),
    ("hyb1", "Hybrid-1 (MLP+CNN+LSTM)", "C: Hybrid baseline"),
    ("hyb2", "Hybrid-2 (MLP+BiLSTM)", "C: Hybrid baseline"),
    ("hyb3", "Hybrid-3 (MLP+GRU)", "C: Hybrid baseline"),
    ("hyb4", "Hybrid-4 (MLP+Transformer)", "C: Hybrid baseline"),
    ("proposed", "Proposed (MLP+CausalTCN+LSTM)", "D: Proposed"),
]
KEYS = [k for k, _, _ in MODEL_SPECS]
NAMES = {k: n for k, n, _ in MODEL_SPECS}
CATEGORY = {k: c for k, _, c in MODEL_SPECS}
DEEP_KEYS = ["static_mlp", "lstm", "hyb1", "hyb2", "hyb3", "hyb4", "proposed"]

# EXTRA_HYBRIDS=1 appends 3 more hybrid baselines (appended last, so the random-search draws of the
# existing models are unchanged and their cached results stay valid)
if os.environ.get("EXTRA_HYBRIDS", "0") == "1":
    for _k, _n in (("hyb5", "Hybrid-5 (MLP+CNN-BiLSTM)"), ("hyb6", "Hybrid-6 (MLP+Attention-LSTM)"),
                   ("hyb7", "Hybrid-7 (MLP+Transformer-LSTM)")):
        MODEL_SPECS.append((_k, _n, "C: Hybrid baseline"))
        KEYS.append(_k)
        NAMES[_k] = _n
        CATEGORY[_k] = "C: Hybrid baseline"
        DEEP_KEYS.append(_k)
PROPOSED = "proposed"


def build_torch_model(key, d_static, n_classes):
    f = N_FEATS_STEP
    if key == "static_mlp":
        return StaticOnlyNet(d_static, n_classes)
    if key == "lstm":
        return TemporalOnlyNet(f, n_classes)
    enc = {"hyb1": lambda: CNNLSTMEncoder(f), "hyb2": lambda: BiLSTMEncoder(f),
           "hyb3": lambda: GRUEncoder(f), "hyb4": lambda: TransformerTemporalEncoder(f),
           "proposed": lambda: CausalTCNLSTMEncoder(f)}[key]()
    return DualBranchNet(d_static, enc, n_classes)


# =============================================================================
# 4. TRAINING / INFERENCE
# =============================================================================
def align_proba(proba, classes, n_classes):
    """Scikit-learn models only know the classes present in the training fold."""
    out = np.zeros((proba.shape[0], n_classes), dtype=np.float32)
    out[:, np.asarray(classes, dtype=int)] = proba
    return out


def fit_predict_sklearn(key, data, y_tr, n_classes):
    if key == "logreg":
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(data["flat_tr"], y_tr)
    elif key == "rf":
        model = RandomForestClassifier(n_estimators=RF_TREES, class_weight="balanced",
                                       n_jobs=-1, random_state=SEED)
        model.fit(data["flat_tr"], y_tr)
    elif key == "hgb":
        model = HistGradientBoostingClassifier(max_iter=HGB_ITER, early_stopping=False,
                                               random_state=SEED)
        model.fit(data["flat_tr"], y_tr, sample_weight=compute_sample_weight("balanced", y_tr))
    else:
        raise KeyError(key)
    return align_proba(model.predict_proba(data["flat_va"]), model.classes_, n_classes)


def fit_predict_torch(key, data, y_tr, y_va, n_classes, seed):
    set_seed(seed)
    model = build_torch_model(key, data["xs_tr"].shape[1], n_classes).to(DEVICE)

    xs_tr = torch.tensor(data["xs_tr"], device=DEVICE)
    xt_tr = torch.tensor(data["seq_tr"], device=DEVICE)
    ytr = torch.tensor(y_tr, dtype=torch.long, device=DEVICE)
    xs_va = torch.tensor(data["xs_va"], device=DEVICE)
    xt_va = torch.tensor(data["seq_va"], device=DEVICE)
    yva = torch.tensor(y_va, dtype=torch.long, device=DEVICE)

    # inverse class weights  c_y = N / (K * N_y)   (0 for classes absent from the training fold)
    counts = np.bincount(y_tr, minlength=n_classes).astype(np.float64)
    cw = np.where(counts > 0, len(y_tr) / (n_classes * np.maximum(counts, 1)), 0.0)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    gen = torch.Generator().manual_seed(seed)
    n = len(ytr)
    history = []
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, generator=gen).to(DEVICE)
        run, seen = 0.0, 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            if len(idx) < 2:                     # BatchNorm needs >1 sample
                continue
            opt.zero_grad()
            loss = loss_fn(model(xs_tr[idx], xt_tr[idx]), ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            run += loss.item() * len(idx)
            seen += len(idx)
        model.eval()                             # validation loss: monitoring only (no model selection)
        with torch.no_grad():
            vloss = F.cross_entropy(model(xs_va, xt_va), yva).item()
        history.append((run / max(seen, 1), vloss))

    model.eval()
    with torch.no_grad():
        proba = F.softmax(model(xs_va, xt_va), dim=1).cpu().numpy().astype(np.float32)
    return proba, np.array(history)


# =============================================================================
# 5. METRICS & STATISTICS
# =============================================================================
def compute_metrics(y_true, proba, prog_track_idx, n_classes):
    pred = proba.argmax(1)
    true_track, pred_track = prog_track_idx[y_true], prog_track_idx[pred]
    # track probabilities = sum of the probabilities of the programmes in that track
    track_pred = (proba @ np.eye(len(TRACK_NAMES))[prog_track_idx]).argmax(1)
    return {
        "track_top1": 100 * accuracy_score(true_track, track_pred),
        "prog_top1": 100 * accuracy_score(y_true, pred),
        "prog_top3": 100 * top_k_accuracy_score(y_true, proba, k=3, labels=np.arange(n_classes)),
        "prog_macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "prog_bal_acc": 100 * balanced_accuracy_score(y_true, pred),
        "track_macro_f1": f1_score(true_track, track_pred, average="macro", zero_division=0),
    }, track_pred


def holm_adjust(p):
    p = np.nan_to_num(np.asarray(p, dtype=float), nan=1.0)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired_p(a, b):
    if np.allclose(a, b):
        return 1.0
    p = stats.ttest_rel(a, b).pvalue
    return 1.0 if np.isnan(p) else float(p)


# =============================================================================
# 6. PLOTTING HELPERS (matplotlib only)
# =============================================================================
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"A: Traditional ML": "#8c8c8c", "B: Single-branch deep": "#4c78a8",
          "C: Hybrid baseline": "#f58518", "D: Proposed": "#d62728"}


def save_fig(fig, name, subdir=None):
    path = os.path.join(OUT_DIR, subdir, name) if subdir else os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)
    return path


def draw_heatmap(ax, data, xlabels, ylabels, fmt="{:.2f}", cmap="viridis", vmin=None, vmax=None,
                 annotate=True, fontsize=7, rotation=45):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=rotation, ha="right" if rotation else "center")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.spines[:].set_visible(False)
    if annotate:
        lo = np.nanmin(data) if vmin is None else vmin
        hi = np.nanmax(data) if vmax is None else vmax
        mid = (lo + hi) / 2
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if np.isnan(v):
                    continue
                dark = (v < mid) if cmap in ("viridis", "magma") else (v > mid)
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fontsize,
                        color="white" if dark else "black")
    return im


def short(name, n=26):
    s = name.title()
    return s if len(s) <= n else s[:n - 1] + "…"


def plot_class_distribution(df, programs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [2.2, 1]})
    vc = df["ACADEMIC_PROGRAM"].value_counts().reindex(programs).sort_values()
    axes[0].barh([short(p, 40) for p in vc.index], vc.values, color="#4c78a8")
    axes[0].set_xscale("log")
    axes[0].set_title("ACADEMIC_PROGRAM class distribution (log scale)")
    axes[0].set_xlabel("students")
    for i, v in enumerate(vc.values):
        axes[0].text(v * 1.05, i, str(v), va="center", fontsize=7)
    vt = df["MACRO_TRACK"].value_counts().reindex(TRACK_NAMES)
    axes[1].bar(range(4), vt.values, color="#f58518")
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels([t.replace(", ", ",\n").replace(" & ", " &\n") for t in TRACK_NAMES], fontsize=7)
    axes[1].set_title("MACRO_TRACK class distribution")
    for i, v in enumerate(vt.values):
        axes[1].text(i, v, f"{v}\n({100 * v / len(df):.1f}%)", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    return save_fig(fig, "fig01_class_distribution.png")


def plot_metric_bars(fold_df):
    metrics = [("track_top1", "Macro-track Top-1 accuracy (%)"),
               ("prog_top1", "Programme Top-1 accuracy (%)"),
               ("prog_top3", "Programme Top-3 accuracy (%)"),
               ("prog_macro_f1", "Programme Macro-F1")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (m, title) in zip(axes.ravel(), metrics):
        g = fold_df.groupby("model")[m].agg(["mean", "std"]).reindex(KEYS)
        ax.barh([NAMES[k] for k in KEYS], g["mean"], xerr=g["std"], capsize=3,
                color=[COLORS[CATEGORY[k]] for k in KEYS])
        ax.invert_yaxis()
        ax.set_title(title)
        for i, (mu, sd) in enumerate(zip(g["mean"], g["std"])):
            ax.text(mu + sd + 0.01 * g["mean"].max(), i, f"{mu:.2f}" if m == "prog_macro_f1" else f"{mu:.1f}",
                    va="center", fontsize=7)
        if m == "track_top1":
            ax.axvline(80, color="k", ls="--", lw=1)
            ax.text(80, -0.9, "80% target", fontsize=7, ha="center")
        if m == "prog_top3":
            ax.axvline(85, color="k", ls="--", lw=1)
            ax.text(85, -0.9, "85% target", fontsize=7, ha="center")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
    fig.legend(handles, COLORS.keys(), loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Objective 1 - 5-fold CV performance (mean ± std)", y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    return save_fig(fig, "fig02_metric_bars.png")


def plot_fold_boxplot(fold_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (m, title) in zip(axes, [("prog_macro_f1", "Programme Macro-F1 per fold"),
                                      ("track_top1", "Macro-track Top-1 accuracy (%) per fold")]):
        data = [fold_df[fold_df.model == k].sort_values("fold")[m].values for k in KEYS]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch, k in zip(bp["boxes"], KEYS):
            patch.set_facecolor(COLORS[CATEGORY[k]])
            patch.set_alpha(0.6)
        for i, d in enumerate(data):
            ax.scatter(np.full(len(d), i + 1) + np.random.uniform(-0.08, 0.08, len(d)), d,
                       color="k", s=10, zorder=3)
        ax.set_xticks(range(1, len(KEYS) + 1))
        ax.set_xticklabels([NAMES[k] for k in KEYS], rotation=40, ha="right")
        ax.set_title(title)
    fig.tight_layout()
    return save_fig(fig, "fig03_fold_boxplots.png")


def plot_metric_heatmap(summary_means):
    cols = ["track_top1", "prog_top1", "prog_top3", "prog_macro_f1", "prog_bal_acc", "track_macro_f1"]
    labels = ["Track\nTop-1 %", "Prog\nTop-1 %", "Prog\nTop-3 %", "Prog\nMacro-F1",
              "Prog\nBal-Acc %", "Track\nMacro-F1"]
    data = summary_means.reindex(KEYS)[cols].values.astype(float)
    norm = (data - data.min(0)) / (data.max(0) - data.min(0) + 1e-12)   # column-wise colour scale
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.imshow(norm, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(KEYS)))
    ax.set_yticklabels([NAMES[k] for k in KEYS])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:.3f}" if "f1" in cols[j] else f"{v:.1f}", ha="center", va="center",
                    fontsize=8, color="white" if norm[i, j] > 0.6 else "black")
    ax.set_title("Mean performance heatmap (colour scaled per metric column)")
    fig.tight_layout()
    return save_fig(fig, "fig04_metric_heatmap.png")


def plot_pvalue_heatmap(pdf):
    cols = [("p_macro_f1", "Prog Macro-F1"), ("p_prog_top1", "Prog Top-1"),
            ("p_prog_top3", "Prog Top-3"), ("p_track_top1", "Track Top-1")]
    others = [k for k in KEYS if k != PROPOSED]
    data = np.array([[pdf.loc[k, c] for c, _ in cols] for k in others], dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    draw_heatmap(ax, -np.log10(np.clip(data, 1e-6, 1)), [l for _, l in cols], [NAMES[k] for k in others],
                 cmap="magma", annotate=False, rotation=0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.3f}" + ("*" if data[i, j] < 0.05 else ""), ha="center",
                    va="center", fontsize=8, color="white" if data[i, j] > 0.2 else "black")
    ax.set_title("Paired t-test p-values: Proposed vs each model (fold-wise)\n* p < 0.05 (uncorrected)")
    fig.tight_layout()
    return save_fig(fig, "fig05_pvalue_heatmap.png")


def plot_confusion(y_true, y_pred, labels, row_labels, title, fname, subdir=None, figsize=(13, 11),
                   fontsize=6, number_ticks=False):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    rows = cm.sum(1, keepdims=True)
    cmn = np.divide(cm, rows, out=np.zeros(cm.shape, dtype=float), where=rows > 0)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    if number_ticks:
        ax.set_xticklabels([str(i + 1) for i in range(len(labels))])
    else:
        ax.set_xticklabels(row_labels, rotation=60, ha="right", fontsize=fontsize + 1)
    ax.set_yticklabels([f"{i + 1}. {l}" if number_ticks else l for i, l in enumerate(row_labels)],
                       fontsize=fontsize + 1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] > 0:
                ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})" if len(labels) <= 6 else f"{cm[i, j]}",
                        ha="center", va="center", fontsize=fontsize + (3 if len(labels) <= 6 else 0),
                        color="white" if cmn[i, j] > 0.55 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, label="row-normalised recall")
    fig.tight_layout()
    return save_fig(fig, fname, subdir)


def plot_track_confusion_grid(oof_pred_track, true_track):
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    short_t = ["Ind&Mgmt", "Civil&Infra", "Mech/Elec/Tech", "Chem&Proc"]
    for ax, k in zip(axes.ravel(), KEYS):
        cm = confusion_matrix(true_track, oof_pred_track[k], labels=np.arange(4))
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.55 else "black")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(short_t, rotation=40, ha="right", fontsize=7)
        ax.set_yticklabels(short_t, fontsize=7)
        acc = 100 * np.mean(oof_pred_track[k] == true_track)
        ax.set_title(f"{NAMES[k]}\nTrack acc {acc:.1f}%", fontsize=8,
                     color=COLORS[CATEGORY[k]] if k == PROPOSED else "black")
    fig.suptitle("Pooled out-of-fold macro-track confusion matrices (row-normalised)")
    fig.tight_layout()
    return save_fig(fig, "fig07_track_confusion_grid.png")


def plot_per_class_f1(oof_proba, y, programs):
    support = np.bincount(y, minlength=len(programs))
    order = np.argsort(-support)
    data = np.array([f1_score(y, oof_proba[k].argmax(1), labels=np.arange(len(programs)),
                              average=None, zero_division=0) for k in KEYS])[:, order]
    fig, ax = plt.subplots(figsize=(15, 5.5))
    im = draw_heatmap(ax, data, [f"{short(programs[i], 22)} (n={support[i]})" for i in order],
                      [NAMES[k] for k in KEYS], cmap="YlGnBu", vmin=0, vmax=1, fontsize=6, rotation=60)
    fig.colorbar(im, ax=ax, fraction=0.025, label="per-class F1")
    ax.set_title("Per-programme F1 (pooled out-of-fold predictions; columns ordered by support)")
    fig.tight_layout()
    return save_fig(fig, "fig08_per_class_f1_heatmap.png")


def plot_topk(oof_proba, y, n_classes):
    ks = range(1, 6)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for k in KEYS:
        acc = [100 * top_k_accuracy_score(y, oof_proba[k], k=kk, labels=np.arange(n_classes)) for kk in ks]
        ax.plot(list(ks), acc, marker="o", lw=2.5 if k == PROPOSED else 1.2,
                color=COLORS[CATEGORY[k]] if k == PROPOSED else None, label=NAMES[k])
    ax.axhline(85, color="k", ls="--", lw=0.8)
    ax.text(1.02, 85.5, "85% Top-3 target", fontsize=7)
    ax.set_xlabel("k")
    ax.set_ylabel("Top-k accuracy (%)")
    ax.set_xticks(list(ks))
    ax.set_title("Programme Top-k recommendation accuracy (pooled OOF)")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    return save_fig(fig, "fig09_topk_curves.png")


def plot_training_curves(histories):
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5), sharex=True)
    for ax, k in zip(axes.ravel(), DEEP_KEYS):
        h = np.stack(histories[k])            # (folds, epochs, 2)
        ep = np.arange(1, h.shape[1] + 1)
        for j, (lab, ls) in enumerate([("train loss (weighted CE)", "-"), ("validation loss (CE)", "--")]):
            mu, sd = h[:, :, j].mean(0), h[:, :, j].std(0)
            ax.plot(ep, mu, ls, label=lab)
            ax.fill_between(ep, mu - sd, mu + sd, alpha=0.2)
        ax.set_title(NAMES[k], fontsize=9)
        ax.set_xlabel("epoch")
    axes.ravel()[0].legend(fontsize=7, frameon=False)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Training curves (mean ± std over folds)")
    fig.tight_layout()
    return save_fig(fig, "fig06_training_curves.png")


def plot_results_table(table):
    fig, ax = plt.subplots(figsize=(17, 0.55 * len(table) + 1.4))
    ax.axis("off")
    tb = ax.table(cellText=table.values, colLabels=table.columns, loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(7.5)
    tb.scale(1, 1.6)
    for (r, c), cell in tb.get_celld().items():
        if r == 0:
            cell.set_facecolor("#333333")
            cell.set_text_props(color="white", weight="bold")
        elif table.iloc[r - 1, 0].startswith("Proposed"):
            cell.set_facecolor("#fde0dd")
    ax.set_title("Objective 1 - results summary (5-fold CV, mean ± std)", pad=12)
    return save_fig(fig, "fig10_results_table.png")


# =============================================================================
# 7. MAIN EXPERIMENT
# =============================================================================
def main():
    t0 = time.time()
    print(f"Device: {DEVICE} | epochs={EPOCHS} | quick_test={QUICK_TEST} | "
          f"saber_pro_features={USE_SABER_PRO_FEATURES}")
    if USE_SABER_PRO_FEATURES:
        print("NOTE: G_SC / PERCENTILE / 2ND_DECILE / QUARTILE are Saber-Pro (end-of-degree) "
              "outcomes; set USE_SABER_PRO_FEATURES=0 for a leakage-safe run.")

    df, y, programs, prog_track_idx, M = prepare_data()
    K = len(programs)
    N = len(df)
    true_track = prog_track_idx[y]
    print(f"[data] {N} students after filtering | {K} programmes | {len(TRACK_NAMES)} macro-tracks")
    print(f"[data] majority-class floors: programme {100 * np.bincount(y).max() / N:.1f}% | "
          f"track {100 * np.bincount(true_track).max() / N:.1f}%")
    print(df["MACRO_TRACK"].value_counts().to_string())
    plot_class_distribution(df, programs)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_proba = {k: np.zeros((N, K), dtype=np.float32) for k in KEYS}
    histories = {k: [] for k in DEEP_KEYS}
    fold_rows = []

    for fold, (tr, va) in enumerate(skf.split(np.zeros(N), y)):
        print(f"\n=== Fold {fold + 1}/{N_SPLITS} | train={len(tr)} val={len(va)} ===")
        data = preprocess_fold(df.iloc[tr], df.iloc[va])
        y_tr, y_va = y[tr], y[va]
        for key in KEYS:
            ts = time.time()
            if key in DEEP_KEYS:
                proba, hist = fit_predict_torch(key, data, y_tr, y_va, K, seed=SEED + fold)
                histories[key].append(hist)
            else:
                proba = fit_predict_sklearn(key, data, y_tr, K)
            oof_proba[key][va] = proba
            m, _ = compute_metrics(y_va, proba, prog_track_idx, K)
            fold_rows.append(dict(model=key, fold=fold + 1, **m))
            print(f"  {NAMES[key]:<32} track={m['track_top1']:5.1f}%  top1={m['prog_top1']:5.1f}%  "
                  f"top3={m['prog_top3']:5.1f}%  macroF1={m['prog_macro_f1']:.3f}  ({time.time() - ts:.0f}s)")

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(os.path.join(OUT_DIR, "fold_metrics.csv"), index=False)

    # ---------------------------------------------------------------- statistics
    metric_cols = ["track_top1", "prog_top1", "prog_top3", "prog_macro_f1", "prog_bal_acc", "track_macro_f1"]
    means = fold_df.groupby("model")[metric_cols].mean()
    stds = fold_df.groupby("model")[metric_cols].std(ddof=1)

    def fold_vec(k, m):
        return fold_df[fold_df.model == k].sort_values("fold")[m].values

    pv = pd.DataFrame(index=KEYS, columns=["p_macro_f1", "p_prog_top1", "p_prog_top3", "p_track_top1"],
                      dtype=float)
    for k in KEYS:
        if k == PROPOSED:
            continue
        for col, m in [("p_macro_f1", "prog_macro_f1"), ("p_prog_top1", "prog_top1"),
                       ("p_prog_top3", "prog_top3"), ("p_track_top1", "track_top1")]:
            pv.loc[k, col] = paired_p(fold_vec(PROPOSED, m), fold_vec(k, m))
    others = [k for k in KEYS if k != PROPOSED]
    pv["p_macro_f1_holm"] = np.nan
    pv.loc[others, "p_macro_f1_holm"] = holm_adjust(pv.loc[others, "p_macro_f1"].values)
    pv.to_csv(os.path.join(OUT_DIR, "pvalues_vs_proposed.csv"))

    # ------------------------------------------------------------ results table
    def ms(k, m, pct=True):
        return f"{means.loc[k, m]:.2f} ± {stds.loc[k, m]:.2f}" if pct else \
               f"{means.loc[k, m]:.3f} ± {stds.loc[k, m]:.3f}"

    rows = []
    for k in KEYS:
        rows.append({
            "Model": NAMES[k], "Category": CATEGORY[k],
            "Macro-Track Top-1 (%)": ms(k, "track_top1"),
            "Programme Top-1 (%)": ms(k, "prog_top1"),
            "Programme Top-3 (%)": ms(k, "prog_top3"),
            "Programme Macro-F1": ms(k, "prog_macro_f1", pct=False),
            "p-value vs Proposed (Macro-F1)": "-" if k == PROPOSED else f"{pv.loc[k, 'p_macro_f1']:.4f}",
            "Holm-adj. p": "-" if k == PROPOSED else f"{pv.loc[k, 'p_macro_f1_holm']:.4f}",
        })
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_CSV, index=False)
    table.to_csv(os.path.join(OUT_DIR, RESULTS_CSV), index=False)
    try:
        with open(os.path.join(OUT_DIR, "objective1_results_table.tex"), "w") as f:
            f.write(table.to_latex(index=False, escape=True))
    except Exception:
        pass
    pd.concat([means.add_suffix("_mean"), stds.add_suffix("_std")], axis=1).to_csv(
        os.path.join(OUT_DIR, "summary_means_stds.csv"))

    print("\n" + "=" * 130)
    print("OBJECTIVE 1 - RESULTS (5-fold stratified CV, mean ± std)")
    print("=" * 130)
    print(table.to_string(index=False))

    # ----------------------------------------------- pooled OOF artefacts & plots
    oof_pred = {k: oof_proba[k].argmax(1) for k in KEYS}
    oof_pred_track = {k: (oof_proba[k] @ np.eye(len(TRACK_NAMES))[prog_track_idx]).argmax(1) for k in KEYS}
    np.savez_compressed(os.path.join(OUT_DIR, "oof_probabilities.npz"), y_true=y, programs=np.array(programs),
                        **{k: oof_proba[k] for k in KEYS})
    pd.DataFrame({"true_program": [programs[i] for i in y],
                  "true_track": [TRACK_NAMES[i] for i in true_track],
                  **{f"pred_program_{k}": [programs[i] for i in oof_pred[k]] for k in KEYS}}
                 ).to_csv(os.path.join(OUT_DIR, "oof_predictions.csv"), index=False)

    plot_metric_bars(fold_df)
    plot_fold_boxplot(fold_df)
    plot_metric_heatmap(means)
    plot_pvalue_heatmap(pv)
    plot_training_curves(histories)
    plot_track_confusion_grid(oof_pred_track, true_track)
    plot_per_class_f1(oof_proba, y, programs)
    plot_topk(oof_proba, y, K)
    plot_results_table(table)

    short_prog = [short(p, 34) for p in programs]
    for k in KEYS:
        plot_confusion(y, oof_pred[k], programs, short_prog,
                       f"{NAMES[k]} - programme confusion (pooled OOF, row-normalised; cell = count)",
                       f"cm_program_{k}.png", subdir="confusion_matrices", figsize=(14, 12),
                       fontsize=5, number_ticks=True)
        plot_confusion(true_track, oof_pred_track[k], TRACK_NAMES, TRACK_NAMES,
                       f"{NAMES[k]} - macro-track confusion (pooled OOF)",
                       f"cm_track_{k}.png", subdir="confusion_matrices", figsize=(7, 6), fontsize=7)
    # headline figures for the proposed model
    plot_confusion(y, oof_pred[PROPOSED], programs, short_prog,
                   "Proposed model - 21-programme confusion matrix (pooled OOF)",
                   "fig11_proposed_confusion_programs.png", figsize=(14, 12), fontsize=5, number_ticks=True)
    plot_confusion(true_track, oof_pred_track[PROPOSED], TRACK_NAMES, TRACK_NAMES,
                   "Proposed model - macro-track confusion matrix (pooled OOF)",
                   "fig12_proposed_confusion_tracks.png", figsize=(7.5, 6.5), fontsize=7)

    rep = classification_report(y, oof_pred[PROPOSED], labels=np.arange(K), target_names=programs,
                                zero_division=0, output_dict=True)
    pd.DataFrame(rep).T.to_csv(os.path.join(OUT_DIR, "proposed_per_class_report.csv"))

    # --------------------------------------------------------------- text report
    best_f1 = means["prog_macro_f1"].idxmax()
    lines = ["OBJECTIVE 1 - AUTOMATIC SUMMARY", "-" * 60,
             f"Students: {N} | Programmes: {K} | Tracks: {len(TRACK_NAMES)} | folds: {N_SPLITS} | epochs: {EPOCHS}",
             f"Saber-Pro features in temporal branch: {USE_SABER_PRO_FEATURES}",
             f"Majority-class floor: programme {100 * np.bincount(y).max() / N:.1f}%, "
             f"track {100 * np.bincount(true_track).max() / N:.1f}%", "",
             f"Best Macro-F1 model: {NAMES[best_f1]} ({means.loc[best_f1, 'prog_macro_f1']:.3f})",
             f"Proposed Macro-F1: {means.loc[PROPOSED, 'prog_macro_f1']:.3f}", ""]
    for k in others:
        d = means.loc[PROPOSED, "prog_macro_f1"] - means.loc[k, "prog_macro_f1"]
        verdict = ("significantly" if pv.loc[k, "p_macro_f1"] < 0.05 else "NOT significantly")
        lines.append(f"Proposed vs {NAMES[k]:<30} dMacro-F1={d:+.3f}  p={pv.loc[k, 'p_macro_f1']:.4f} "
                     f"(Holm {pv.loc[k, 'p_macro_f1_holm']:.4f}) -> {verdict} different at 0.05")
    lines += ["", "Target check (best model on each metric):"]
    bt = means["track_top1"].idxmax()
    b3 = means["prog_top3"].idxmax()
    lines.append(f"  Macro-track Top-1 > 80%: best = {NAMES[bt]} {means.loc[bt, 'track_top1']:.1f}% "
                 f"-> {'MET' if means.loc[bt, 'track_top1'] > 80 else 'NOT MET'}; "
                 f"proposed = {means.loc[PROPOSED, 'track_top1']:.1f}%")
    lines.append(f"  Programme Top-3 > 85%:  best = {NAMES[b3]} {means.loc[b3, 'prog_top3']:.1f}% "
                 f"-> {'MET' if means.loc[b3, 'prog_top3'] > 85 else 'NOT MET'}; "
                 f"proposed = {means.loc[PROPOSED, 'prog_top3']:.1f}%")
    lines += ["", "With only 5 folds, paired t-tests have 4 degrees of freedom and low power; "
              "report non-significant differences as such."]
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(OUT_DIR, "experiment_report.txt"), "w") as f:
        f.write(report)
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump(dict(seed=SEED, n_splits=N_SPLITS, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                       weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP, rf_trees=RF_TREES, hgb_iter=HGB_ITER,
                       use_saber_pro_features=USE_SABER_PRO_FEATURES, temporal_steps=TEMPORAL_STEPS,
                       static_cols=STATIC_COLS, device=str(DEVICE), quick_test=QUICK_TEST,
                       torch=torch.__version__), f, indent=2)

    print(f"\nAll outputs saved in ./{OUT_DIR}/ and ./{RESULTS_CSV}  (total {(time.time() - t0) / 60:.1f} min)")
    return table


if __name__ == "__main__":
    main()
