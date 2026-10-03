"""RO1 baseline: leakage-safe stratified 5-fold CV, target = QUARTILE."""
import json
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score, accuracy_score
from scipy.stats import wilcoxon

SEED = 42
d = pd.read_csv("/home/user/hanafithman79/dataset/dataset.csv", encoding="latin-1")
d = d.drop(columns=[c for c in d.columns if c.startswith("Unnamed")])
y = d["QUARTILE"].astype(int)
# leakage: Saber Pro outcomes, institution/program, selection/percentile/decile fields, IDs
drop = ["QUARTILE", "PERCENTILE", "2ND_DECILE", "SEL", "SEL_IHE", "G_SC", "UNIVERSITY", "ACADEMIC_PROGRAM",
        "QR_PRO", "CR_PRO", "CC_PRO", "ENG_PRO", "WC_PRO", "FEP_PRO", "Cod_SPro", "COD_S11"]
X = d.drop(columns=drop)
X["SCHOOL_NAME"] = X["SCHOOL_NAME"].str.strip()
num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat = [c for c in X.columns if c not in num]
print("features:", len(X.columns), "num", num, "cat", len(cat))

def prep():
    return ColumnTransformer([
        ("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
        ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                        ("o", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20))]), cat)])

models = {
    "majority": DummyClassifier(strategy="most_frequent"),
    "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "rf": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=4, random_state=SEED),
}
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows, oof = [], pd.DataFrame({"true_label": y})
for name, m in models.items():
    oof["pred_" + name] = -1
for k, (tr, te) in enumerate(skf.split(X, y)):
    for name, m in models.items():
        p = Pipeline([("prep", prep()), ("m", m)]).fit(X.iloc[tr], y.iloc[tr])
        pr = p.predict(X.iloc[te])
        oof.loc[oof.index[te], "pred_" + name] = pr
        rows.append(dict(model=name, fold=k, macro_f1=f1_score(y.iloc[te], pr, average="macro"),
                         bal_acc=balanced_accuracy_score(y.iloc[te], pr),
                         macro_recall=recall_score(y.iloc[te], pr, average="macro"),
                         acc=accuracy_score(y.iloc[te], pr)))
    print("fold", k, "done", flush=True)
f = pd.DataFrame(rows)
rng = np.random.default_rng(SEED)
def ci(v):
    b = [rng.choice(v, len(v)).mean() for _ in range(2000)]
    return np.percentile(b, [2.5, 97.5])
summ = f.groupby("model")[["macro_f1", "bal_acc", "macro_recall", "acc"]].mean()
for mt in ["macro_f1"]:
    for mo in summ.index:
        lo, hi = ci(f[f.model == mo][mt].values)
        summ.loc[mo, mt + "_ci"] = f"[{lo:.3f},{hi:.3f}]"
print(summ.round(3).to_string())
base = f[f.model == "logreg"].sort_values("fold")
for mo in ["rf"]:
    c = f[f.model == mo].sort_values("fold")
    try:
        print(mo, "vs logreg Wilcoxon p (n=5, min possible 0.0625):", wilcoxon(c.macro_f1.values, base.macro_f1.values).pvalue)
    except Exception as e:
        print("wilcoxon failed", e)
f.to_csv("/home/user/studypathways_objective1/results/ro1_fold_metrics.csv", index=False)
oof.to_csv("/home/user/studypathways_objective1/results/ro1_oof_predictions.csv", index=False)
summ.to_csv("/home/user/studypathways_objective1/results/ro1_summary.csv")
