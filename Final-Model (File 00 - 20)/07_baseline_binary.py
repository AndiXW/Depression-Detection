# 07_baseline_binary.py  (fixed)
import os, json, time
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
import joblib

# --------------------------------------
# Paths
# --------------------------------------
ROOT = r"C:\Users\huyng\Documents\sdsu fall 2025"
PROC_BIN = os.path.join(ROOT, r"ds577_data_binary_proc\v2_no_stop")
OUT_MODELS  = os.path.join(ROOT, "ds577_models")
OUT_METRICS = os.path.join(ROOT, "ds577_metrics")
os.makedirs(OUT_MODELS, exist_ok=True)
os.makedirs(OUT_METRICS, exist_ok=True)

# --------------------------------------
# Speed toggle: skip GridSearch if you want a fast rerun
# --------------------------------------
DO_GRID = False  # set True if you want to re-tune; False uses best params you already found

BEST_TFIDF_NGRAM = (1, 2)
BEST_TFIDF_MINDF = 5
BEST_C = 0.5

TARGET_TUNE = 250_000  # when DO_GRID=True, limit rows used for grid tuning

# --------------------------------------
# Load & clean
# --------------------------------------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_proc"] = df["text_proc"].astype(str)
    df["text_proc"] = df["text_proc"].replace({"nan": "", "NaN": "", "None": ""})
    df = df.dropna(subset=["text_proc", "label"])
    df = df[df["text_proc"].str.len() > 0]
    return df

train = pd.read_csv(os.path.join(PROC_BIN, "train.csv"))
val   = pd.read_csv(os.path.join(PROC_BIN, "val.csv"))
test  = pd.read_csv(os.path.join(PROC_BIN, "test.csv"))

train = clean_df(train)
val   = clean_df(val)
test  = clean_df(test)

# unify dtypes for safety
train["label"] = train["label"].astype(int)
val["label"]   = val["label"].astype(int)
test["label"]  = test["label"].astype(int)

tv = pd.concat([train[["text_proc","label"]], val[["text_proc","label"]]], ignore_index=True)

X_test = test["text_proc"]
y_test = test["label"]

# --------------------------------------
# Build pipeline (TF-IDF + Logistic Regression)
# --------------------------------------
def make_pipeline(ngram=(1,2), min_df=5, C=0.5):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=ngram, min_df=min_df, max_df=0.9,
            max_features=200_000, sublinear_tf=True, dtype=np.float32
        )),
        ("clf", LogisticRegression(
            max_iter=2000, solver="saga", class_weight="balanced", n_jobs=None, C=C, random_state=42
        ))
    ])

start = time.time()
if DO_GRID:
    # Sample for tuning to keep runtime reasonable
    if len(tv) > TARGET_TUNE:
        tv_sample, _ = train_test_split(
            tv, train_size=TARGET_TUNE, stratify=tv["label"], random_state=42
        )
    else:
        tv_sample = tv

    X_tune, y_tune = tv_sample["text_proc"], tv_sample["label"]

    pipe = make_pipeline()  # defaults; grid will override
    grid = {
        "tfidf__ngram_range": [(1,1), (1,2)],
        "tfidf__min_df": [5, 10],
        "clf__C": [0.5, 1.0, 2.0],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1)
    gs.fit(X_tune, y_tune)
    print("Best params:", gs.best_params_, "| best CV macro-F1:", round(gs.best_score_, 4))
    best = gs.best_estimator_
    best_params = gs.best_params_
else:
    # Use your previously discovered best hyperparams to save time
    best_params = {
        "tfidf__ngram_range": BEST_TFIDF_NGRAM,
        "tfidf__min_df": BEST_TFIDF_MINDF,
        "clf__C": BEST_C
    }
    best = make_pipeline(
        ngram=BEST_TFIDF_NGRAM, min_df=BEST_TFIDF_MINDF, C=BEST_C
    )

# Refit on ALL train+val
best.fit(tv["text_proc"], tv["label"])

# --------------------------------------
# Evaluate on test
# --------------------------------------
y_pred = best.predict(X_test)

# predict_proba -> [:,1] is P(dep=1)
# (LogReg with saga supports predict_proba)
y_proba_pos = best.named_steps["clf"].predict_proba(
    best.named_steps["tfidf"].transform(X_test)
)[:, 1]
# also a 2-col matrix if you need it later
y_proba = np.vstack([1.0 - y_proba_pos, y_proba_pos]).T

test_acc     = accuracy_score(y_test, y_pred)
test_f1      = f1_score(y_test, y_pred, average="macro")
test_roc_auc = roc_auc_score(y_test, y_proba_pos)
test_pr_auc  = average_precision_score(y_test, y_proba_pos)

print(f"TEST acc: {test_acc:.4f}  macro-F1: {test_f1:.4f}")
print("\nTEST report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, labels=[0,1])
print("Confusion matrix (labels 0=not-dep, 1=dep):\n", cm)
print("TEST ROC-AUC:", round(test_roc_auc,4), "PR-AUC:", round(test_pr_auc,4))

# --------------------------------------
# Save model + metrics in the locations that file 13 expects
# --------------------------------------
joblib_path = os.path.join(OUT_MODELS, "logreg_binary_v2_no_stop.joblib")
joblib.dump(best, joblib_path)

metrics = {
    "best_params": best_params,
    "rows": {"train_val_full": int(len(tv)), "test": int(len(test))},
    "test_acc": float(test_acc),
    "test_macro_f1": float(test_f1),
    "test_roc_auc": float(test_roc_auc),
    "test_pr_auc": float(test_pr_auc),
    "confusion_matrix": cm.tolist(),
    "seconds": round(time.time() - start, 1),
    "did_grid_search": bool(DO_GRID)
}
with open(os.path.join(OUT_METRICS, "logreg_binary_v2_no_stop_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("Saved baseline model to:", joblib_path)
print("Saved metrics JSON to:", os.path.join(OUT_METRICS, "logreg_binary_v2_no_stop_metrics.json"))
