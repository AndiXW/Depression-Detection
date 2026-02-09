# 02_eda_and_baseline.py
import os, re, json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import joblib

# ---------- paths ----------
DATA_DIR = r"C:\Users\huyng\Documents\sdsu fall 2025\ds577_data"
OUT_FIGS = r"C:\Users\huyng\Documents\sdsu fall 2025\ds577_figs"
OUT_MODELS = r"C:\Users\huyng\Documents\sdsu fall 2025\ds577_models"
OUT_METRICS = r"C:\Users\huyng\Documents\sdsu fall 2025\ds577_metrics"
os.makedirs(OUT_FIGS, exist_ok=True)
os.makedirs(OUT_MODELS, exist_ok=True)
os.makedirs(OUT_METRICS, exist_ok=True)

train = pd.read_csv(os.path.join(DATA_DIR, "labelled_train.csv"))
val   = pd.read_csv(os.path.join(DATA_DIR, "labelled_val.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "labelled_test.csv"))

# ---------- quick EDA + leakage checks ----------
def basic_eda(df, name):
    print(f"\n--- {name} ---")
    print("rows:", len(df))
    print("label counts:\n", df["label"].value_counts())
    lens = df["text"].astype(str).str.len()
    print("text length (chars): min/median/mean/max =",
          int(lens.min()), int(lens.median()), round(lens.mean(),1), int(lens.max()))
basic_eda(train, "TRAIN")
basic_eda(val,   "VAL")
basic_eda(test,  "TEST")

# Note: we intentionally DO NOT use columns like 'subreddit' as features to avoid leakage.

# ---------- light text cleaning ----------
def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)       # URLs
    s = re.sub(r"[@#]\w+", " ", s)                # mentions/hashtags
    s = re.sub(r"&amp;", " ", s)
    s = re.sub(r"[^a-z0-9\s']", " ", s)           # keep basic chars + apostrophes
    s = re.sub(r"\s+", " ", s).strip()
    return s

for df in (train, val, test):
    df["text_clean"] = df["text"].astype(str).map(clean_text)

# ---------- baseline pipeline: TF-IDF + Logistic Regression ----------
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.9)),
    ("clf", LogisticRegression(max_iter=2000, solver="saga", multi_class="auto", n_jobs=None))
])

param_grid = {
    "tfidf__ngram_range": [(1,1), (1,2)],
    "tfidf__min_df": [3,5],
    "clf__C": [0.5, 1.0, 2.0]
}

X_train, y_train = train["text_clean"], train["label"]
X_val,   y_val   = val["text_clean"],   val["label"]
X_test,  y_test  = test["text_clean"],  test["label"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1)
search.fit(X_train, y_train)
print("\nBest params:", search.best_params_)
print("Best CV f1_macro:", round(search.best_score_, 4))

# eval on VAL
best = search.best_estimator_
val_pred = best.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
val_f1m = f1_score(y_val, val_pred, average="macro")
print("\nVAL accuracy:", round(val_acc,4), "VAL macro-F1:", round(val_f1m,4))
print("\nVAL classification report:\n", classification_report(y_val, val_pred))

# retrain on TRAIN+VAL then test
tv = pd.concat([train, val], ignore_index=True)
X_tv, y_tv = tv["text_clean"], tv["label"]
best.fit(X_tv, y_tv)

test_pred = best.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)
test_f1m = f1_score(y_test, test_pred, average="macro")
print("\nTEST accuracy:", round(test_acc,4), "TEST macro-F1:", round(test_f1m,4))
print("\nTEST classification report:\n", classification_report(y_test, test_pred))
cm = confusion_matrix(y_test, test_pred, labels=sorted(y_test.unique()))
print("Confusion matrix (labels in order):", sorted(y_test.unique()))
print(cm)

# ---------- ROC-AUC & PR-AUC (macro, OVR) ----------
classes = sorted(y_test.unique())
y_test_bin = label_binarize(y_test, classes=classes)
ovr = OneVsRestClassifier(best.named_steps["clf"])
# Use the TF-IDF transform from the pipeline
X_test_vec = best.named_steps["tfidf"].transform(X_test)
ovr.fit(best.named_steps["tfidf"].transform(X_tv), y_tv)
probs = ovr.predict_proba(X_test_vec)
roc_macro = roc_auc_score(y_test_bin, probs, average="macro", multi_class="ovr")
pr_macro  = average_precision_score(y_test_bin, probs, average="macro")
print("\nTEST ROC-AUC (macro OVR):", round(roc_macro,4),
      " | PR-AUC (macro OVR):", round(pr_macro,4))

# ---------- top features per class ----------
def top_terms_per_class(estimator, vectorizer, k=15):
    clf = estimator.named_steps["clf"]
    vec = estimator.named_steps["tfidf"]
    feat_names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_
    out = {}
    for idx, cls in enumerate(clf.classes_):
        topk = np.argsort(coefs[idx])[-k:][::-1]
        out[str(cls)] = feat_names[topk].tolist()
    return out

tops = top_terms_per_class(best, None, k=15)
print("\nTop terms by class (sanity check):")
for cls, terms in tops.items():
    print(f"- {cls}: {', '.join(terms)}")

# ---------- save artifacts ----------
joblib.dump(best, os.path.join(OUT_MODELS, "tfidf_logreg_labelled.pkl"))
with open(os.path.join(OUT_METRICS, "labelled_results.json"), "w", encoding="utf-8") as f:
    json.dump({
        "best_params": search.best_params_,
        "cv_f1_macro": float(search.best_score_),
        "val_acc": float(val_acc), "val_f1_macro": float(val_f1m),
        "test_acc": float(test_acc), "test_f1_macro": float(test_f1m),
        "test_roc_auc_macro_ovr": float(roc_macro),
        "test_pr_auc_macro_ovr": float(pr_macro),
        "labels": classes,
        "confusion_matrix": cm.tolist(),
        "top_terms": tops
    }, f, indent=2)
print("\nSaved model and metrics.")
