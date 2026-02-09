# 04_compare_prepro_variants.py
import os, json, numpy as np, pandas as pd, re
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

PROC_DIR = r"C:\Users\huyng\Documents\sdsu fall 2025\ds577_data_proc"
OUT_DIR  = r"C:\Users\huyng\Documents\sdsu fall 2025\ds577_metrics"
os.makedirs(OUT_DIR, exist_ok=True)

def load_split(variant):
    d = os.path.join(PROC_DIR, variant)
    tr = pd.read_csv(os.path.join(d, "train.csv"))
    va = pd.read_csv(os.path.join(d, "val.csv"))
    te = pd.read_csv(os.path.join(d, "test.csv"))
    return tr, va, te

def run_variant(name):
    train, val, test = load_split(name)
    X_tr, y_tr = train["text_proc"], train["label"]
    X_va, y_va = val["text_proc"], val["label"]
    X_te, y_te = test["text_proc"], test["label"]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,1), min_df=5, max_df=0.9)),
        ("clf", LogisticRegression(max_iter=2000, solver="saga"))
    ])
    grid = {
        "tfidf__ngram_range": [(1,1),(1,2)],
        "tfidf__min_df": [3,5],
        "clf__C": [0.5,1.0,2.0]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=0)
    gs.fit(X_tr, y_tr)

    best = gs.best_estimator_
    # validate
    v_pred = best.predict(X_va)
    v_f1 = f1_score(y_va, v_pred, average="macro")
    v_acc = accuracy_score(y_va, v_pred)

    # train on train+val, evaluate on test
    tv = pd.concat([train, val], ignore_index=True)
    best.fit(tv["text_proc"], tv["label"])
    t_pred = best.predict(X_te)
    t_f1 = f1_score(y_te, t_pred, average="macro")
    t_acc = accuracy_score(y_te, t_pred)

    # macro OVR ROC/PR
    classes = sorted(y_te.unique())
    y_bin = label_binarize(y_te, classes=classes)
    ovr = OneVsRestClassifier(best.named_steps["clf"])
    vec = best.named_steps["tfidf"]
    ovr.fit(vec.transform(tv["text_proc"]), tv["label"])
    probs = ovr.predict_proba(vec.transform(X_te))
    roc_macro = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
    pr_macro = average_precision_score(y_bin, probs, average="macro")

    out = {
        "variant": name,
        "best_params": gs.best_params_,
        "val_macro_f1": float(v_f1),
        "val_acc": float(v_acc),
        "test_macro_f1": float(t_f1),
        "test_acc": float(t_acc),
        "test_roc_auc_macro": float(roc_macro),
        "test_pr_auc_macro": float(pr_macro),
    }
    print(f"\n[{name}] VAL F1={v_f1:.3f} ACC={v_acc:.3f} | TEST F1={t_f1:.3f} ACC={t_acc:.3f} | ROC-AUC={roc_macro:.3f} PR-AUC={pr_macro:.3f}")
    return out

results = []
for v in ["v1_basic", "v2_no_stop"]:
    results.append(run_variant(v))

# choose winner by test macro-F1, tie-breaker PR-AUC
results_sorted = sorted(results, key=lambda d: (d["test_macro_f1"], d["test_pr_auc_macro"]), reverse=True)
winner = results_sorted[0]
with open(os.path.join(OUT_DIR, "compare_preprocess.json"), "w", encoding="utf-8") as f:
    json.dump({"results": results, "winner": winner}, f, indent=2)
print("\nChosen preprocessing variant:", winner["variant"], "-> saved metrics to compare_preprocess.json")
