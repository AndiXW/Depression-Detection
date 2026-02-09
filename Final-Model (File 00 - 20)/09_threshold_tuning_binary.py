import os, json, numpy as np, pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROC_BIN = r"C:\Users\huyng\Documents\sdsu fall 2025\ds577_data_binary_proc\v2_no_stop"
OUT = r"C:\Users\huyng\Documents\sdsu fall 2025\ds577_metrics"
os.makedirs(OUT, exist_ok=True)

# Use the best params you found earlier
BEST = {"tfidf__ngram_range": (1,2), "tfidf__min_df": 5, "clf__C": 0.5}

train = pd.read_csv(os.path.join(PROC_BIN, "train.csv"))
val   = pd.read_csv(os.path.join(PROC_BIN, "val.csv"))
test  = pd.read_csv(os.path.join(PROC_BIN, "test.csv"))
for d in (train, val, test):
    d["text_proc"] = d["text_proc"].astype(str)

# Build the fixed pipeline
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=BEST["tfidf__ngram_range"], min_df=BEST["tfidf__min_df"], max_df=0.9,
                              max_features=200_000, sublinear_tf=True, dtype=np.float32)),
    ("clf", LogisticRegression(max_iter=2000, solver="saga", class_weight="balanced", C=BEST["clf__C"]))
])

# 1) Fit on TRAIN only; pick threshold via VAL
pipe.fit(train["text_proc"], train["label"])
val_probs = pipe.named_steps["clf"].predict_proba(pipe.named_steps["tfidf"].transform(val["text_proc"]))[:,1]

# grid of thresholds
ts = np.linspace(0.2, 0.8, 61)  # 0.2 -> 0.8 step=0.01
best_t, best_f1 = 0.5, -1.0
for t in ts:
    val_pred = (val_probs >= t).astype(int)
    f1 = f1_score(val["label"], val_pred, average="macro")
    if f1 > best_f1:
        best_f1, best_t = f1, t

print(f"Chosen threshold on VAL for macro-F1: t*={best_t:.2f}  (VAL macro-F1={best_f1:.3f})")

# 2) Refit on TRAIN+VAL and evaluate on TEST using t*
tv = pd.concat([train, val], ignore_index=True)
pipe.fit(tv["text_proc"], tv["label"])
test_probs = pipe.named_steps["clf"].predict_proba(pipe.named_steps["tfidf"].transform(test["text_proc"]))[:,1]
test_pred = (test_probs >= best_t).astype(int)

acc = accuracy_score(test["label"], test_pred)
f1m = f1_score(test["label"], test_pred, average="macro")
cm  = confusion_matrix(test["label"], test_pred, labels=[0,1])

print("TEST (with threshold): acc=", round(acc,4), "macro-F1=", round(f1m,4))
print("Confusion matrix (0=not-dep,1=dep):\n", cm)
print("Classification report:\n", classification_report(test["label"], test_pred))

with open(os.path.join(OUT, "binary_threshold_tuned.json"), "w", encoding="utf-8") as f:
    json.dump({"best_threshold": float(best_t), "val_macro_f1": float(best_f1),
               "test_acc": float(acc), "test_macro_f1": float(f1m), "confusion_matrix": cm.tolist()}, f, indent=2)
print("Saved binary_threshold_tuned.json")
