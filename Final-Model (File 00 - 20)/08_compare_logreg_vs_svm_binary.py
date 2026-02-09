import os, json, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize

PROC_BIN = r"C:\Users\huyng\Documents\sdsu fall 2025\ds577_data_binary_proc\v2_no_stop"
OUT = r"C:\Users\huyng\Documents\sdsu fall 2025\ds577_metrics"
os.makedirs(OUT, exist_ok=True)

train = pd.read_csv(os.path.join(PROC_BIN, "train.csv"))
val   = pd.read_csv(os.path.join(PROC_BIN, "val.csv"))
test  = pd.read_csv(os.path.join(PROC_BIN, "test.csv"))

# force strings
for d in (train, val, test):
    d["text_proc"] = d["text_proc"].astype(str)

tv = pd.concat([train[["text_proc","label"]], val[["text_proc","label"]]], ignore_index=True)
# sample for tuning
TARGET_TUNE = 200_000
if len(tv) > TARGET_TUNE:
    tv_s, _ = train_test_split(tv, train_size=TARGET_TUNE, stratify=tv["label"], random_state=42)
else:
    tv_s = tv.copy()

X_tune, y_tune = tv_s["text_proc"], tv_s["label"]
X_test, y_test = test["text_proc"], test["label"]

# --- Logistic Regression search ---
pipe_lr = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.9, max_features=200_000, sublinear_tf=True, dtype=np.float32)),
    ("clf", LogisticRegression(max_iter=2000, solver="saga", class_weight="balanced"))
])
grid_lr = {"clf__C": [0.5, 1.0, 2.0], "tfidf__ngram_range": [(1,1),(1,2)], "tfidf__min_df":[5,10]}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
gs_lr = GridSearchCV(pipe_lr, grid_lr, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1)
gs_lr.fit(X_tune, y_tune)
best_lr = gs_lr.best_estimator_

# eval LogReg on test
pred_lr = best_lr.predict(X_test)
f1_lr = f1_score(y_test, pred_lr, average="macro")
acc_lr = accuracy_score(y_test, pred_lr)
# PR-AUC/ROC-AUC via probabilities
probs_lr = best_lr.named_steps["clf"].predict_proba(best_lr.named_steps["tfidf"].transform(X_test))[:,1]
pr_lr = average_precision_score(y_test, probs_lr)
roc_lr = roc_auc_score(y_test, probs_lr)
print(f"[LogReg] TEST F1={f1_lr:.3f} ACC={acc_lr:.3f} PR-AUC={pr_lr:.3f} ROC-AUC={roc_lr:.3f} best={gs_lr.best_params_}")

# --- Linear SVM search ---
pipe_svm = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.9, max_features=200_000, sublinear_tf=True, dtype=np.float32)),
    ("clf", LinearSVC(class_weight="balanced"))
])
grid_svm = {"clf__C": [0.5, 1.0, 2.0], "tfidf__ngram_range": [(1,1),(1,2)], "tfidf__min_df":[5,10]}
gs_svm = GridSearchCV(pipe_svm, grid_svm, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1)
gs_svm.fit(X_tune, y_tune)
best_svm = gs_svm.best_estimator_

# eval LinearSVC on test (use decision_function for PR/ROC)
pred_svm = best_svm.predict(X_test)
f1_svm = f1_score(y_test, pred_svm, average="macro")
acc_svm = accuracy_score(y_test, pred_svm)
scores_svm = best_svm.named_steps["clf"].decision_function(best_svm.named_steps["tfidf"].transform(X_test))
pr_svm = average_precision_score(y_test, scores_svm)
roc_svm = roc_auc_score(y_test, scores_svm)
print(f"[LinSVM] TEST F1={f1_svm:.3f} ACC={acc_svm:.3f} PR-AUC={pr_svm:.3f} ROC-AUC={roc_svm:.3f} best={gs_svm.best_params_}")

res = {
    "logreg": {"f1": float(f1_lr), "acc": float(acc_lr), "pr_auc": float(pr_lr), "roc_auc": float(roc_lr), "best": gs_lr.best_params_},
    "linearsvm": {"f1": float(f1_svm), "acc": float(acc_svm), "pr_auc": float(pr_svm), "roc_auc": float(roc_svm), "best": gs_svm.best_params_}
}
with open(os.path.join(OUT, "binary_logreg_vs_svm.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print("Saved metrics to binary_logreg_vs_svm.json")
