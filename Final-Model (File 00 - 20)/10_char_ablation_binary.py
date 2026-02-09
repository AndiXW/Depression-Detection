# 10_char_ablation_binary.py
import os, json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score, classification_report, confusion_matrix

PROC_BIN = r"C:\Users\huyng\Documents\sdsu fall 2025\ds577_data_binary_proc\v2_no_stop"
OUT = r"C:\Users\huyng\Documents\sdsu fall 2025\ds577_metrics"
os.makedirs(OUT, exist_ok=True)

# Load splits
train = pd.read_csv(os.path.join(PROC_BIN, "train.csv"))
val   = pd.read_csv(os.path.join(PROC_BIN, "val.csv"))
test  = pd.read_csv(os.path.join(PROC_BIN, "test.csv"))
for d in (train, val, test):
    d["text_proc"] = d["text_proc"].astype(str)

# Train+Val pool
tv = pd.concat([train[["text_proc","label"]], val[["text_proc","label"]]], ignore_index=True)

# --- Model A: word-only (your locked baseline) ---
pipe_word = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=5, max_df=0.9,
                              max_features=200_000, sublinear_tf=True, dtype=np.float32)),
    ("clf", LogisticRegression(max_iter=2000, solver="saga", class_weight="balanced", C=0.5))
])

# --- Model B: word + char ---
vec_word = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=5, max_df=0.9,
                           max_features=150_000, sublinear_tf=True, dtype=np.float32)
vec_char = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=5, max_df=0.9,
                           max_features=80_000, sublinear_tf=True, dtype=np.float32)
union = FeatureUnion([("w", vec_word), ("c", vec_char)])
pipe_wchar = Pipeline([
    ("vec", union),
    ("clf", LogisticRegression(max_iter=2000, solver="saga", class_weight="balanced", C=0.5))
])

def eval_on_test(pipe, name):
    pipe.fit(tv["text_proc"], tv["label"])
    Xte, yte = test["text_proc"], test["label"]
    pred = pipe.predict(Xte)
    # LR is probabilistic -> probs for ranking metrics
    if name == "word":
        probs = pipe.named_steps["clf"].predict_proba(pipe.named_steps["tfidf"].transform(Xte))[:,1]
    else:
        # vec is FeatureUnion in this branch
        probs = pipe.named_steps["clf"].predict_proba(pipe.named_steps["vec"].transform(Xte))[:,1]
    f1 = f1_score(yte, pred, average="macro")
    acc = accuracy_score(yte, pred)
    pr  = average_precision_score(yte, probs)
    roc = roc_auc_score(yte, probs)
    print(f"[{name}] TEST F1={f1:.3f} ACC={acc:.3f} PR-AUC={pr:.3f} ROC-AUC={roc:.3f}")
    return {"name": name, "f1": float(f1), "acc": float(acc), "pr_auc": float(pr), "roc_auc": float(roc)}

res_word  = eval_on_test(pipe_word,  "word")
res_wchar = eval_on_test(pipe_wchar, "word+char")

# Save a tiny report
with open(os.path.join(OUT, "binary_char_ablation.json"), "w", encoding="utf-8") as f:
    json.dump({"word": res_word, "word_char": res_wchar}, f, indent=2)

print("Saved metrics to binary_char_ablation.json")
