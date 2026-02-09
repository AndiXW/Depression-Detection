"""
18_per_class_performance_fig.py

Per-class performance on the 4-class labelled Reddit subset.

- Loads ds577_data/labelled_train.csv  (text, label)
- Splits into train/test (stratified)
- Trains a TF-IDF + Logistic Regression baseline
- Computes per-class F1-scores on the test split
- Saves a bar chart figure: figures/eval_per_class_f1_4class.png
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
BASE_DIR = Path(r"C:\Users\huyng\Documents\sdsu fall 2025")
DATA_DIR = BASE_DIR / "ds577_data"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

LABELLED_TRAIN = DATA_DIR / "labelled_train.csv"

# --------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------
if not LABELLED_TRAIN.exists():
    raise FileNotFoundError(f"Could not find {LABELLED_TRAIN}. "
                            f"Check the path or filename.")

df = pd.read_csv(LABELLED_TRAIN)

# Expect columns: 'text', 'label'
if "text" not in df.columns or "label" not in df.columns:
    raise ValueError(f"Expected columns ['text', 'label'] in {LABELLED_TRAIN}, "
                     f"got {df.columns.tolist()}")

print("[DATA] labelled_train.csv shape:", df.shape)
print("[DATA] label distribution:")
print(df["label"].value_counts())

X = df["text"]
y = df["label"]

# --------------------------------------------------------------------
# Train/test split (stratified)
# --------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"[SPLIT] train={len(X_train)}, test={len(X_test)}")

# --------------------------------------------------------------------
# Model: TF-IDF + Logistic Regression (multiclass)
# --------------------------------------------------------------------
pipeline = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                min_df=5,
                ngram_range=(1, 1)
            )
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=200,
                C=2.0,
                multi_class="auto"
            )
        ),
    ]
)

print("[TRAIN] Fitting TF-IDF + Logistic Regression...")
pipeline.fit(X_train, y_train)

# --------------------------------------------------------------------
# Evaluation: per-class F1 on test set
# --------------------------------------------------------------------
y_pred = pipeline.predict(X_test)

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

# Extract per-class F1-scores (ignore 'accuracy', 'macro avg', 'weighted avg')
class_labels = []
f1_scores = []

for label, metrics in report.items():
    if label in ("accuracy", "macro avg", "weighted avg"):
        continue
    class_labels.append(label)
    f1_scores.append(metrics["f1-score"])

# Sort by F1 descending for nicer plotting
class_labels = np.array(class_labels)
f1_scores = np.array(f1_scores)

order = np.argsort(f1_scores)[::-1]
class_labels = class_labels[order]
f1_scores = f1_scores[order]

print("[RESULTS] Per-class F1 on test split:")
for lbl, f1 in zip(class_labels, f1_scores):
    print(f"  {lbl:20s}  F1 = {f1:.3f}")

# --------------------------------------------------------------------
# Plot: bar chart of per-class F1 (with custom colors)
# --------------------------------------------------------------------
plt.figure(figsize=(7, 6))

# simple nice color palette (4 colors)
palette = ["#E8DD0C", "#3CA053", "#D81118", "#F007D5"]
colors = palette[: len(class_labels)]

bars = plt.bar(class_labels, f1_scores, color=colors)

plt.ylim(0, 1.0)
plt.ylabel("F1-score")
plt.title("Per-class F1 on 4-class labelled subset")

# Annotate bars with F1 values
for bar, f1 in zip(bars, f1_scores):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.02,
        f"{f1:.2f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()

out_path = FIG_DIR / "eval_per_class_f1_4class.png"
plt.savefig(out_path, dpi=300)
plt.close()

print(f"[SAVED] {out_path}")
