import os
import re
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------------------------------------------
# Paths & constants
# -------------------------------------------------------------------
BASE_DIR = r"."
BIN_PROC_DIR = os.path.join(BASE_DIR, "ds577_data_binary_proc", "v2_no_stop")
FIG_DIR = os.path.join(BASE_DIR, "figs")

os.makedirs(FIG_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(BIN_PROC_DIR, "train.csv")
VAL_PATH = os.path.join(BIN_PROC_DIR, "val.csv")
TEST_PATH = os.path.join(BIN_PROC_DIR, "test.csv")

TEXT_COL = "text_proc"
LABEL_COL = "label"

MAX_SNIPPET_LEN = 120  # characters

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def sanitize_text(text: str, max_len: int = MAX_SNIPPET_LEN) -> str:
    """Heavily anonymize and truncate a Reddit post for the poster."""
    if not isinstance(text, str):
        text = str(text)

    # Remove newlines and excessive whitespace
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Anonymize URLs, usernames, subreddits (basic heuristics)
    text = re.sub(r"http\S+", "[link]", text)
    text = re.sub(r"u/[A-Za-z0-9_\-]+", "[user]", text)
    text = re.sub(r"r/[A-Za-z0-9_\-]+", "[subreddit]", text)

    # Truncate
    if len(text) > max_len:
        text = text[: max_len - 3].rstrip() + "..."

    return text


def label_to_str(y: int) -> str:
    """Map numeric label to human-readable text."""
    return "Depressed" if int(y) == 1 else "Not depressed"


def pick_example(df: pd.DataFrame, mask: np.ndarray, sort_by: str, ascending: bool):
    """Pick one example row from df[mask], sorted by sort_by."""
    sub = df[mask].sort_values(sort_by, ascending=ascending)
    if sub.empty:
        return None
    return sub.iloc[0]


# -------------------------------------------------------------------
# 1) Load data
# -------------------------------------------------------------------
print("[LOAD] binary train/val/test from:", BIN_PROC_DIR)
train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print("[SHAPE] train:", train_df.shape, " val:", val_df.shape, " test:", test_df.shape)
print("[COLS] train:", list(train_df.columns))

# -------------------------------------------------------------------
# 2) Train a quick logistic regression baseline on train + val
# -------------------------------------------------------------------
train_all = pd.concat([train_df, val_df], ignore_index=True)

X_train = train_all[TEXT_COL].astype(str)
y_train = train_all[LABEL_COL].astype(int)

X_test = test_df[TEXT_COL].astype(str)
y_test = test_df[LABEL_COL].astype(int)

print("[TRAIN] Fitting TF-IDF + LogisticRegression on train+val...")
clf = make_pipeline(
    TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=5,
    ),
    LogisticRegression(
        max_iter=1000,
        C=0.5,
        n_jobs=-1,
        class_weight="balanced",
    ),
)
clf.fit(X_train, y_train)

print("[PREDICT] On test set...")
proba_test = clf.predict_proba(X_test)[:, 1]
pred_test = (proba_test >= 0.5).astype(int)

test_copy = test_df.copy()
test_copy["y_true"] = y_test
test_copy["y_pred"] = pred_test
test_copy["p_dep"] = proba_test
test_copy["snippet"] = test_copy[TEXT_COL].apply(sanitize_text)

# -------------------------------------------------------------------
# 3) Select qualitative examples
# -------------------------------------------------------------------
mask_tp_dep = (test_copy["y_true"] == 1) & (test_copy["y_pred"] == 1)
mask_fn_dep = (test_copy["y_true"] == 1) & (test_copy["y_pred"] == 0)
mask_fp_dep = (test_copy["y_true"] == 0) & (test_copy["y_pred"] == 1)
mask_tn_ok  = (test_copy["y_true"] == 0) & (test_copy["y_pred"] == 0)

examples = []

# 1) Clear depressed TP (high p_dep)
row = pick_example(test_copy, mask_tp_dep, sort_by="p_dep", ascending=False)
if row is not None:
    examples.append({
        "Post snippet (shortened)": row["snippet"],
        "True label": label_to_str(row["y_true"]),
        "Model prediction": label_to_str(row["y_pred"]),
        "Comment": "Correct, clear depressive signal",
    })

# 2) Missed depressed FN (very low p_dep)
row = pick_example(test_copy, mask_fn_dep, sort_by="p_dep", ascending=True)
if row is not None:
    examples.append({
        "Post snippet (shortened)": row["snippet"],
        "True label": label_to_str(row["y_true"]),
        "Model prediction": label_to_str(row["y_pred"]),
        "Comment": "Missed subtle expression of depression",
    })

# 3) Over-sensitive FP (high p_dep but true=Not dep)
row = pick_example(test_copy, mask_fp_dep, sort_by="p_dep", ascending=False)
if row is not None:
    examples.append({
        "Post snippet (shortened)": row["snippet"],
        "True label": label_to_str(row["y_true"]),
        "Model prediction": label_to_str(row["y_pred"]),
        "Comment": "Over-sensitive to depressive keywords/context",
    })

# 4) Confident Not-depressed TN (low p_dep)
row = pick_example(test_copy, mask_tn_ok, sort_by="p_dep", ascending=True)
if row is not None:
    examples.append({
        "Post snippet (shortened)": row["snippet"],
        "True label": label_to_str(row["y_true"]),
        "Model prediction": label_to_str(row["y_pred"]),
        "Comment": "Correct, everyday stress but not depression",
    })

examples_df = pd.DataFrame(examples)
print("\n[EXAMPLES]")
print(examples_df)

# -------------------------------------------------------------------
# 4) Save CSV for you to review/edit
# -------------------------------------------------------------------
csv_path = os.path.join(FIG_DIR, "qual_examples_table.csv")
examples_df.to_csv(csv_path, index=False, encoding="utf-8")
print(f"\n[WRITE] Saved qualitative examples CSV -> {csv_path}")

# -------------------------------------------------------------------
# 5) Render a poster-ready PNG table
# -------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 14,
})

fig_height = 1.0 + 1.5 * len(examples_df)
fig, ax = plt.subplots(figsize=(9, fig_height))
ax.axis("off")

table = ax.table(
    cellText=examples_df.values,
    colLabels=examples_df.columns,
    loc="center",
    cellLoc="left",
)

table.auto_set_font_size(False)
table.set_fontsize(14)
table.auto_set_column_width(col=list(range(len(examples_df.columns))))

# Make header row bold/grey
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#DDDDDD")

plt.tight_layout()

png_path = os.path.join(FIG_DIR, "fig_E7_qual_examples.png")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"[WRITE] Saved qualitative examples PNG -> {png_path}")
print("\nDone. You can open the PNG and drop it into your poster.")
