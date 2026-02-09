import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================== PATHS (adapted to your setup) ==================

BASE = r"C:\Users\huyng\Documents\sdsu fall 2025"

# Binary, preprocessed v2_no_stop
BIN_DIR = os.path.join(BASE, "ds577_data_binary_proc", "v2_no_stop")
BIN_TRAIN = os.path.join(BIN_DIR, "train.csv")
BIN_VAL   = os.path.join(BIN_DIR, "val.csv")
BIN_TEST  = os.path.join(BIN_DIR, "test.csv")

# 4-class labelled data
LABEL_TRAIN = os.path.join(BASE, "ds577_data", "labelled_train.csv")

OUT_DIR = os.path.join(BASE, "ds577_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Column names based on your check
BIN_TEXT_COL  = "text_proc"   # in binary *_proc CSVs
BIN_LABEL_COL = "label"       # 0/1
LAB_TEXT_COL  = "text"        # in labelled_train.csv
LAB_LABEL_COL = "label"       # 4-class labels


# ================== HELPERS ==================

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def lengths(series: pd.Series) -> pd.Series:
    """Character lengths of text column."""
    return series.fillna("").astype(str).str.len()


# ================== LOAD DATA ==================

print("[LOAD] binary train/val/test…")
bin_train = load_csv(BIN_TRAIN)
bin_val   = load_csv(BIN_VAL)
bin_test  = load_csv(BIN_TEST)

print("[LOAD] 4-class labelled train…")
lab_train = load_csv(LABEL_TRAIN)

# Basic sanity checks
for name, df, col in [
    ("bin_train", bin_train, BIN_TEXT_COL),
    ("bin_val",   bin_val,   BIN_TEXT_COL),
    ("bin_test",  bin_test,  BIN_TEXT_COL),
]:
    if col not in df.columns:
        raise RuntimeError(f"{name} is missing column '{col}'. Columns={list(df.columns)}")

for name, df, col in [
    ("lab_train", lab_train, LAB_TEXT_COL),
]:
    if col not in df.columns:
        raise RuntimeError(f"{name} is missing column '{col}'. Columns={list(df.columns)}")


# ================== COMPUTE LENGTHS ==================

len_train_bin = lengths(bin_train[BIN_TEXT_COL])
len_val_bin   = lengths(bin_val[BIN_TEXT_COL])
len_test_bin  = lengths(bin_test[BIN_TEXT_COL])

len_lab_train = lengths(lab_train[LAB_TEXT_COL])

# Cap x-axis for nicer plots (e.g., 0–3000 chars)
MAX_LEN = 3000

# ================== FIGURE TL1: Binary train/val/test ==================

plt.figure(figsize=(8, 5))
bins = np.linspace(0, MAX_LEN, 80)

plt.hist(len_train_bin.clip(upper=MAX_LEN), bins=bins, alpha=0.5, label="Train", density=True)
plt.hist(len_val_bin.clip(upper=MAX_LEN),   bins=bins, alpha=0.5, label="Validation", density=True)
plt.hist(len_test_bin.clip(upper=MAX_LEN),  bins=bins, alpha=0.5, label="Test", density=True)

plt.xlabel("Post length (characters)")
plt.ylabel("Density")
plt.title("TL1 – Text length distribution (binary train/val/test)")
plt.legend()
plt.tight_layout()
tl1_path = os.path.join(OUT_DIR, "TL1_binary_len_hist.png")
plt.savefig(tl1_path, dpi=300)
plt.close()
print("[SAVE]", tl1_path)


# ================== FIGURE TL2: 4-class labelled train ==================

plt.figure(figsize=(8, 5))
bins = np.linspace(0, MAX_LEN, 80)

plt.hist(len_lab_train.clip(upper=MAX_LEN), bins=bins, color="gray", alpha=0.8, density=True)

plt.xlabel("Post length (characters)")
plt.ylabel("Density")
plt.title("TL2 – Text length distribution (4-class labelled train)")
plt.tight_layout()
tl2_path = os.path.join(OUT_DIR, "TL2_labelled_len_hist.png")
plt.savefig(tl2_path, dpi=300)
plt.close()
print("[SAVE]", tl2_path)


# ================== FIGURE TL3: Binary train by label (boxplot) ==================

# Map binary labels to readable names
label_map = {0: "Not depressed", 1: "Depressed"}
bin_train_named = bin_train.copy()
bin_train_named["label_name"] = bin_train_named[BIN_LABEL_COL].map(label_map)

data_not_dep = lengths(bin_train_named.loc[bin_train_named["label_name"] == "Not depressed", BIN_TEXT_COL])
data_dep     = lengths(bin_train_named.loc[bin_train_named["label_name"] == "Depressed",     BIN_TEXT_COL])

plt.figure(figsize=(6, 5))
plt.boxplot(
    [data_not_dep.clip(upper=MAX_LEN), data_dep.clip(upper=MAX_LEN)],
    labels=["Not depressed", "Depressed"],
    showfliers=False,
)
plt.ylabel("Post length (characters)")
plt.title("TL3 – Binary train text lengths by class")
plt.tight_layout()
tl3_path = os.path.join(OUT_DIR, "TL3_binary_len_by_label_box.png")
plt.savefig(tl3_path, dpi=300)
plt.close()
print("[SAVE]", tl3_path)


# ================== FIGURE TL4: Binary train CDF ==================

plt.figure(figsize=(8, 5))

# Sort and build empirical CDF
sorted_len = np.sort(len_train_bin.values)
y = np.arange(1, len(sorted_len) + 1) / len(sorted_len)

plt.plot(sorted_len, y, linewidth=1.5)
plt.xlim(0, MAX_LEN)
plt.xlabel("Post length (characters)")
plt.ylabel("Cumulative proportion of posts")
plt.title("TL4 – CDF of text length (binary train)")
plt.grid(alpha=0.3)
plt.tight_layout()
tl4_path = os.path.join(OUT_DIR, "TL4_binary_len_cdf.png")
plt.savefig(tl4_path, dpi=300)
plt.close()
print("[SAVE]", tl4_path)

print("Done: TL1–TL4 figures created in:", OUT_DIR)
