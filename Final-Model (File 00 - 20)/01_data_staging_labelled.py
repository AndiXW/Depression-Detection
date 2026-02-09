# 01_data_staging_labelled.py
import os, glob
import pandas as pd

# 1) Point to your local KaggleHub cache (from your earlier printout)
BASE = r"C:\Users\huyng\.cache\kagglehub\datasets\entenam\reddit-mental-health-dataset\versions\1"
LAB_DIR = os.path.join(BASE, "Original Reddit Data", "Labelled Data")

# 2) If names differ, we’ll auto-discover all labelled CSVs
labelled_files = sorted(glob.glob(os.path.join(LAB_DIR, "*.csv")))
print("Labelled files:", [os.path.basename(f) for f in labelled_files])

dfs = []
for f in labelled_files:
    try:
        df = pd.read_csv(f)
        df["__source_file"] = os.path.basename(f)
        dfs.append(df)
    except Exception as e:
        print("Skip", f, "->", e)

if not dfs:
    raise SystemExit("No labelled CSVs found. Check LAB_DIR.")

lbl = pd.concat(dfs, ignore_index=True)

# 3) Normalize columns + build a single 'text' feature
rename_map = {"selftext": "body", "Label": "label"}
for k, v in rename_map.items():
    if k in lbl.columns:
        lbl = lbl.rename(columns={k: v})

lbl["title"] = lbl.get("title", "").fillna("")
lbl["body"]  = lbl.get("body", "").fillna("")
lbl["text"]  = (lbl["title"].astype(str) + " " + lbl["body"].astype(str)).str.strip()

# 4) Basic filtering and dedup
lbl = lbl.dropna(subset=["text"])
lbl = lbl[lbl["text"].str.len() > 5]
lbl = lbl.drop_duplicates(subset=["text"])

print("Shape after clean:", lbl.shape)
print("Label counts:\n", lbl.get("label").value_counts(dropna=False).head(20))

# --- normalize labels (case / spacing) ---
import re

lbl["label"] = lbl["label"].astype(str)

lbl["label_norm"] = (
    lbl["label"]
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
    .str.lower()
)

canon_map = {
    "drug and alcohol": "Drug and Alcohol",
    "personality": "Personality",
    "trauma and stress": "Trauma and Stress",
    "early life": "Early Life",
}

lbl["label"] = lbl["label_norm"].map(canon_map).fillna(
    lbl["label_norm"].str.title()
)

lbl = lbl.drop(columns=["label_norm"])
print("After normalization:\n", lbl["label"].value_counts())

# 5) Keep only what we’ll model on (leakage-safe for now)
feature_df = lbl[["text", "label"]].copy()

# 6) Stratified train/val/test
from sklearn.model_selection import train_test_split
y = feature_df["label"]
train_val, test = train_test_split(feature_df, test_size=0.2, stratify=y, random_state=42)
y_tv = train_val["label"]
train, val = train_test_split(train_val, test_size=0.25, stratify=y_tv, random_state=42)  # 0.8*0.25=0.2

print("Splits:", train.shape, val.shape, test.shape)

# 7) Save to your project folder
OUT_DIR = r"C:\Users\huyng\Documents\sdsu fall 2025\ds577_data"
os.makedirs(OUT_DIR, exist_ok=True)
train.to_csv(os.path.join(OUT_DIR, "labelled_train.csv"), index=False)
val.to_csv(os.path.join(OUT_DIR, "labelled_val.csv"), index=False)
test.to_csv(os.path.join(OUT_DIR, "labelled_test.csv"), index=False)
print("Saved to:", OUT_DIR)



