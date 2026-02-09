# 03_preprocess_and_save.py
import os, re, json
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

DATA_DIR = r"C:\Users\huyng\Documents\sdsu fall 2025\ds577_data"
OUT_DIR  = r"C:\Users\huyng\Documents\sdsu fall 2025\ds577_data_proc"
os.makedirs(OUT_DIR, exist_ok=True)

def v1_basic_clean(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)          # URLs
    s = re.sub(r"[@#]\w+", " ", s)                   # @user, #hashtag
    s = re.sub(r"&amp;", " ", s)                     # html entity
    s = re.sub(r"[^\w\s']", " ", s)                  # keep letters/digits/_/space/'
    s = re.sub(r"\s+", " ", s).strip()
    return s

STOP = set(ENGLISH_STOP_WORDS)
def v2_no_stop_clean(s: str) -> str:
    s = v1_basic_clean(s)
    tokens = s.split()
    tokens = [t for t in tokens if t not in STOP]
    return " ".join(tokens)

VARIANTS = {
    "v1_basic": v1_basic_clean,
    "v2_no_stop": v2_no_stop_clean,
}

def load_split(name):
    path = os.path.join(DATA_DIR, f"labelled_{name}.csv")
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} missing required columns text/label")
    return df

def save_variant(df_train, df_val, df_test, variant_name, func):
    def apply(df):
        out = df.copy()
        out["text_proc"] = df["text"].map(func)
        # drop empties after cleaning
        out = out[out["text_proc"].str.len() > 0]
        return out[["text_proc", "label"]]
    t, v, te = apply(df_train), apply(df_val), apply(df_test)

    subdir = os.path.join(OUT_DIR, variant_name)
    os.makedirs(subdir, exist_ok=True)
    t.to_csv(os.path.join(subdir, "train.csv"), index=False)
    v.to_csv(os.path.join(subdir, "val.csv"), index=False)
    te.to_csv(os.path.join(subdir, "test.csv"), index=False)

    stats = {
        "rows": {
            "train": int(len(t)),
            "val":   int(len(v)),
            "test":  int(len(te)),
        },
        "avg_len_chars": {
            "train": float(t["text_proc"].str.len().mean()),
            "val":   float(v["text_proc"].str.len().mean()),
            "test":  float(te["text_proc"].str.len().mean()),
        }
    }
    with open(os.path.join(subdir, "preprocess_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[{variant_name}] saved → {subdir} ; stats:", stats)

if __name__ == "__main__":
    train = load_split("train")
    val   = load_split("val")
    test  = load_split("test")

    for name, func in VARIANTS.items():
        save_variant(train, val, test, name, func)

    print("Done. All processed splits are under:", OUT_DIR)
