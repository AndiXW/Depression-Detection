# 06_preprocess_binary.py
import os, re, json, pandas as pd

BIN_DIR = r"C:\Users\huyng\Documents\sdsu fall 2025\ds577_data_binary"
OUT_DIR = r"C:\Users\huyng\Documents\sdsu fall 2025\ds577_data_binary_proc"
VARIANT = "v2_no_stop"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, VARIANT), exist_ok=True)

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
STOP = set(ENGLISH_STOP_WORDS)

def v1_basic_clean(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[@#]\w+", " ", s)
    s = re.sub(r"&amp;", " ", s)
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def v2_no_stop_clean(s: str) -> str:
    s = v1_basic_clean(s)
    toks = s.split()
    toks = [t for t in toks if t not in STOP]
    return " ".join(toks)

CLEAN = v2_no_stop_clean  # chosen winner

def load(name):
    p = os.path.join(BIN_DIR, f"binary_{name}.csv")
    df = pd.read_csv(p)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{p} missing text/label")
    return df

def apply_and_save(df, name):
    out = df.copy()
    out["text_proc"] = out["text"].map(CLEAN)
    out = out[out["text_proc"].str.len() > 0]
    keep = ["text_proc", "label", "author", "created_utc", "__src"]
    for k in keep:
        if k not in out.columns: out[k] = None
    out = out[keep]
    sub = os.path.join(OUT_DIR, VARIANT)
    out.to_csv(os.path.join(sub, f"{name}.csv"), index=False)
    return out

train = load("train"); val = load("val"); test = load("test")
t = apply_and_save(train, "train")
v = apply_and_save(val, "val")
te = apply_and_save(test, "test")

stats = {
    "variant": VARIANT,
    "rows": {"train": len(t), "val": len(v), "test": len(te)},
    "avg_len_chars": {
        "train": float(t["text_proc"].str.len().mean()),
        "val": float(v["text_proc"].str.len().mean()),
        "test": float(te["text_proc"].str.len().mean())
    }
}
with open(os.path.join(OUT_DIR, VARIANT, "preprocess_stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)
print("Saved to:", os.path.join(OUT_DIR, VARIANT), "| stats:", stats)
