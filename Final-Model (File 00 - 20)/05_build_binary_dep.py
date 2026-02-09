# 05_build_binary_dep.py
import os, glob, re, json, pandas as pd, numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

# 1) paths (adjust if your Kaggle cache is elsewhere)
RAW_BASE = r"C:\Users\huyng\.cache\kagglehub\datasets\entenam\reddit-mental-health-dataset\versions\1\Original Reddit Data\raw data"
OUT_DIR  = r"C:\Users\huyng\Documents\sdsu fall 2025\ds577_data_binary"
os.makedirs(OUT_DIR, exist_ok=True)

# 2) find files
pos_files = glob.glob(os.path.join(RAW_BASE, "**", "dep*.csv"), recursive=True)
neg_globs = [
    os.path.join(RAW_BASE, "**", "anx*.csv"),
    os.path.join(RAW_BASE, "**", "lon*.csv"),
    os.path.join(RAW_BASE, "**", "mh*.csv"),
]
neg_files = []
for g in neg_globs:
    neg_files.extend(glob.glob(g, recursive=True))
# ethics: skip self-harm "sw*" shard for now
print("pos (dep*) files:", len(pos_files), "neg files:", len(neg_files))

# 3) robust loader -> normalize to ['title','body','author','created_utc']
def normalize_cols(df):
    cols = {c.lower(): c for c in df.columns}
    def get(*names):
        for n in names:
            if n in cols: return df[cols[n]]
        return pd.Series([None]*len(df))
    title = get("title")
    body  = get("selftext","body","text","content")
    author = get("author","user","username")
    created = get("created_utc","created","timestamp","date")
    out = pd.DataFrame({
        "title": title.astype(str).fillna(""),
        "body":  body.astype(str).fillna(""),
        "author": author.astype(str),
        "created_utc_raw": created
    })
    # try to coerce timestamps
    out["created_utc"] = pd.to_datetime(out["created_utc_raw"], errors="coerce", unit="s")
    # if not epoch seconds, try generic parsing
    mask_na = out["created_utc"].isna() & out["created_utc_raw"].notna()
    if mask_na.any():
        out.loc[mask_na, "created_utc"] = pd.to_datetime(out.loc[mask_na, "created_utc_raw"], errors="coerce")
    return out

def load_many(file_list, label):
    dfs = []
    for f in file_list:
        try:
            d = pd.read_csv(f)
            d = normalize_cols(d)
            d["label"] = label
            d["__src"] = os.path.relpath(f, RAW_BASE)
            dfs.append(d)
        except Exception as e:
            print("skip:", f, "->", e)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

pos = load_many(pos_files, 1)
neg = load_many(neg_files, 0)
df  = pd.concat([pos, neg], ignore_index=True)
print("raw combined:", df.shape, "positives:", int((df["label"]==1).sum()))

# 4) build text, basic clean, filter
def basic_text_clean(s):
    s = str(s)
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[@#]\w+", " ", s)
    s = re.sub(r"&amp;", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

df["text"] = (df["title"].fillna("") + " " + df["body"].fillna("")).str.strip()
df["text"] = df["text"].map(basic_text_clean)
df = df[df["text"].str.len() > 5].drop_duplicates(subset=["text"])
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
print("after clean/dedup:", df.shape)

# 5) prepare splits
has_author = df["author"].notna().any()
has_time = df["created_utc"].notna().any()

def split_author_disjoint_time_aware(data):
    # (a) time-aware test: latest 10% by time (if available)
    if has_time:
        cutoff = data["created_utc"].quantile(0.9)
        test = data[data["created_utc"] >= cutoff]
        rest = data[data["created_utc"] < cutoff]
    else:
        # no time; simple 10% stratified as test
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        idx_train, idx_test = next(sss.split(data, data["label"]))
        rest = data.iloc[idx_train]
        test = data.iloc[idx_test]

    # (b) author-disjoint between train/val (if author exists)
    if has_author and rest["author"].notna().any():
        # ensure authors in test are removed from rest
        test_auth = set(test["author"].dropna().unique())
        rest2 = rest[~rest["author"].isin(test_auth)].copy()

        # 20% of remaining as val using GroupShuffleSplit on author
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        idx_tr, idx_val = next(gss.split(rest2, groups=rest2["author"].fillna("na")))
        train = rest2.iloc[idx_tr]
        val = rest2.iloc[idx_val]
    else:
        # fallback: stratified split
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        idx_tr, idx_val = next(sss2.split(rest, rest["label"]))
        train = rest.iloc[idx_tr]
        val   = rest.iloc[idx_val]

    return train, val, test

train, val, test = split_author_disjoint_time_aware(df)
print("splits:", train.shape, val.shape, test.shape)

# 6) save minimal columns for modeling
keep = ["text", "label", "author", "created_utc", "__src"]
train[keep].to_csv(os.path.join(OUT_DIR, "binary_train.csv"), index=False)
val[keep].to_csv(os.path.join(OUT_DIR, "binary_val.csv"), index=False)
test[keep].to_csv(os.path.join(OUT_DIR, "binary_test.csv"), index=False)

stats = {
    "rows": {"train": len(train), "val": len(val), "test": len(test)},
    "positives": {
        "train": int((train["label"]==1).sum()),
        "val":   int((val["label"]==1).sum()),
        "test":  int((test["label"]==1).sum()),
    },
    "has_author": bool(has_author),
    "has_time": bool(has_time),
}
with open(os.path.join(OUT_DIR, "binary_build_stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2, default=str)
print("saved to:", OUT_DIR, "| stats:", stats)
