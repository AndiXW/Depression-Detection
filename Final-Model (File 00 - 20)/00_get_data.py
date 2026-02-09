import os, glob
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

# 1) Download the dataset to local cache (returns a folder path)
path = kagglehub.dataset_download("entenam/reddit-mental-health-dataset")
print("Downloaded to:", path)

# 2) See what files are inside so we know what to load
files = [p.replace(path+"\\","") for p in glob.glob(os.path.join(path, "**"), recursive=True) if os.path.isfile(p)]
print("Files found:")
for f in files:
    print("  -", f)

# 3) Load a file (adjust name after you see the listing)
# Example: try to auto-pick a likely CSV/Parquet
candidates = [f for f in files if f.lower().endswith((".csv", ".parquet"))]
print("\nCandidate data files:", candidates[:10])

# If you see a specific file you want, use dataset_load for a clean, reproducible load:
# df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS,
#                             "entenam/reddit-mental-health-dataset",
#                             "<PUT_EXACT_FILE_NAME_HERE>.csv")

# For now, quick-load the first candidate:
if candidates:
    file_to_load = os.path.join(path, candidates[0])
    if file_to_load.lower().endswith(".csv"):
        df = pd.read_csv(file_to_load)
    else:
        df = pd.read_parquet(file_to_load)
    print("\nShape:", df.shape)
    print("Columns:", df.columns.tolist()[:20])
    print(df.head())
else:
    print("No CSV/Parquet files detected. Check the printed file list above.")
