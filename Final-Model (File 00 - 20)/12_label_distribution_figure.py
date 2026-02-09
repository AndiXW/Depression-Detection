# 12_label_distribution_figure.py
#
# Figures from binary depression dataset:
#   FIG 1: Label distribution from processed data (v2_no_stop)
#   FIG 2: Post length distribution by label from raw binary data

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- PATHS ----------
# Processed (only labels) - for clean label distribution
BASE_DIR_PROC = r"C:\Users\huyng\Documents\sdsu fall 2025\ds577_data_binary_proc\v2_no_stop"
# Raw (has text + label) - for richer EDA
BASE_DIR_RAW = r"C:\Users\huyng\Documents\sdsu fall 2025\ds577_data_binary"

files_proc = ["train.csv", "val.csv", "test.csv"]

# =============================================================================
# FIGURE 1: LABEL DISTRIBUTION FROM PROCESSED DATA
# =============================================================================

parts = []

for fname in files_proc:
    fpath = os.path.join(BASE_DIR_PROC, fname)
    print(f"[FIG 1] Loading {fpath} ...")
    df_part = pd.read_csv(fpath, usecols=["label"])
    parts.append(df_part)

df_proc = pd.concat(parts, ignore_index=True)
print("[FIG 1] Combined shape:", df_proc.shape)
print("[FIG 1] Columns:", df_proc.columns.tolist())

label_counts = df_proc["label"].value_counts().sort_values(ascending=False)
label_props = df_proc["label"].value_counts(normalize=True).sort_values(ascending=False) * 100

print("\n[FIG 1] Label counts:")
print(label_counts)
print("\n[FIG 1] Label percentages:")
print(label_props.round(2))

plt.figure(figsize=(5.5, 5))

x = np.arange(len(label_counts))
bar_width = 0.5  # wide bars -> smaller gap

bars = plt.bar(x, label_counts.values, width=bar_width)

plt.xlabel("Label", fontsize=12)
plt.ylabel("Number of posts", fontsize=12)
plt.title("Label distribution: Depression vs Not-Depression", fontsize=13)
plt.xticks(x, label_counts.index.astype(str))

for xi, bar, pct in zip(x, bars, label_props.values):
    y = bar.get_height()
    plt.text(
        xi,
        y,
        f"{pct:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.tight_layout()

out_path1 = os.path.join(BASE_DIR_PROC, "fig_binary_label_distribution.png")
plt.savefig(out_path1, dpi=300)
print(f"\nSaved FIGURE 1 to:\n{out_path1}")

plt.show()

# =============================================================================
# FIGURE 2: POST LENGTH DISTRIBUTION BY LABEL (FROM RAW DATA)
# =============================================================================

if not os.path.isdir(BASE_DIR_RAW):
    print(f"\n[FIG 2] Raw directory does not exist: {BASE_DIR_RAW}")
else:
    raw_files = [f for f in os.listdir(BASE_DIR_RAW) if f.lower().endswith(".csv")]
    if not raw_files:
        print(f"\n[FIG 2] No CSV files found in {BASE_DIR_RAW}; skipping FIGURE 2.")
    else:
        print(f"\n[FIG 2] Found CSVs in raw dir: {raw_files}")

        parts_raw = []
        candidate_text_cols = ["cleaned_text", "text", "selftext", "body", "title"]
        chosen_text_col = None

        for fname in raw_files:
            fpath = os.path.join(BASE_DIR_RAW, fname)
            print(f"[FIG 2] Inspecting {fpath} ...")

            try:
                df_head = pd.read_csv(fpath, nrows=5)
            except Exception as e:
                print(f"[FIG 2] Could not read {fname} (head). Skipping. Error: {e}")
                continue

            cols = df_head.columns.tolist()
            print(f"[FIG 2] Columns in {fname}:", cols)

            text_col = None
            for c in candidate_text_cols:
                if c in cols:
                    text_col = c
                    break

            if text_col is None:
                print(f"[FIG 2] No text-like column found in {fname}. Skipping this file.")
                continue

            # remember the first-found text column name (just for logging)
            if chosen_text_col is None:
                chosen_text_col = text_col

            usecols = []
            if "label" in cols:
                usecols.append("label")
            if text_col in cols:
                usecols.append(text_col)

            if "label" not in usecols:
                print(f"[FIG 2] File {fname} has no 'label' column. Skipping.")
                continue

            try:
                df_part = pd.read_csv(fpath, usecols=usecols)
            except Exception as e:
                print(f"[FIG 2] Could not read {fname} fully. Skipping. Error: {e}")
                continue

            df_part = df_part.dropna(subset=["label", text_col])
            parts_raw.append(df_part)

        if not parts_raw:
            print("\n[FIG 2] No usable raw CSVs with both label + text. Skipping FIGURE 2.")
        else:
            df_raw = pd.concat(parts_raw, ignore_index=True)
            print("[FIG 2] Combined raw shape:", df_raw.shape)
            print("[FIG 2] Columns:", df_raw.columns.tolist())
            if chosen_text_col is not None:
                print(f"[FIG 2] Using text column: {chosen_text_col}")

            # SAMPLE for speed
            max_rows = 200_000
            if len(df_raw) > max_rows:
                df_sample = df_raw.sample(n=max_rows, random_state=42)
                print(f"[FIG 2] Using a sample of {max_rows} rows.")
            else:
                df_sample = df_raw

            # compute character length
            text_col_for_len = chosen_text_col if chosen_text_col in df_sample.columns else candidate_text_cols[0]
            df_sample["len_chars"] = df_sample[text_col_for_len].astype(str).str.len()

            print("\n[FIG 2] Length stats (chars) by label:")
            print(df_sample.groupby("label")["len_chars"].describe())

            clip_max = 2000
            df_sample["len_clipped"] = df_sample["len_chars"].clip(0, clip_max)

            plt.figure(figsize=(7, 5))

            labels_sorted = sorted(df_sample["label"].unique())
            for lab in labels_sorted:
                subset = df_sample[df_sample["label"] == lab]["len_clipped"]
                plt.hist(
                    subset,
                    bins=50,
                    alpha=0.5,
                    label=str(lab),
                )

            plt.xlabel("Post length (characters, clipped at 2000)", fontsize=12)
            plt.ylabel("Number of posts", fontsize=12)
            plt.title("Post length distribution by label (raw binary data)", fontsize=13)
            plt.legend(title="Label")

            plt.tight_layout()

            out_path2 = os.path.join(BASE_DIR_PROC, "fig_binary_length_distribution.png")
            plt.savefig(out_path2, dpi=300)
            print(f"\nSaved FIGURE 2 to:\n{out_path2}")

            plt.show()
