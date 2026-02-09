# 15_model_comparison_figs.py
"""
Create bar charts comparing classical models vs DistilBERT
on TEST macro-F1 and PR-AUC for the binary depression task.

Output:
  - ds577_figures/FIG6a_model_comparison_f1.png
  - ds577_figures/FIG6b_model_comparison_prauc.png
"""

import os
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(__file__)
FIG_DIR = os.path.join(BASE_DIR, "ds577_figures")
os.makedirs(FIG_DIR, exist_ok=True)

# === IMPORTANT: metrics dictionary ===
# Fill these from your actual runs.
# I pre-filled the classical ones from your earlier logs.

metrics = {
    "LogReg (word TF-IDF)": {
        "f1": 0.7569,
        "pr_auc": 0.7613,
    },
    "LogReg (word+char TF-IDF)": {
        "f1": 0.7580,
        "pr_auc": 0.7640,
    },
    "DistilBERT": {
        # TODO: update these based on ds577_transformers/distilbert_binary_gpu/results.json
        # Put your real test macro-F1 & PR-AUC here!
        "f1": 0.80,
        "pr_auc": 0.85,
    },
}

methods = list(metrics.keys())
f1_vals = [metrics[m]["f1"] for m in methods]
pr_vals = [metrics[m]["pr_auc"] for m in methods]

x = np.arange(len(methods))
width = 0.6

# --- Figure 6a: macro-F1 ---
plt.figure(figsize=(6, 4), dpi=200)
bars = plt.bar(x, f1_vals, width)
plt.xticks(x, methods, rotation=20, ha="right")
plt.ylabel("Test Macro-F1")
plt.ylim(0.6, 1.0)
plt.title("Classical TF-IDF Models vs DistilBERT (Macro-F1)")

for b, v in zip(bars, f1_vals):
    plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
             f"{v:.3f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
out_f1 = os.path.join(FIG_DIR, "FIG6a_model_comparison_f1.png")
plt.savefig(out_f1)
plt.close()
print(f"[SAVE] {out_f1}")

# --- Figure 6b: PR-AUC ---
plt.figure(figsize=(6, 4), dpi=200)
bars = plt.bar(x, pr_vals, width)
plt.xticks(x, methods, rotation=20, ha="right")
plt.ylabel("Test PR-AUC")
plt.ylim(0.6, 1.0)
plt.title("Classical TF-IDF Models vs DistilBERT (PR-AUC)")

for b, v in zip(bars, pr_vals):
    plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
             f"{v:.3f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
out_pr = os.path.join(FIG_DIR, "FIG6b_model_comparison_prauc.png")
plt.savefig(out_pr)
plt.close()
print(f"[SAVE] {out_pr}")
