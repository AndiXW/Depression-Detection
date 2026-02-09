# 13_preprocessing_comparison_figs.py
#
# Make 5 small bar charts comparing preprocessing / models.

import os
import matplotlib.pyplot as plt
import numpy as np

BASE = r"C:\Users\huyng\Documents\sdsu fall 2025"
OUT_DIR = os.path.join(BASE, "ds577_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 4-class, v1 vs v2 ----------
fourclass_models = ["v1_basic", "v2_no_stop"]
fourclass_f1  = [0.662, 0.674]
fourclass_pr  = [0.728, 0.687]

def grouped_bar(xlabels, series, series_labels, title, ylabel, filename):
    x = np.arange(len(xlabels))
    width = 0.35
    plt.figure(figsize=(5, 3.2))

    for i, (values, lab) in enumerate(zip(series, series_labels)):
        plt.bar(x + (i - 0.5) * width, values, width, label=lab)

    plt.xticks(x, xlabels, rotation=0)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 1.0)
    plt.legend(fontsize=7)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    print("Saved:", path)

grouped_bar(
    fourclass_models,
    [fourclass_f1, fourclass_pr],
    ["macro-F1", "PR-AUC"],
    "4-class topic model – preprocessing comparison",
    "Score",
    "PC1_4class_v1_vs_v2.png",
)

# ---------- Binary: word vs word+char ----------
bin_models = ["word", "word+char"]
bin_f1 = [0.757, 0.758]
bin_pr = [0.761, 0.764]

grouped_bar(
    bin_models,
    [bin_f1, bin_pr],
    ["macro-F1", "PR-AUC"],
    "Binary TF-IDF – word vs word+char",
    "Score",
    "PC2_binary_word_vs_char.png",
)

# ---------- Binary: LogReg vs SVM ----------
bin_clf = ["LogReg", "Linear SVM"]
bin_clf_f1 = [0.750, 0.736]
bin_clf_pr = [0.753, 0.739]

grouped_bar(
    bin_clf,
    [bin_clf_f1, bin_clf_pr],
    ["macro-F1", "PR-AUC"],
    "Binary TF-IDF – LogReg vs Linear SVM",
    "Score",
    "PC3_binary_logreg_vs_svm.png",
)

# ---------- DistilBERT vs baseline (you fill these) ----------

# TODO: open ds577_transformers/distilbert_binary_gpu/results.json once
# and update these values based on its TEST metrics.
# Example (FAKE numbers, REPLACE THEM):
distilbert_f1 = 0.80   # <- replace
distilbert_pr = 0.82   # <- replace

logreg_best_f1 = 0.7569  # from 07_baseline_binary.py
logreg_best_pr = 0.7613  # from 07_baseline_binary.py

models_dl = ["TF-IDF + LogReg", "DistilBERT"]
f1_dl = [logreg_best_f1, distilbert_f1]
pr_dl = [logreg_best_pr, distilbert_pr]

grouped_bar(
    models_dl,
    [f1_dl, pr_dl],
    ["macro-F1", "PR-AUC"],
    "Binary depression detection – classic vs DistilBERT",
    "Score",
    "PC4_PC5_binary_logreg_vs_distilbert.png",
)
