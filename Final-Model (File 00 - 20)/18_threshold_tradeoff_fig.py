import json
import os
import matplotlib.pyplot as plt

BASE = r"C:\Users\huyng\Documents\sdsu fall 2025"
METRICS_PATH = os.path.join(BASE, "ds577_metrics", "binary_threshold_tuned.json")
OUT_PATH = os.path.join(BASE, "ds577_figs", "18_threshold_tradeoffs.png")

with open(METRICS_PATH, "r") as f:
    m = json.load(f)

tn, fp = m["confusion_matrix"][0]
fn, tp = m["confusion_matrix"][1]

total_not_dep = tn + fp
total_dep = fn + tp

fp_rate = fp / total_not_dep
fn_rate = fn / total_dep

fig, ax = plt.subplots(figsize=(9, 9))

bars = ax.bar(
    ["False positives\n(not-dep → dep)", "False negatives\n(dep → not-dep)"],
    [fp, fn],
)

for bar, val, rate in zip(bars, [fp, fn], [fp_rate, fn_rate]):
    ax.text(
        bar.get_x() + bar.get_width() / 5,
        bar.get_height() + max(fp, fn) * 0.01,
        f"{val:,}\n({rate:.1%})",
        ha="center",
        va="bottom",
        fontsize=11,
    )

ax.set_title(f"Error trade-offs at tuned threshold (t = {m['best_threshold']:.2f})")
ax.set_ylabel("Number of posts")

# --- move metrics table further DOWN and CENTER it ---
cell_text = [
    ["Test accuracy", f"{m['test_acc']:.3f}"],
    ["Test macro-F1", f"{m['test_macro_f1']:.3f}"],
    ["Val macro-F1 at t*", f"{m['val_macro_f1']:.3f}"],
]

table = plt.table(
    cellText=cell_text,
    colLabels=["Metric", "Value"],
    cellLoc="center",
    colLoc="center",
    loc="bottom",
    bbox=[0.25, -0.35, 0.5, 0.22],  # [x0, y0, width, height]
)
table.auto_set_font_size(False)
table.set_fontsize(10)

plt.tight_layout(rect=[0, 0.15, 1, 1])  # leave extra space at bottom
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", OUT_PATH)
