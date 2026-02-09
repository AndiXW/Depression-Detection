# 17_distilbert_arch_fig.py
"""
Create a simple schematic of the DistilBERT-based depression classifier.

Output:
  ds577_figures/FIG8_distilbert_architecture.png
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

BASE_DIR = os.path.dirname(__file__)
FIG_DIR = os.path.join(BASE_DIR, "ds577_figures")
os.makedirs(FIG_DIR, exist_ok=True)

def add_box(ax, xy, width, height, text, facecolor="#e0f0ff"):
    x, y = xy
    rect = Rectangle((x, y), width, height,
                     linewidth=1.2, edgecolor="black",
                     facecolor=facecolor)
    ax.add_patch(rect)
    ax.text(x + width / 2, y + height / 2, text,
            ha="center", va="center", fontsize=9)

def add_arrow(ax, x1, y1, x2, y2, text=None):
    arrow = FancyArrow(x1, y1, x2 - x1, y2 - y1,
                       width=0.02, length_includes_head=True,
                       head_width=0.08, head_length=0.1, color="black")
    ax.add_patch(arrow)
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.05, text,
                ha="center", va="bottom", fontsize=8)

def main():
    fig, ax = plt.subplots(figsize=(8, 3), dpi=200)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # 1. Input Reddit post
    add_box(ax, (0.3, 1.5), 2.2, 1.0,
            "Raw Reddit post\n(text)", facecolor="#f0f0f0")

    # 2. Tokenization
    add_box(ax, (3.0, 1.5), 2.0, 1.0,
            "Tokenizer\n(wordpiece)", facecolor="#ffe9c4")

    # 3. DistilBERT encoder
    add_box(ax, (5.4, 1.5), 2.3, 1.4,
            "DistilBERT encoder\n(6 Transformer layers)", facecolor="#d8e9ff")

    # 4. [CLS] pooled embedding
    add_box(ax, (8.0, 2.1), 1.6, 0.8,
            "[CLS] pooled\nrepresentation", facecolor="#e2ffd8")

    # 5. Classification head
    add_box(ax, (8.0, 0.8), 1.6, 0.8,
            "Linear + Dropout\nSigmoid / Softmax", facecolor="#ffd8d8")

    # Arrows
    add_arrow(ax, 2.5, 2.0, 3.0, 2.0)
    add_arrow(ax, 5.0, 2.0, 5.4, 2.0)
    add_arrow(ax, 7.7, 2.0, 8.0, 2.0, text="[CLS]")
    add_arrow(ax, 8.8, 1.6, 8.8, 1.6)  # small connector
    add_arrow(ax, 8.8, 1.6, 8.8, 1.2)

    # Output label text
    ax.text(9.7, 1.2, "P(depressed)\nvs\nP(not depressed)",
            ha="left", va="center", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "FIG8_distilbert_architecture.png")
    plt.savefig(out_path)
    plt.close()
    print("[SAVE]", out_path)

if __name__ == "__main__":
    main()
