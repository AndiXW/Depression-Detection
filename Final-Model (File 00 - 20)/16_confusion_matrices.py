import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------- Paths -------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "ds577_data_binary_proc", "v2_no_stop")
FIG_DIR = os.path.join(BASE_DIR, "ds577_figures")
os.makedirs(FIG_DIR, exist_ok=True)

# classical TF-IDF + LogisticRegression pipeline (already trained)
CLASSICAL_MODEL_PATH = os.path.join(
    BASE_DIR, "ds577_models", "tfidf_logreg_binary.pkl"
)

# base directory where DistilBERT training (file 11) saved stuff
DISTILBERT_BASE_DIR = os.path.join(
    BASE_DIR, "ds577_transformers", "distilbert_binary_gpu"
)


# ---------- Helpers -----------------------------------------------------------

def get_latest_checkpoint_dir(base_dir: str) -> str:
    """
    Return a directory that actually contains model weights.
    1) If base_dir has a model file, use it.
    2) Otherwise, pick the last 'checkpoint-*' subfolder.
    """
    direct_files = ["pytorch_model.bin", "model.safetensors"]
    if any(os.path.exists(os.path.join(base_dir, f)) for f in direct_files):
        return base_dir

    ckpts = sorted(glob.glob(os.path.join(base_dir, "checkpoint-*")))
    if not ckpts:
        raise FileNotFoundError(
            f"No model weights found in '{base_dir}' "
            f"(no pytorch_model.bin/model.safetensors or checkpoint-*)."
        )
    return ckpts[-1]


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len: int = 128):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def plot_confusion_matrix(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Not depression (0)", "Depression (1)"]

    plt.figure(figsize=(4.5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[FIG] saved confusion matrix -> {out_path}")


# ---------- Prediction functions ---------------------------------------------

def predict_classical(texts):
    print(f"[CLASSICAL] loading model: {CLASSICAL_MODEL_PATH}\n")
    clf = joblib.load(CLASSICAL_MODEL_PATH)
    # pipeline's predict returns 0/1 directly
    y_pred = clf.predict(texts)
    return np.array(y_pred)


def predict_distilbert(texts, batch_size=64, max_len=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DISTILBERT] device: {device}")

    model_dir = get_latest_checkpoint_dir(DISTILBERT_BASE_DIR)
    print(f"[DISTILBERT] using checkpoint dir: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    ds = TextDataset(texts, tokenizer, max_len=max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_logits = []

    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attn)
            logits = outputs.logits
            all_logits.append(logits.cpu())

    logits = torch.cat(all_logits, dim=0)          # (N, 2)
    probs = torch.softmax(logits, dim=1).numpy()   # convert to numpy
    y_pred = probs.argmax(axis=1)

    # return both labels and prob of positive class
    return y_pred, probs[:, 1]


# ---------- Main -------------------------------------------------------------

def main():
    # 1) Load test data
    test_path = os.path.join(DATA_DIR, "test.csv")
    df_test = pd.read_csv(test_path)
    print(f"[DATA] test path: {test_path}")
    print(f"[DATA] cols: {list(df_test.columns)}")

    X_test = df_test["text_proc"].astype(str).values
    y_test = df_test["label"].values

    pos_rate = (y_test == 1).mean()
    print(
        f"[DATA] test size: {len(df_test)} positives: {y_test.sum()} "
        f"({pos_rate:.3f} pos rate)"
    )

    # 2) Classical model confusion matrix
    y_pred_cl = predict_classical(X_test)
    print("\n[CLASSICAL] classification report:")
    print(classification_report(y_test, y_pred_cl))

    cm_path_cl = os.path.join(FIG_DIR, "cm_classical_binary.png")
    plot_confusion_matrix(
        y_test,
        y_pred_cl,
        "Confusion matrix – TF-IDF + Logistic Regression",
        cm_path_cl,
    )

    # 3) DistilBERT confusion matrix
    try:
        y_pred_bert, probs_bert = predict_distilbert(
            X_test, batch_size=64, max_len=128
        )
    except FileNotFoundError as e:
        print("\n[DISTILBERT] WARNING:", e)
        print("Skip DistilBERT confusion matrix (no checkpoint found).")
        return

    print("\n[DISTILBERT] classification report:")
    print(classification_report(y_test, y_pred_bert))

    cm_path_bert = os.path.join(FIG_DIR, "cm_distilbert_binary.png")
    plot_confusion_matrix(
        y_test,
        y_pred_bert,
        "Confusion matrix – DistilBERT (fine-tuned)",
        cm_path_bert,
    )


if __name__ == "__main__":
    main()
