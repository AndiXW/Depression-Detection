# 11_distilbert_binary_GPU_win.py
# DistilBERT binary classifier (Depression vs Not) — Windows-safe + GPU optimized
# Fixes: drops unexpected kwargs (e.g., num_items_in_batch) at both Trainer & model.forward.
# Keeps FAST_DEBUG so tokenization "Map" is fast while we iterate.

import os, json
from pathlib import Path
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, average_precision_score, roc_auc_score,
    classification_report, confusion_matrix
)
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer, set_seed
)

# ========= KNOBS =========
MODEL_NAME       = "distilbert-base-uncased"

RAW_BIN          = r"C:\Users\huyng\Documents\sdsu fall 2025\ds577_data_binary"
OUTDIR           = r"C:\Users\huyng\Documents\sdsu fall 2025\ds577_transformers\distilbert_binary_gpu"

# FAST_DEBUG shrinks data + max_len + parallel tokenization for quick iterations
FAST_DEBUG       = False
DEBUG_FRACTION   = 0.02     # 2% of train for quick checks
DEBUG_MAX_LEN    = 128

TRAIN_FRACTION   = 1.0      # used when FAST_DEBUG=False
EPOCHS           = 2        # bump to 2–3 for final
MAX_LEN          = 192      # when FAST_DEBUG=False

# Tokenization workers (speeds up the 3 "Map" phases)
NUM_PROC_TOKENIZE = 12       # try 4–8; set 0 if Windows complains

# DataLoader workers (start 0; after 1 clean run, raise to 6–8)
NUM_WORKERS      = 8

# Batch sizes
BS_TRAIN_GPU     = 32       # raise until VRAM limit; lower if OOM
BS_EVAL_GPU      = 128

# Optimizer / logging
LR               = 5e-5
WEIGHT_DECAY     = 0.01
LOGGING_STEPS    = 200
SAVE_STEPS       = 2000
SAVE_TOTAL_LIMIT = 2
SEED             = 42
# =========================

# --- tiny helper to print when map uses num_proc ---
def _map(ds, fn, num_proc):
    if num_proc and num_proc > 0:
        print(f"Map (num_proc={num_proc}): ", end="", flush=True)
    out = ds.map(fn, batched=True, num_proc=(num_proc if num_proc and num_proc > 0 else None))
    return out

def build_splits():
    def load(name):
        df = pd.read_csv(os.path.join(RAW_BIN, f"binary_{name}.csv"))
        df = df[["text", "label"]].dropna()
        df["text"] = df["text"].astype(str)
        df["label"] = df["label"].astype(int)
        return df

    train_df, val_df, test_df = load("train"), load("val"), load("test")

    if FAST_DEBUG:
        train_df, _ = train_test_split(
            train_df, train_size=DEBUG_FRACTION,
            stratify=train_df["label"], random_state=SEED
        )
        print(f"[FAST_DEBUG] Using {len(train_df)} train rows, max_len={DEBUG_MAX_LEN}")
    elif 0 < TRAIN_FRACTION < 1.0:
        train_df, _ = train_test_split(
            train_df, train_size=TRAIN_FRACTION,
            stratify=train_df["label"], random_state=SEED
        )

    print(f"[DATA] rows  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    return train_df, val_df, test_df

def tokenize_to_datasets(train_df, val_df, test_df):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    max_len = DEBUG_MAX_LEN if FAST_DEBUG else MAX_LEN

    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=max_len,
        )

    def to_ds(df):
        ds = Dataset.from_pandas(df, preserve_index=False)
        ds = _map(ds, tok, NUM_PROC_TOKENIZE)
        ds = ds.remove_columns(["text"])
        ds = ds.rename_column("label", "labels")
        return ds

    train_ds = to_ds(train_df)
    val_ds   = to_ds(val_df)
    test_ds  = to_ds(test_df)

    ds = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})
    print("[COLUMNS]", ds["train"].column_names)  # expect ['labels','input_ids','attention_mask']
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    return ds, collator

def eval_split(trainer, split):
    out = trainer.predict(split)
    logits = out.predictions
    labels = out.label_ids
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "acc":      float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "pr_auc":   float(average_precision_score(labels, probs)),
        "roc_auc":  float(roc_auc_score(labels, probs)),
        "cm":       confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
        "report":   classification_report(labels, preds, output_dict=True),
    }

# ---- Model wrapper to drop unknown kwargs (robust fix) ----
class CleanForward(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base_model = base_model
    def forward(self, **kwargs):
        # Drop anything Trainer/Accelerate might inject
        for k in ("num_items_in_batch", "num_tokens", "unused_kwargs"):
            kwargs.pop(k, None)
        return self.base_model(**kwargs)

# ---- Trainer patch to ensure inputs dict is clean too ----
class PatchedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs.pop("num_items_in_batch", None)
        inputs.pop("num_tokens", None)
        return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                    num_items_in_batch=num_items_in_batch)

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    set_seed(SEED)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    use_gpu = torch.cuda.is_available()
    bf16_ok = bool(use_gpu and torch.cuda.is_bf16_supported())
    fp16_ok = bool(use_gpu and not bf16_ok)

    print(f"[DEVICE] GPU={use_gpu}  bf16={bf16_ok}  fp16={fp16_ok}")
    if use_gpu:
        print("[DEVICE]", torch.cuda.get_device_name(0), "| CUDA", torch.version.cuda)

    # Data
    train_df, val_df, test_df = build_splits()
    ds, collator = tokenize_to_datasets(train_df, val_df, test_df)

    # Model (+ wrapper to strip bad kwargs)
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = CleanForward(base)

    # (Optional) torch.compile — leave OFF until one clean run; re-enable later if you want.
    # if hasattr(torch, "compile"):
    #     try:
    #         model = torch.compile(model)
    #     except Exception as e:
    #         print("torch.compile skipped:", e)

    bs_train = BS_TRAIN_GPU if use_gpu else 8
    bs_eval  = BS_EVAL_GPU  if use_gpu else 16

    args = TrainingArguments(
        output_dir=OUTDIR,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=bs_train,
        per_device_eval_batch_size=bs_eval,

        dataloader_num_workers=NUM_WORKERS,   # after clean run → 6–8
        dataloader_pin_memory=True,
        dataloader_drop_last=True,

        bf16=bf16_ok,
        fp16=fp16_ok,

        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to="none",

        # Important with new Trainer behavior
        remove_unused_columns=False,
        label_names=["labels"],
        include_num_input_tokens_seen=False,   # prevents kwarg injection in most versions
    )

    # Fresh start for debugging: comment RESUME during fixes
    # ckpts = sorted(Path(OUTDIR).glob("checkpoint-*"),
    #                key=lambda p: int(p.name.split("-")[-1]))
    # resume_path = str(ckpts[-1]) if ckpts else None
    resume_path = None

    trainer = PatchedTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
    )

    trainer.train(resume_from_checkpoint=resume_path)

    # Eval
    val_metrics  = eval_split(trainer, ds["validation"])
    test_metrics = eval_split(trainer, ds["test"])
    out = {
        "use_gpu": use_gpu,
        "bf16": bf16_ok,
        "fp16": fp16_ok,
        "max_len": (DEBUG_MAX_LEN if FAST_DEBUG else MAX_LEN),
        "epochs": EPOCHS,
        "train_rows": len(ds["train"]),
        "val": val_metrics,
        "test": test_metrics,
    }
    with open(os.path.join(OUTDIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Saved:", os.path.join(OUTDIR, "results.json"))

if __name__ == "__main__":
    main()
