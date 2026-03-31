"""
Fine-tunes Helsinki-NLP/opus-mt-en-mul for English → Lunyoro/Rutooro
and a reverse model for Lunyoro → English.

Usage:
    python fine_tune.py --direction en2lun   # English → Lunyoro
    python fine_tune.py --direction lun2en   # Lunyoro → English
    python fine_tune.py                      # trains both (default)

Models saved to:
    model/en2lun/
    model/lun2en/
"""
import os
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data", "training")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

# Base models — opus-mt-en-mul covers English→many languages (good starting point)
# For reverse we use opus-mt-mul-en
BASE_MODELS = {
    "en2lun": "Helsinki-NLP/opus-mt-en-mul",
    "lun2en": "Helsinki-NLP/opus-mt-mul-en",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0


class TranslationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: MarianTokenizer,
                 src_col: str, tgt_col: str, max_len: int = 128):
        self.tokenizer = tokenizer
        self.src = df[src_col].tolist()
        self.tgt = df[tgt_col].tolist()
        self.max_len = max_len

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src_enc = self.tokenizer(
            self.src[idx], max_length=self.max_len,
            truncation=True, padding="max_length", return_tensors="pt"
        )
        tgt_enc = self.tokenizer(
                text_target=self.tgt[idx], max_length=self.max_len,
                truncation=True, padding="max_length", return_tensors="pt"
            )
        labels = tgt_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100  # ignore padding in loss
        return {
            "input_ids":      src_enc["input_ids"].squeeze(),
            "attention_mask": src_enc["attention_mask"].squeeze(),
            "labels":         labels,
        }


def train_direction(direction: str, epochs: int = 10, batch_size: int = 16, lr: float = 5e-5):
    print(f"\n{'='*60}")
    print(f"Training direction: {direction}")
    print(f"Device: {DEVICE}  |  GPUs: {NUM_GPUS}")
    print(f"{'='*60}")

    base_model = BASE_MODELS[direction]
    src_col = "english" if direction == "en2lun" else "lunyoro"
    tgt_col = "lunyoro" if direction == "en2lun" else "english"
    save_path = os.path.join(MODEL_DIR, direction)
    os.makedirs(save_path, exist_ok=True)

    # Load data
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv")).fillna("")
    val_df   = pd.read_csv(os.path.join(DATA_DIR, "val.csv")).fillna("")

    print(f"Train: {len(train_df)}  Val: {len(val_df)}")

    # Load tokenizer and model
    print(f"Loading base model: {base_model}")
    tokenizer = MarianTokenizer.from_pretrained(base_model)
    model = MarianMTModel.from_pretrained(base_model).to(DEVICE)

    # Use all available GPUs
    if NUM_GPUS > 1:
        print(f"Using {NUM_GPUS} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    effective_batch = batch_size * max(NUM_GPUS, 1)
    print(f"Effective batch size: {effective_batch} ({batch_size} x {max(NUM_GPUS,1)} GPUs)")

    train_ds = TranslationDataset(train_df, tokenizer, src_col, tgt_col)
    val_ds   = TranslationDataset(val_df,   tokenizer, src_col, tgt_col)

    train_loader = DataLoader(train_ds, batch_size=effective_batch, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=effective_batch, shuffle=False, num_workers=4, pin_memory=True)

    # Unwrap model for optimizer (DataParallel wraps it)
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    optimizer = AdamW(raw_model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            # DataParallel returns per-GPU losses as a tensor — mean to scalar
            if loss.dim() > 0:
                loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.mean().item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # Save best checkpoint — always save the unwrapped model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            raw_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f}) → {save_path}")

    print(f"\nDone. Best val_loss={best_val_loss:.4f}")
    print(f"Model saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=["en2lun", "lun2en", "both"], default="both")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=5e-5)
    args = parser.parse_args()

    directions = ["en2lun", "lun2en"] if args.direction == "both" else [args.direction]
    for d in directions:
        train_direction(d, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
