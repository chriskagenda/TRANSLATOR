"""
Fine-tunes facebook/nllb-200-distilled-600M for Lunyoro/Rutooro translation.
Assigns GPU 1 exclusively so MarianMT stays on GPU 0.

Usage:
    python fine_tune_nllb.py --direction en2lun
    python fine_tune_nllb.py --direction lun2en
    python fine_tune_nllb.py               # both (default)

Models saved to:
    model/nllb_en2lun/
    model/nllb_lun2en/
"""
import os
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    NllbTokenizer,
    AutoModelForSeq2SeqLM,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data", "training")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
BASE_MODEL = "facebook/nllb-200-distilled-600M"

# NLLB language codes — run_Latn (Rundi) is closest to Lunyoro/Rutooro
LANG_EN  = "eng_Latn"
LANG_LUN = "run_Latn"

# Use both GPUs via DataParallel if available, else single GPU, else CPU
if torch.cuda.device_count() >= 2:
    DEVICE = torch.device("cuda:0")
    USE_MULTI_GPU = True
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    USE_MULTI_GPU = False
else:
    DEVICE = torch.device("cpu")
    USE_MULTI_GPU = False

print(f"NLLB will use: {torch.cuda.device_count()} GPU(s), DataParallel={USE_MULTI_GPU}")


class NLLBDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, src_col: str, tgt_col: str,
                 src_lang: str, tgt_lang: str, max_len: int = 256):
        self.tokenizer = tokenizer
        self.src = df[src_col].tolist()
        self.tgt = df[tgt_col].tolist()
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_len = max_len

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        self.tokenizer.src_lang = self.src_lang
        src_enc = self.tokenizer(
            self.src[idx], max_length=self.max_len,
            truncation=True, padding="max_length", return_tensors="pt"
        )
        self.tokenizer.src_lang = self.tgt_lang
        tgt_enc = self.tokenizer(
            self.tgt[idx], max_length=self.max_len,
            truncation=True, padding="max_length", return_tensors="pt"
        )
        labels = tgt_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids":      src_enc["input_ids"].squeeze(),
            "attention_mask": src_enc["attention_mask"].squeeze(),
            "labels":         labels,
        }


def train_nllb(direction: str, epochs: int = 10, batch_size: int = 16, lr: float = 5e-5):
    print(f"\n{'='*60}")
    print(f"NLLB Training: {direction}  |  Device: {DEVICE}")
    print(f"{'='*60}")

    src_col  = "english" if direction == "en2lun" else "lunyoro"
    tgt_col  = "lunyoro" if direction == "en2lun" else "english"
    src_lang = LANG_EN   if direction == "en2lun" else LANG_LUN
    tgt_lang = LANG_LUN  if direction == "en2lun" else LANG_EN
    save_path = os.path.join(MODEL_DIR, f"nllb_{direction}")
    os.makedirs(save_path, exist_ok=True)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv")).fillna("")
    val_df   = pd.read_csv(os.path.join(DATA_DIR, "val.csv")).fillna("")
    print(f"Train: {len(train_df)}  Val: {len(val_df)}")

    print(f"Loading {BASE_MODEL}...")
    tokenizer = NllbTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(DEVICE)
    model.gradient_checkpointing_enable()  # saves VRAM at cost of ~20% speed

    if USE_MULTI_GPU:
        print(f"Wrapping with DataParallel across {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    effective_batch = batch_size * (torch.cuda.device_count() if USE_MULTI_GPU else 1)
    print(f"Effective batch size: {effective_batch}")

    train_ds = NLLBDataset(train_df, tokenizer, src_col, tgt_col, src_lang, tgt_lang)
    val_ds   = NLLBDataset(val_df,   tokenizer, src_col, tgt_col, src_lang, tgt_lang)

    train_loader = DataLoader(train_ds, batch_size=effective_batch, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=effective_batch, shuffle=False, num_workers=0, pin_memory=True)

    raw_model = model.module if USE_MULTI_GPU else model
    optimizer = AdamW(raw_model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            if loss.dim() > 0:
                loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.mean().item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            raw_model = model.module if USE_MULTI_GPU else model
            raw_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f}) → {save_path}")

    print(f"\nDone. Best val_loss={best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=["en2lun", "lun2en", "both"], default="both")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=5e-5)
    args = parser.parse_args()

    directions = ["en2lun", "lun2en"] if args.direction == "both" else [args.direction]
    for d in directions:
        train_nllb(d, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
