"""
Back-translation augmentation.

Takes Lunyoro sentences from the cleaned corpus, translates them to English
using the current lun2en model, then adds those (synthetic_english, lunyoro)
pairs to the training data.

This forces consistency between both directions and effectively doubles
the training data with diverse sentence structures.

Run:
    python back_translate.py

Output:
    data/cleaned/english_nyoro_clean.csv  (updated with back-translated pairs)
"""
import os
import torch
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer

BASE    = os.path.dirname(__file__)
DATA    = os.path.join(BASE, "data", "cleaned")
MODEL   = os.path.join(BASE, "model", "lun2en")
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
BATCH   = 64


def back_translate():
    if not os.path.isdir(MODEL):
        print("lun2en model not found. Run fine_tune.py first.")
        return

    print(f"Loading lun2en model from {MODEL}...")
    tokenizer = MarianTokenizer.from_pretrained(MODEL)
    model = MarianMTModel.from_pretrained(MODEL).to(DEVICE)
    model.eval()

    pairs = pd.read_csv(os.path.join(DATA, "english_nyoro_clean.csv"))
    lunyoro_sentences = pairs["lunyoro"].dropna().unique().tolist()
    print(f"Back-translating {len(lunyoro_sentences)} Lunyoro sentences...")

    synthetic = []
    for i in range(0, len(lunyoro_sentences), BATCH):
        batch = lunyoro_sentences[i:i + BATCH]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                           padding=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, num_beams=4, max_length=512,
                no_repeat_ngram_size=3, repetition_penalty=1.3,
            )
        for j, ids in enumerate(output_ids):
            en  = tokenizer.decode(ids, skip_special_tokens=True).strip()
            lun = batch[j].strip()

            # ── quality filters ──────────────────────────────────────────────
            if not en or not lun:
                continue
            # Must be real English (not a copy of Lunyoro)
            if en.lower() == lun.lower():
                continue
            # Minimum length
            if len(en) < 8 or len(lun) < 8:
                continue
            # Reject if English output is suspiciously short vs Lunyoro input
            # (model likely failed to translate)
            if len(en.split()) < len(lun.split()) * 0.25:
                continue
            # Reject repeated word runs (hallucination signal)
            words = en.lower().split()
            if len(words) > 3:
                bigrams = list(zip(words, words[1:]))
                repeat_ratio = len([b for b in bigrams if b[0] == b[1]]) / len(bigrams)
                if repeat_ratio > 0.3:
                    continue
            # Reject if output contains mostly non-ASCII (model output garbage)
            ascii_ratio = sum(1 for c in en if ord(c) < 128) / max(len(en), 1)
            if ascii_ratio < 0.85:
                continue

            synthetic.append({"english": en, "lunyoro": lun})

        if (i // BATCH) % 10 == 0:
            print(f"  {i}/{len(lunyoro_sentences)} done...")

    print(f"Generated {len(synthetic)} back-translated pairs")

    new_pairs = pd.DataFrame(synthetic)
    combined  = pd.concat([pairs, new_pairs], ignore_index=True)
    combined  = combined.drop_duplicates(subset=["english", "lunyoro"])
    combined  = combined.reset_index(drop=True)
    combined.to_csv(os.path.join(DATA, "english_nyoro_clean.csv"), index=False)
    print(f"Total pairs after augmentation: {len(combined)}")
    print("Saved to data/cleaned/english_nyoro_clean.csv")
    print("\nNow run: python prepare_training_data.py && python fine_tune.py --direction both")


if __name__ == "__main__":
    back_translate()
