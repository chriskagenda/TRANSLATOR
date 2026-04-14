"""
Full retraining pipeline — ingests new CSVs from the root /CSV folder,
merges them into the existing cleaned data, rebuilds training splits,
fine-tunes both MarianMT and NLLB models, then rebuilds the semantic index.

Run from the backend directory:
    python retrain.py

Flags:
    --skip-marian     skip MarianMT fine-tuning
    --skip-nllb       skip NLLB fine-tuning
    --skip-index      skip semantic index rebuild
    --direction       en2lun | lun2en | both  (default: both)
"""
import os
import sys
import argparse
import shutil
import subprocess
import pandas as pd
import re
import unicodedata

BASE     = os.path.dirname(os.path.abspath(__file__))
ROOT_CSV = os.path.abspath(os.path.join(BASE, "..", "..", "CSV"))
DATA_DIR = os.path.join(BASE, "data")
CLEAN_DIR = os.path.join(DATA_DIR, "cleaned")

# ── helpers ──────────────────────────────────────────────────────────────────

_APOSTROPHE_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'",
    "\u201C": '"', "\u201D": '"',
    "\u02BC": "'", "\u0060": "'",
})


def clean_text(text) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_APOSTROPHE_MAP)
    text = text.strip().strip('"').strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def run(script: str, *args):
    cmd = [sys.executable, os.path.join(BASE, script)] + list(args)
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=BASE)
    if result.returncode != 0:
        print(f"ERROR: {script} failed (exit {result.returncode})")
        sys.exit(result.returncode)


# ── Step 1: ingest new sentence pairs ────────────────────────────────────────

def ingest_sentence_pairs():
    src = os.path.join(ROOT_CSV, "english_nyoro.csv")
    if not os.path.exists(src):
        print("  No english_nyoro.csv found in CSV/ — skipping sentence pairs")
        return 0

    new_df = pd.read_csv(src)
    new_df.columns = [c.strip() for c in new_df.columns]
    new_df = new_df.rename(columns={"English": "english", "Nyoro": "lunyoro"})
    new_df["english"] = new_df["english"].apply(clean_text)
    new_df["lunyoro"] = new_df["lunyoro"].apply(clean_text)
    new_df = new_df[["english", "lunyoro"]]
    new_df = new_df[(new_df["english"].str.len() > 3) & (new_df["lunyoro"].str.len() > 3)]

    clean_path = os.path.join(CLEAN_DIR, "english_nyoro_clean.csv")
    if os.path.exists(clean_path):
        existing = pd.read_csv(clean_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["english"])
        added = len(combined) - len(existing)
    else:
        combined = new_df.drop_duplicates(subset=["english"])
        added = len(combined)

    combined.reset_index(drop=True).to_csv(clean_path, index=False)
    print(f"  Sentence pairs: +{added} new rows → {len(combined)} total")
    return added


# ── Step 2: ingest new word entries ──────────────────────────────────────────

WORD_KEEP_COLS = [
    "word", "definitionEnglish", "definitionNative",
    "exampleSentence1", "exampleSentence1English",
    "exampleSentence2", "exampleSentence2English",
    "dialect", "pos", "domain",
]


def _load_word_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    # keep only columns that exist in this file
    cols = [c for c in WORD_KEEP_COLS if c in df.columns]
    df = df[cols].copy()
    # add missing columns as empty strings
    for c in WORD_KEEP_COLS:
        if c not in df.columns:
            df[c] = ""
    for col in WORD_KEEP_COLS:
        df[col] = df[col].apply(clean_text)
    return df[WORD_KEEP_COLS]


def ingest_word_entries():
    frames = []
    for fname in ["word_entries_rows.csv", "word_entries_rows (1).csv"]:
        src = os.path.join(ROOT_CSV, fname)
        if os.path.exists(src):
            frames.append(_load_word_csv(src))
            print(f"  Loaded {fname}")

    if not frames:
        print("  No word entry CSVs found in CSV/ — skipping")
        return 0

    new_df = pd.concat(frames, ignore_index=True)
    new_df = new_df[new_df["word"].str.len() > 0]

    clean_path = os.path.join(CLEAN_DIR, "word_entries_clean.csv")
    if os.path.exists(clean_path):
        existing = pd.read_csv(clean_path).fillna("")
        # ensure same columns
        for c in WORD_KEEP_COLS:
            if c not in existing.columns:
                existing[c] = ""
        existing = existing[WORD_KEEP_COLS]
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["word"], keep="first")
        added = len(combined) - len(existing)
    else:
        combined = new_df.drop_duplicates(subset=["word"], keep="first")
        added = len(combined)

    combined.reset_index(drop=True).to_csv(clean_path, index=False)
    print(f"  Word entries: +{added} new rows → {len(combined)} total")
    return added


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-marian", action="store_true")
    parser.add_argument("--skip-nllb",   action="store_true")
    parser.add_argument("--skip-index",  action="store_true")
    parser.add_argument("--direction",   default="both", choices=["en2lun", "lun2en", "both"])
    args = parser.parse_args()

    os.makedirs(CLEAN_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("Step 1/5 — Ingesting new CSV data from CSV/")
    print("=" * 60)
    pairs_added = ingest_sentence_pairs()
    words_added = ingest_word_entries()

    if pairs_added == 0 and words_added == 0:
        print("  No new data found. Aborting.")
        sys.exit(0)

    print("\n" + "=" * 60)
    print("Step 2/5 — Rebuilding training splits")
    print("=" * 60)
    run("prepare_training_data.py")

    print("\n" + "=" * 60)
    print("Step 3/5 — Fine-tuning MarianMT models")
    print("=" * 60)
    if not args.skip_marian:
        run("fine_tune.py", "--direction", args.direction)
    else:
        print("  Skipped (--skip-marian)")

    print("\n" + "=" * 60)
    print("Step 4/5 — Fine-tuning NLLB models")
    print("=" * 60)
    if not args.skip_nllb:
        run("fine_tune_nllb.py", "--direction", args.direction)
    else:
        print("  Skipped (--skip-nllb)")

    print("\n" + "=" * 60)
    print("Step 5/5 — Rebuilding semantic search index + vocab whitelist")
    print("=" * 60)
    if not args.skip_index:
        run("train.py")
        run("build_lunyoro_vocab.py")
        run("patch_index.py")
    else:
        print("  Skipped (--skip-index)")

    print("\n" + "=" * 60)
    print("Retraining complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
