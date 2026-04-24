"""
Cleans and merges all previously unprocessed raw files into the cleaned CSVs,
then rebuilds training splits.

Files processed:
  raw/word_submissions_rows.csv / (1) / (2)  → example_translation_en + example_runyoro pairs
  raw/word_entries_rows_root.csv / (1)        → definitionEnglish + exampleSentence pairs
  raw/sentence_submissions_rows.csv / (1)     → Lunyoro-only (added to spellcheck corpus only)

Run:
    python clean_unprocessed_raw.py
"""
import os, re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE      = Path(__file__).parent
RAW       = BASE / "data" / "raw"
CLEAN_DIR = BASE / "data" / "cleaned"
OUT_DIR   = BASE / "data" / "training"


def clean_text(val) -> str:
    if not isinstance(val, str) or not val.strip():
        return ""
    val = val.strip().strip('"').strip("'")
    val = re.sub(r'\s+', ' ', val)
    return val


def is_valid_pair(en: str, lun: str) -> bool:
    if len(en) < 4 or len(lun) < 4:
        return False
    if en.lower().strip() == lun.lower().strip():
        return False
    # reject if lunyoro side looks like English (no Bantu markers)
    bantu = ("oku","omu","aba","ebi","eki","ama","ngu","nga","oru","obu","eri","aka","eka","en","em")
    lun_lower = lun.lower()
    has_bantu = any(lun_lower.startswith(m) or f" {m}" in lun_lower for m in bantu)
    all_ascii_lun = all(ord(c) < 128 for c in lun if c.isalpha())
    if not has_bantu and all_ascii_lun and len(lun.split()) < 3:
        return False
    return True


# ── 1. word_submissions_rows — extract example sentence pairs ─────────────────
print("=== word_submissions_rows ===")
ws_pairs = []
for fname in ["word_submissions_rows.csv", "word_submissions_rows (1).csv", "word_submissions_rows (2).csv"]:
    p = RAW / fname
    if not p.exists():
        continue
    df = pd.read_csv(p).fillna("")
    before = len(ws_pairs)
    for _, row in df.iterrows():
        en  = clean_text(row.get("example_translation_en", ""))
        lun = clean_text(row.get("example_rutooro", "") or row.get("example_runyoro", ""))
        if is_valid_pair(en, lun):
            ws_pairs.append({"english": en, "lunyoro": lun})
        # also add word ↔ english definition as a short pair
        word = clean_text(row.get("translation_rutooro", "") or row.get("translation_runyoro", ""))
        if word and en and is_valid_pair(en, word):
            ws_pairs.append({"english": en, "lunyoro": word})
    print(f"  {fname}: +{len(ws_pairs)-before} pairs")

ws_df = pd.DataFrame(ws_pairs).drop_duplicates(subset=["english","lunyoro"])
print(f"  Total from word_submissions: {len(ws_df)}")


# ── 2. word_entries_rows_root — richer dictionary with example sentences ──────
print("\n=== word_entries_rows_root ===")
root_pairs = []
root_dict  = []
for fname in ["word_entries_rows_root.csv", "word_entries_rows_root (1).csv"]:
    p = RAW / fname
    if not p.exists():
        continue
    df = pd.read_csv(p).fillna("")
    before_p = len(root_pairs)
    before_d = len(root_dict)
    for _, row in df.iterrows():
        word    = clean_text(row.get("word", ""))
        defn_en = clean_text(row.get("definitionEnglish", ""))
        ex1_lun = clean_text(row.get("exampleSentence1", ""))
        ex1_en  = clean_text(row.get("exampleSentence1English", ""))
        ex2_lun = clean_text(row.get("exampleSentence2", ""))
        ex2_en  = clean_text(row.get("exampleSentence2English", ""))

        # example sentence pairs
        if is_valid_pair(ex1_en, ex1_lun):
            root_pairs.append({"english": ex1_en, "lunyoro": ex1_lun})
        if is_valid_pair(ex2_en, ex2_lun):
            root_pairs.append({"english": ex2_en, "lunyoro": ex2_lun})
        # word ↔ definition
        if word and defn_en and is_valid_pair(defn_en, word):
            root_pairs.append({"english": defn_en, "lunyoro": word})

        # dictionary entry
        if word:
            root_dict.append({
                "word": word,
                "definitionEnglish": defn_en,
                "definitionNative": clean_text(row.get("definitionNative", "")),
                "exampleSentence1": ex1_lun,
                "exampleSentence1English": ex1_en,
                "exampleSentence2": ex2_lun,
                "exampleSentence2English": ex2_en,
                "dialect": clean_text(row.get("dialect", "")),
                "pos": clean_text(row.get("pos", "")),
                "domain": clean_text(row.get("domain", "")),
            })
    print(f"  {fname}: +{len(root_pairs)-before_p} pairs, +{len(root_dict)-before_d} dict entries")

root_pairs_df = pd.DataFrame(root_pairs).drop_duplicates(subset=["english","lunyoro"])
root_dict_df  = pd.DataFrame(root_dict)
print(f"  Total from root entries: {len(root_pairs_df)} pairs, {len(root_dict_df)} dict entries")


# ── 3. sentence_submissions — Lunyoro-only, no English → skip for pairs ───────
print("\n=== sentence_submissions (Lunyoro-only, skipping for pairs) ===")
for fname in ["sentence_submissions_rows.csv", "sentence_submissions_rows (1).csv"]:
    p = RAW / fname
    if p.exists():
        df = pd.read_csv(p).fillna("")
        print(f"  {fname}: {len(df)} rows — no English source, skipped for training pairs")


# ── 4. Merge into existing cleaned CSVs ───────────────────────────────────────
print("\n=== Merging into cleaned CSVs ===")

# Sentence pairs
existing_pairs = pd.read_csv(CLEAN_DIR / "english_nyoro_clean.csv").fillna("")
all_new_pairs  = pd.concat([ws_df, root_pairs_df], ignore_index=True)
all_new_pairs["english"] = all_new_pairs["english"].apply(clean_text)
all_new_pairs["lunyoro"] = all_new_pairs["lunyoro"].apply(clean_text)

combined_pairs = pd.concat([existing_pairs, all_new_pairs], ignore_index=True)
combined_pairs = combined_pairs[
    (combined_pairs["english"].str.len() >= 4) &
    (combined_pairs["lunyoro"].str.len() >= 4)
]
combined_pairs = combined_pairs.drop_duplicates(subset=["english", "lunyoro"]).reset_index(drop=True)
new_pair_count = len(combined_pairs) - len(existing_pairs)
combined_pairs.to_csv(CLEAN_DIR / "english_nyoro_clean.csv", index=False)
print(f"Sentence pairs: {len(existing_pairs):,} → {len(combined_pairs):,} (+{new_pair_count:,} new)")

# Dictionary entries
existing_dict = pd.read_csv(CLEAN_DIR / "word_entries_clean.csv").fillna("")
for col in existing_dict.columns:
    if col not in root_dict_df.columns:
        root_dict_df[col] = ""
root_dict_df = root_dict_df.reindex(columns=existing_dict.columns, fill_value="")

combined_dict = pd.concat([existing_dict, root_dict_df], ignore_index=True)
combined_dict = combined_dict[combined_dict["word"].str.strip() != ""]
combined_dict["_filled"] = combined_dict.apply(lambda r: r.astype(bool).sum(), axis=1)
combined_dict = combined_dict.sort_values("_filled", ascending=False)
combined_dict = combined_dict.drop_duplicates(subset=["word"], keep="first")
combined_dict = combined_dict.drop(columns=["_filled"]).reset_index(drop=True)
new_dict_count = len(combined_dict) - len(existing_dict)
combined_dict.to_csv(CLEAN_DIR / "word_entries_clean.csv", index=False)
print(f"Dictionary entries: {len(existing_dict):,} → {len(combined_dict):,} (+{new_dict_count:,} new)")


# ── 5. Rebuild training splits ────────────────────────────────────────────────
print("\n=== Rebuilding training splits ===")
from prepare_training_data import build_corpus

corpus = build_corpus()
train, temp = train_test_split(corpus, test_size=0.1, random_state=42)
val, test   = train_test_split(temp,   test_size=0.5, random_state=42)
train.to_csv(OUT_DIR / "train.csv", index=False)
val.to_csv(  OUT_DIR / "val.csv",   index=False)
test.to_csv( OUT_DIR / "test.csv",  index=False)
print(f"Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
print(f"Total: {len(train)+len(val)+len(test):,} pairs")
print("\nDone. Run fine_tune.py to retrain with the expanded dataset.")
