"""
Cleans and merges the new submission datasets into the existing cleaned CSVs.

Sources:
  data/sentence submission 10 april/
    - Runyoro-English_Translation.xlsx  → direct sentence pairs
    - sentence_submissions_rows.csv     → translation_runyoro / translation_rutooro (no English source)
    - sentence_submissions_rows (1).csv → same format

  data/word submision 10 april/
    - corpus_sentences_rows (1).csv     → native + english sentence pairs
    - submissions_rows.csv              → word + example sentences
    - submissions_rows (1).csv          → same
    - submissions_rows (2).csv          → same

Run:
    python clean_new_submissions.py
"""
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

BASE      = os.path.dirname(__file__)
DATA_DIR  = os.path.join(BASE, "data")
CLEAN_DIR = os.path.join(DATA_DIR, "cleaned")
OUT_DIR   = os.path.join(DATA_DIR, "training")
SENT_DIR  = os.path.join(DATA_DIR, "sentence submission 10 april")
WORD_DIR  = os.path.join(DATA_DIR, "word submision 10 april")


def clean_text(val) -> str:
    if not isinstance(val, str) or not val.strip():
        return ""
    val = val.strip().strip('"').strip("'")
    val = re.sub(r'\s+', ' ', val)
    return val


def is_valid_pair(en: str, lun: str) -> bool:
    if len(en) < 3 or len(lun) < 3:
        return False
    if en.lower().strip() == lun.lower().strip():
        return False
    return True


# ── 1. Load new sentence pairs ────────────────────────────────────────────────

new_pairs = []

# Runyoro-English_Translation.xlsx — direct pairs
xlsx = pd.read_excel(os.path.join(SENT_DIR, "Runyoro-English_Translation.xlsx")).fillna("")
xlsx.columns = [c.strip() for c in xlsx.columns]
for _, row in xlsx.iterrows():
    en  = clean_text(row.get("English", ""))
    lun = clean_text(row.get("Runyoro", ""))
    if is_valid_pair(en, lun):
        new_pairs.append({"english": en, "lunyoro": lun})
print(f"Runyoro-English_Translation.xlsx: {len(new_pairs)} pairs")

# sentence_submissions CSVs — have runyoro + rutooro but no English source
# We use rutooro as lunyoro (they're the same language, different dialect spelling)
for fname in ["sentence_submissions_rows.csv", "sentence_submissions_rows (1).csv"]:
    path = os.path.join(SENT_DIR, fname)
    df = pd.read_csv(path).fillna("")
    before = len(new_pairs)
    for _, row in df.iterrows():
        lun = clean_text(row.get("translation_rutooro", "") or row.get("translation_runyoro", ""))
        if lun and len(lun) >= 5:
            # No English source — skip for sentence pairs, will use for spellcheck corpus only
            pass
    print(f"{fname}: skipped (no English source — Lunyoro-only sentences)")

# corpus_sentences — has native (Lunyoro) + english
corpus = pd.read_csv(os.path.join(WORD_DIR, "corpus_sentences_rows (1).csv")).fillna("")
before = len(new_pairs)
for _, row in corpus.iterrows():
    en  = clean_text(row.get("english", ""))
    lun = clean_text(row.get("native", ""))
    if is_valid_pair(en, lun):
        new_pairs.append({"english": en, "lunyoro": lun})
print(f"corpus_sentences_rows (1).csv: +{len(new_pairs)-before} pairs")

# ── 2. Load new word/example pairs ───────────────────────────────────────────

new_dict_rows = []

for fname in ["submissions_rows.csv", "submissions_rows (1).csv", "submissions_rows (2).csv"]:
    path = os.path.join(WORD_DIR, fname)
    df = pd.read_csv(path).fillna("")
    before_p = len(new_pairs)
    before_d = len(new_dict_rows)
    for _, row in df.iterrows():
        lun_word = clean_text(row.get("translation_rutooro", "") or row.get("translation_runyoro", ""))
        ex_lun   = clean_text(row.get("example_rutooro", "") or row.get("example_runyoro", ""))
        ex_en    = clean_text(row.get("example_translation_en", ""))

        # Example sentence pairs
        if is_valid_pair(ex_en, ex_lun):
            new_pairs.append({"english": ex_en, "lunyoro": ex_lun})

        # Dictionary entry
        if lun_word:
            new_dict_rows.append({
                "word": lun_word,
                "definitionEnglish": ex_en,
                "definitionNative": ex_lun,
                "exampleSentence1": ex_lun,
                "exampleSentence1English": ex_en,
                "exampleSentence2": "",
                "exampleSentence2English": "",
                "dialect": clean_text(row.get("dialect", "")),
                "pos": "",
                "domain": "",
            })
    print(f"{fname}: +{len(new_pairs)-before_p} pairs, +{len(new_dict_rows)-before_d} dict entries")

# ── 3. Merge into existing cleaned CSVs ──────────────────────────────────────

# Sentence pairs
existing_pairs = pd.read_csv(os.path.join(CLEAN_DIR, "english_nyoro_clean.csv"))
new_pairs_df   = pd.DataFrame(new_pairs)
new_pairs_df["english"] = new_pairs_df["english"].apply(clean_text)
new_pairs_df["lunyoro"] = new_pairs_df["lunyoro"].apply(clean_text)
combined_pairs = pd.concat([existing_pairs, new_pairs_df], ignore_index=True)
combined_pairs = combined_pairs[(combined_pairs["english"].str.len() >= 3) & (combined_pairs["lunyoro"].str.len() >= 3)]
combined_pairs = combined_pairs.drop_duplicates(subset=["english", "lunyoro"]).reset_index(drop=True)
combined_pairs.to_csv(os.path.join(CLEAN_DIR, "english_nyoro_clean.csv"), index=False)
print(f"\nSentence pairs: {len(existing_pairs):,} → {len(combined_pairs):,} (+{len(combined_pairs)-len(existing_pairs):,})")

# Dictionary
existing_dict = pd.read_csv(os.path.join(CLEAN_DIR, "word_entries_clean.csv")).fillna("")
new_dict_df   = pd.DataFrame(new_dict_rows)
for col in existing_dict.columns:
    if col not in new_dict_df.columns:
        new_dict_df[col] = ""
new_dict_df = new_dict_df[existing_dict.columns]
combined_dict = pd.concat([existing_dict, new_dict_df], ignore_index=True)
combined_dict = combined_dict[combined_dict["word"].str.strip() != ""]
combined_dict["_filled"] = combined_dict.apply(lambda r: r.astype(bool).sum(), axis=1)
combined_dict = combined_dict.sort_values("_filled", ascending=False)
combined_dict = combined_dict.drop_duplicates(subset=["word"], keep="first")
combined_dict = combined_dict.drop(columns=["_filled"]).reset_index(drop=True)
combined_dict.to_csv(os.path.join(CLEAN_DIR, "word_entries_clean.csv"), index=False)
print(f"Dictionary entries: {len(existing_dict):,} → {len(combined_dict):,} (+{len(combined_dict)-len(existing_dict):,})")

# ── 4. Rebuild training splits ────────────────────────────────────────────────
print("\nRebuilding training splits...")
from prepare_training_data import build_corpus
corpus_df = build_corpus()
train, temp = train_test_split(corpus_df, test_size=0.1, random_state=42)
val, test   = train_test_split(temp, test_size=0.5, random_state=42)
train.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
val.to_csv(  os.path.join(OUT_DIR, "val.csv"),   index=False)
test.to_csv( os.path.join(OUT_DIR, "test.csv"),  index=False)
print(f"Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
print("\nDone. Now run:")
print("  python fine_tune.py --direction both --epochs 10 --batch_size 32")
print("  python fine_tune_nllb.py --direction both --epochs 10 --batch_size 4")
