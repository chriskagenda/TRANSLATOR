"""
Preprocess CSV data for training and lookup.
Outputs cleaned sentence pairs and dictionary entries.
Run directly to write cleaned CSVs to data/cleaned/.
"""
import pandas as pd
import re
import os
import unicodedata

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CLEAN_DIR = os.path.join(DATA_DIR, "cleaned")

# Normalise curly/smart apostrophes and quotes to their ASCII equivalents
_APOSTROPHE_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'",  # ' '
    "\u201C": '"', "\u201D": '"',  # " "
    "\u02BC": "'",                  # modifier letter apostrophe
    "\u0060": "'",                  # grave accent used as apostrophe
})


def clean_text(text) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    # Unicode NFC normalisation — makes encoding consistent across sources
    text = unicodedata.normalize("NFC", text)
    # Normalise curly apostrophes/quotes to ASCII equivalents
    text = text.translate(_APOSTROPHE_MAP)
    text = text.strip().strip('"').strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def _correct_pos(row) -> str:
    """
    Auto-correct POS tags that are clearly wrong based on Bantu prefix rules.
    Only overrides when the existing tag contradicts the prefix evidence strongly.
    """
    word = str(row.get("word", "")).lower().strip()
    pos  = str(row.get("pos",  "")).upper().strip()

    # Verb infinitives: ku- / oku- / okw- prefix
    verb_prefixes = ("oku", "okw", "ku-", "kw")
    if any(word.startswith(p) for p in verb_prefixes):
        if pos in ("N", "PRON", "ADJ", ""):
            return "V"

    # Noun class prefixes — only correct if tagged V or PRON (not ADJ, could be valid)
    noun_prefixes = ("om", "ab", "ob", "eb", "ek", "ak", "en", "em",
                     "in", "im", "oru", "ama", "obu", "otu", "eri", "aga")
    if any(word.startswith(p) for p in noun_prefixes):
        if pos in ("V", "PRON"):
            return "N"

    # Keep existing tag if no clear contradiction
    return pos if pos else ""


def load_sentence_pairs() -> pd.DataFrame:
    clean_path = os.path.join(CLEAN_DIR, "english_nyoro_clean.csv")
    if os.path.exists(clean_path):
        return pd.read_csv(clean_path)
    # fallback: clean on the fly from original
    path = os.path.join(DATA_DIR, "english_nyoro.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"English": "english", "Nyoro": "lunyoro"})
    df["english"] = df["english"].apply(clean_text)
    df["lunyoro"] = df["lunyoro"].apply(clean_text)
    df = df[(df["english"].str.len() > 3) & (df["lunyoro"].str.len() > 3)]
    df = df.drop_duplicates(subset=["english"])
    df = df.reset_index(drop=True)
    return df


def load_dictionary() -> pd.DataFrame:
    clean_path = os.path.join(CLEAN_DIR, "word_entries_clean.csv")
    if os.path.exists(clean_path):
        df = pd.read_csv(clean_path).fillna("")
        df["pos"] = df.apply(_correct_pos, axis=1)
        return df

    # fallback: clean on the fly from originals
    frames = []
    for fname in ["word_entries_rows.csv", "word_entries_rows (1).csv"]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)

    keep_cols = [
        "word", "definitionEnglish", "definitionNative",
        "exampleSentence1", "exampleSentence1English",
        "exampleSentence2", "exampleSentence2English",
        "dialect", "pos", "domain",
    ]
    df = df[keep_cols].copy()

    for col in keep_cols:
        df[col] = df[col].apply(clean_text)

    df = df[df["word"].str.len() > 0]

    # Auto-correct obviously wrong POS tags using Bantu prefix rules
    df["pos"] = df.apply(_correct_pos, axis=1)

    # For duplicate words keep the entry with the most filled fields
    df["_filled"] = df.apply(lambda r: r.astype(bool).sum(), axis=1)
    df = df.sort_values("_filled", ascending=False)
    df = df.drop_duplicates(subset=["word"], keep="first")
    df = df.drop(columns=["_filled"])

    df = df.reset_index(drop=True)
    return df


def save_cleaned():
    os.makedirs(CLEAN_DIR, exist_ok=True)

    pairs = load_sentence_pairs()
    pairs_path = os.path.join(CLEAN_DIR, "english_nyoro_clean.csv")
    pairs.to_csv(pairs_path, index=False)
    print(f"Sentence pairs: {len(pairs)} rows -> {pairs_path}")

    dictionary = load_dictionary()
    dict_path = os.path.join(CLEAN_DIR, "word_entries_clean.csv")
    dictionary.to_csv(dict_path, index=False)
    print(f"Dictionary entries: {len(dictionary)} rows -> {dict_path}")

    return pairs, dictionary


if __name__ == "__main__":
    pairs, dictionary = save_cleaned()
    print("\nSample sentence pairs:")
    print(pairs.head(3).to_string())
    print("\nSample dictionary entries:")
    print(dictionary.head(3).to_string())
