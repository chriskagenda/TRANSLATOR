"""
Cleans the new Excel dictionary datasets, merges them into the existing
cleaned CSVs, then rebuilds training splits.

Run:
    python clean_extra.py
"""
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

BASE      = os.path.dirname(__file__)
DATA_DIR  = os.path.join(BASE, "data")
EXTRA_DIR = os.path.join(DATA_DIR, "extra")
CLEAN_DIR = os.path.join(DATA_DIR, "cleaned")
OUT_DIR   = os.path.join(DATA_DIR, "training")


# ── helpers ──────────────────────────────────────────────────────────────────

def clean_text(val) -> str:
    if not isinstance(val, str) or not val.strip():
        return ""
    val = val.strip().strip('"').strip("'")
    val = re.sub(r'\s+', ' ', val)
    return val


def is_valid_pair(en: str, lun: str) -> bool:
    """Keep pairs that have real content on both sides."""
    if len(en) < 2 or len(lun) < 2:
        return False
    # skip if lunyoro side is just the same as english (no translation happened)
    if en.lower().strip() == lun.lower().strip():
        return False
    return True


# ── load and normalise each Excel file ───────────────────────────────────────

def load_excel_as_dict_entries(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]

    # normalise column names across the different files
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "word" in cl and "related" not in cl:
            col_map[c] = "word"
        elif "definition" in cl:
            col_map[c] = "definition_en"
        elif "runyoro" in cl or "rutooro" in cl:
            # first one found = runyoro example, second = rutooro example
            if "example_runyoro" not in col_map.values():
                col_map[c] = "example_runyoro"
            else:
                col_map[c] = "example_rutooro"
    df = df.rename(columns=col_map)

    keep = [c for c in ["word", "definition_en", "example_runyoro", "example_rutooro"] if c in df.columns]
    df = df[keep].copy()
    for col in keep:
        df[col] = df[col].apply(clean_text)
    return df


def extract_pairs_from_dict(df: pd.DataFrame) -> pd.DataFrame:
    """Turn dictionary entries into (english, lunyoro) translation pairs."""
    pairs = []

    for _, row in df.iterrows():
        word    = row.get("word", "")
        defn_en = row.get("definition_en", "")
        ex_run  = row.get("example_runyoro", "")
        ex_rut  = row.get("example_rutooro", "")

        # word ↔ english definition
        if word and defn_en and is_valid_pair(defn_en, word):
            pairs.append({"english": defn_en, "lunyoro": word})

        # english example ↔ runyoro example
        if ex_run and defn_en and is_valid_pair(defn_en, ex_run):
            pairs.append({"english": defn_en, "lunyoro": ex_run})

        # english example ↔ rutooro example
        if ex_rut and defn_en and is_valid_pair(defn_en, ex_rut):
            pairs.append({"english": defn_en, "lunyoro": ex_rut})

    return pd.DataFrame(pairs)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CLEAN_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    excel_files = [
        f for f in os.listdir(EXTRA_DIR)
        if f.endswith(".xlsx") or f.endswith(".xls")
    ]

    if not excel_files:
        print("No Excel files found in data/extra/")
        return

    # 1. Load and clean all new Excel files
    new_pairs_list = []
    new_dict_list  = []

    for fname in excel_files:
        path = os.path.join(EXTRA_DIR, fname)
        print(f"Loading: {fname}")
        df = load_excel_as_dict_entries(path)
        pairs = extract_pairs_from_dict(df)
        print(f"  → {len(df)} entries, {len(pairs)} translation pairs extracted")
        new_pairs_list.append(pairs)
        new_dict_list.append(df)

    new_pairs = pd.concat(new_pairs_list, ignore_index=True)
    new_dict  = pd.concat(new_dict_list,  ignore_index=True)

    # 2. Merge new pairs into existing english_nyoro_clean.csv
    existing_pairs = pd.read_csv(os.path.join(CLEAN_DIR, "english_nyoro_clean.csv"))
    combined_pairs = pd.concat([existing_pairs, new_pairs], ignore_index=True)
    combined_pairs["english"] = combined_pairs["english"].apply(clean_text)
    combined_pairs["lunyoro"] = combined_pairs["lunyoro"].apply(clean_text)
    combined_pairs = combined_pairs[
        (combined_pairs["english"].str.len() >= 2) &
        (combined_pairs["lunyoro"].str.len() >= 2)
    ]
    combined_pairs = combined_pairs.drop_duplicates(subset=["english", "lunyoro"])
    combined_pairs = combined_pairs.reset_index(drop=True)
    combined_pairs.to_csv(os.path.join(CLEAN_DIR, "english_nyoro_clean.csv"), index=False)
    print(f"\nSentence pairs: {len(existing_pairs)} → {len(combined_pairs)} (+{len(combined_pairs)-len(existing_pairs)})")

    # 3. Merge new dict entries into existing word_entries_clean.csv
    existing_dict = pd.read_csv(os.path.join(CLEAN_DIR, "word_entries_clean.csv")).fillna("")
    # align columns — new dict may have fewer columns
    for col in existing_dict.columns:
        if col not in new_dict.columns:
            new_dict[col] = ""
    new_dict = new_dict[existing_dict.columns]
    combined_dict = pd.concat([existing_dict, new_dict], ignore_index=True)
    combined_dict = combined_dict[combined_dict["word"].str.strip() != ""]
    combined_dict["_filled"] = combined_dict.apply(lambda r: r.astype(bool).sum(), axis=1)
    combined_dict = combined_dict.sort_values("_filled", ascending=False)
    combined_dict = combined_dict.drop_duplicates(subset=["word"], keep="first")
    combined_dict = combined_dict.drop(columns=["_filled"]).reset_index(drop=True)
    combined_dict.to_csv(os.path.join(CLEAN_DIR, "word_entries_clean.csv"), index=False)
    print(f"Dictionary entries: {len(existing_dict)} → {len(combined_dict)} (+{len(combined_dict)-len(existing_dict)})")

    # 4. Rebuild training splits
    print("\nRebuilding training splits...")
    from prepare_training_data import build_corpus
    corpus = build_corpus()
    train, temp = train_test_split(corpus, test_size=0.1, random_state=42)
    val, test   = train_test_split(temp,   test_size=0.5, random_state=42)
    train.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val.to_csv(  os.path.join(OUT_DIR, "val.csv"),   index=False)
    test.to_csv( os.path.join(OUT_DIR, "test.csv"),  index=False)
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    print("\nDone. Now run: python fine_tune.py --direction both --epochs 10 --batch_size 32")


if __name__ == "__main__":
    main()
