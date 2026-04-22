"""
Builds a unified, augmented parallel corpus from:
  1. english_nyoro_clean.csv  (6 200 sentence pairs)
  2. Dictionary example sentences (≈2 284 pairs)
  3. Word-level definition pairs  (1 142 pairs)
  4. Any extra CSVs placed in data/extra/ with columns: english, lunyoro

Outputs:
  data/training/train.csv
  data/training/val.csv
  data/training/test.csv
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
CLEAN_DIR = os.path.join(DATA_DIR, "cleaned")
OUT_DIR   = os.path.join(DATA_DIR, "training")


def build_corpus() -> pd.DataFrame:
    # 1. Main sentence pairs — detect domain from content
    pairs = pd.read_csv(os.path.join(CLEAN_DIR, "english_nyoro_clean.csv"))
    pairs = pairs[["english", "lunyoro"]].copy()

    # Tag domains based on keywords
    def tag_domain(text: str) -> str:
        t = text.lower()
        if any(w in t for w in ["god", "lord", "jesus", "christ", "pray", "church", "bible", "holy", "spirit", "gospel", "psalm", "disciple", "apostle", "prophet"]):
            return "RELIGIOUS"
        if any(w in t for w in ["hospital", "doctor", "medicine", "disease", "health", "patient", "nurse", "clinic", "treatment", "symptom"]):
            return "MEDICAL"
        if any(w in t for w in ["school", "teacher", "student", "learn", "education", "class", "university", "college", "study"]):
            return "EDUCATION"
        if any(w in t for w in ["government", "law", "court", "police", "president", "minister", "parliament", "election", "vote"]):
            return "GOVERNMENT"
        if any(w in t for w in ["farm", "crop", "harvest", "soil", "plant", "animal", "cattle", "agriculture"]):
            return "AGRICULTURE"
        return "GENERAL"

    pairs["domain"] = pairs["english"].apply(tag_domain)

    # Prepend domain tag to source sentence
    pairs["english"] = "[" + pairs["domain"] + "] " + pairs["english"]
    pairs = pairs[["english", "lunyoro"]]

    # 2. Dictionary example sentences
    d = pd.read_csv(os.path.join(CLEAN_DIR, "word_entries_clean.csv")).fillna("")

    ex1 = d[(d["exampleSentence1English"] != "") & (d["exampleSentence1"] != "")][
        ["exampleSentence1English", "exampleSentence1"]
    ].rename(columns={"exampleSentence1English": "english", "exampleSentence1": "lunyoro"})

    ex2 = d[(d["exampleSentence2English"] != "") & (d["exampleSentence2"] != "")][
        ["exampleSentence2English", "exampleSentence2"]
    ].rename(columns={"exampleSentence2English": "english", "exampleSentence2": "lunyoro"})

    # 3. Word-level definition pairs
    word_pairs = d[d["definitionEnglish"] != ""][["definitionEnglish", "word"]].rename(
        columns={"definitionEnglish": "english", "word": "lunyoro"}
    )

    corpus = pd.concat([pairs, ex1, ex2, word_pairs], ignore_index=True)

    # 4. New sentence pairs from April submission
    sent_path = os.path.join(CLEAN_DIR, "runyoro_english_sentences_clean.csv")
    if os.path.exists(sent_path):
        sent_df = pd.read_csv(sent_path).fillna("")
        sent_df = sent_df[["english", "lunyoro"]].dropna()
        sent_df = sent_df[(sent_df["english"].str.len() >= 3) & (sent_df["lunyoro"].str.len() >= 3)]
        corpus = pd.concat([corpus, sent_df], ignore_index=True)
        print(f"  Loaded runyoro_english_sentences_clean.csv ({len(sent_df)} pairs)")

    # 5. Rutooro dictionary — generate pairs from word + definition + examples
    dict_path = os.path.join(CLEAN_DIR, "rutooro_dictionary_clean.csv")
    if os.path.exists(dict_path):
        rdict = pd.read_csv(dict_path).fillna("")
        dict_pairs = []

        # word (lunyoro) <-> definition (english)
        for _, row in rdict.iterrows():
            word = str(row.get("word", "")).strip()
            defn = str(row.get("definition", "")).strip()
            ex_r = str(row.get("example_runyoro", "")).strip()
            ex_t = str(row.get("example_rutooro", "")).strip()

            if word and defn and len(defn) >= 5:
                dict_pairs.append({"english": defn, "lunyoro": word})

            # example sentence pairs
            if ex_r and ex_t and len(ex_r) >= 5 and len(ex_t) >= 5:
                dict_pairs.append({"english": ex_t, "lunyoro": ex_r})

        dict_pairs_df = pd.DataFrame(dict_pairs).drop_duplicates()
        corpus = pd.concat([corpus, dict_pairs_df], ignore_index=True)
        print(f"  Loaded rutooro_dictionary_clean.csv ({len(dict_pairs_df)} pairs)")

    # 6. Small extra cleaned CSVs (empaako, idioms, numbers, interjections, proverbs)
    small_extras = [
        "empaako_pairs.csv", "idioms_pairs.csv", "numbers_pairs.csv",
        "interjections_pairs_clean.csv", "proverbs_pairs_clean.csv",
    ]
    for fname in small_extras:
        fpath = os.path.join(CLEAN_DIR, fname)
        if os.path.exists(fpath):
            try:
                extra = pd.read_csv(fpath).rename(columns=str.lower)
                if "english" in extra.columns and "lunyoro" in extra.columns:
                    extra = extra[["english", "lunyoro"]].dropna()
                    corpus = pd.concat([corpus, extra], ignore_index=True)
                    print(f"  Loaded {fname} ({len(extra)} pairs)")
            except Exception as e:
                print(f"  Skipped {fname} — {e}")

    # 5. R/L rule augmentation — teach the model correct Lunyoro orthography
    # For every pair where the Lunyoro side contains L, add:
    #   a) The R/L-corrected version as the canonical target
    #   b) A "wrong → right" pair so the model learns to prefer correct forms
    from language_rules import apply_rl_rule_to_text
    import re

    def has_l(text: str) -> bool:
        return bool(re.search(r'[lL]', text))

    rl_pairs = []
    seen_english = set(corpus["english"].str.lower())

    for _, row in corpus.iterrows():
        lun = str(row["lunyoro"])
        eng = str(row["english"])
        if not has_l(lun):
            continue

        corrected = apply_rl_rule_to_text(lun)
        if corrected == lun:
            continue  # already correct, no augmentation needed

        # Add the corrected form as an additional training target
        # This teaches the model: given this English, produce the R/L-correct Lunyoro
        aug_key = eng.lower() + "_rl"
        if aug_key not in seen_english:
            seen_english.add(aug_key)
            rl_pairs.append({"english": eng, "lunyoro": corrected})

    if rl_pairs:
        rl_df = pd.DataFrame(rl_pairs)
        corpus = pd.concat([corpus, rl_df], ignore_index=True)
        print(f"  R/L augmentation: +{len(rl_df)} corrected pairs")
    # Basic quality filters
    corpus["english"] = corpus["english"].str.strip()
    corpus["lunyoro"] = corpus["lunyoro"].str.strip()
    # Apply R/L rule to all Lunyoro targets so the model learns correct orthography
    corpus["lunyoro"] = corpus["lunyoro"].apply(
        lambda t: apply_rl_rule_to_text(str(t)) if isinstance(t, str) else t
    )
    corpus = corpus[(corpus["english"].str.len() >= 2) & (corpus["lunyoro"].str.len() >= 2)]

    # Filter out garbage dictionary notation entries
    # These are malformed rows like "(pl. same) . (pl. same)" with no real content
    garbage_patterns = re.compile(
        r'^(\[GENERAL\]\s*)?'           # optional domain tag
        r'[\s.,;:()\[\]]*'              # leading punctuation/whitespace only
        r'(\(pl\.?\s*(same|nil)\)'      # "(pl. same)" or "(pl. nil)"
        r'|n\.,?\s*\(pl\.'              # "n., (pl."
        r'|v\.,?\s*\(pl\.'              # "v., (pl."
        r'|\(pl\.\s*\w*\))'             # any "(pl. X)"
        r'[\s.,;:()\[\]]*$',            # trailing junk
        re.IGNORECASE
    )
    # Also drop rows where English is mostly punctuation/abbreviations and very short real words
    def is_garbage(text: str) -> bool:
        # Strip domain tag
        t = re.sub(r'^\[.*?\]\s*', '', str(text)).strip()
        # Must have at least one real word (3+ alpha chars)
        real_words = re.findall(r'[a-zA-Z]{3,}', t)
        if not real_words:
            return True
        # Reject if it's just dictionary notation
        if garbage_patterns.match(text):
            return True
        # Reject if >60% of tokens are abbreviations like "n.", "v.", "pl.", "adj."
        tokens = t.split()
        abbrev = sum(1 for tok in tokens if re.match(r'^[a-z]{1,3}\.$', tok, re.I))
        if tokens and abbrev / len(tokens) > 0.6:
            return True
        return False

    before = len(corpus)
    corpus = corpus[~corpus["english"].apply(is_garbage)]
    removed = before - len(corpus)
    if removed:
        print(f"  Removed {removed} garbage/notation-only rows")

    corpus = corpus.drop_duplicates(subset=["english", "lunyoro"])
    corpus = corpus.reset_index(drop=True)
    return corpus


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    corpus = build_corpus()
    print(f"Total pairs: {len(corpus)}")

    train, temp = train_test_split(corpus, test_size=0.1, random_state=42)
    val, test   = train_test_split(temp,   test_size=0.5, random_state=42)

    train.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val.to_csv(  os.path.join(OUT_DIR, "val.csv"),   index=False)
    test.to_csv( os.path.join(OUT_DIR, "test.csv"),  index=False)

    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    print("Saved to", OUT_DIR)


if __name__ == "__main__":
    main()
