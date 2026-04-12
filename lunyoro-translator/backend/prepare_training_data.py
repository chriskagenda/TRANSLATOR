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

    # 4. Any extra datasets dropped into data/extra/
    extra_dir = os.path.join(DATA_DIR, "extra")
    if os.path.isdir(extra_dir):
        for fname in os.listdir(extra_dir):
            if fname.endswith(".csv"):
                fpath = os.path.join(extra_dir, fname)
                try:
                    extra = pd.read_csv(fpath).rename(columns=str.lower)
                    if "english" in extra.columns and "lunyoro" in extra.columns:
                        extra = extra[["english", "lunyoro"]].dropna()
                        corpus = pd.concat([corpus, extra], ignore_index=True)
                        print(f"  Loaded extra dataset: {fname} ({len(extra)} pairs)")
                    else:
                        print(f"  Skipped {fname} — needs 'english' and 'lunyoro' columns")
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
