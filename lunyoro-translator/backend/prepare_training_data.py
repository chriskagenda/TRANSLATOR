"""
Builds a unified, augmented parallel corpus from:
  1. english_nyoro_clean.csv  (6 200 sentence pairs)
  2. Dictionary example sentences (≈2 284 pairs)
  3. Word-level definition pairs  (1 142 pairs)

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
    # 1. Main sentence pairs
    pairs = pd.read_csv(os.path.join(CLEAN_DIR, "english_nyoro_clean.csv"))
    pairs = pairs[["english", "lunyoro"]].copy()

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

    # Basic quality filters
    corpus["english"] = corpus["english"].str.strip()
    corpus["lunyoro"] = corpus["lunyoro"].str.strip()
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
