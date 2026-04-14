"""
Builds a whitelist of token IDs that appear in the Runyoro/Rutooro training data.
Used during NLLB inference to constrain generation to only known Lunyoro tokens.

Run:
    python build_lunyoro_vocab.py
Output:
    model/lunyoro_token_whitelist.json
"""
import os
import json
import pandas as pd
from transformers import NllbTokenizer

BASE      = os.path.dirname(__file__)
DATA_DIR  = os.path.join(BASE, "data", "cleaned")
MODEL_DIR = os.path.join(BASE, "model")


def build_whitelist():
    tok = NllbTokenizer.from_pretrained(os.path.join(MODEL_DIR, "nllb_en2lun"))

    # Collect all Lunyoro text from cleaned data
    pairs = pd.read_csv(os.path.join(DATA_DIR, "english_nyoro_clean.csv"))
    d     = pd.read_csv(os.path.join(DATA_DIR, "word_entries_clean.csv")).fillna("")

    lunyoro_texts = (
        pairs["lunyoro"].tolist() +
        d["word"].tolist() +
        d["definitionNative"].tolist() +
        d["exampleSentence1"].tolist() +
        d["exampleSentence2"].tolist()
    )
    lunyoro_texts = [t for t in lunyoro_texts if isinstance(t, str) and t.strip()]

    print(f"Building whitelist from {len(lunyoro_texts):,} Lunyoro texts...")

    # Tokenize all Lunyoro text and collect unique token IDs
    allowed_ids = set()

    # Always allow special tokens
    for tok_id in tok.all_special_ids:
        allowed_ids.add(tok_id)

    # Always allow the run_Latn language token
    run_latn_id = tok.convert_tokens_to_ids("run_Latn")
    eng_latn_id = tok.convert_tokens_to_ids("eng_Latn")
    allowed_ids.add(run_latn_id)
    allowed_ids.add(eng_latn_id)

    # Tokenize in batches
    batch_size = 512
    for i in range(0, len(lunyoro_texts), batch_size):
        batch = lunyoro_texts[i:i + batch_size]
        tok.src_lang = "run_Latn"
        encoded = tok(batch, add_special_tokens=False)
        for ids in encoded["input_ids"]:
            allowed_ids.update(ids)

    # Also allow all punctuation, numbers, whitespace tokens
    vocab = tok.get_vocab()
    for token, tid in vocab.items():
        clean = token.lstrip("▁")
        if not clean.isalpha() or len(clean) <= 1:
            allowed_ids.add(tid)

    whitelist = sorted(allowed_ids)
    out_path = os.path.join(MODEL_DIR, "lunyoro_token_whitelist.json")
    with open(out_path, "w") as f:
        json.dump(whitelist, f)

    print(f"Whitelist: {len(whitelist):,} tokens allowed out of {len(vocab):,} total")
    print(f"Saved to: {out_path}")
    return whitelist


if __name__ == "__main__":
    build_whitelist()
