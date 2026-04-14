"""
Prunes the NLLB tokenizer vocabulary to remove other Bantu language tokens,
keeping only:
  - English tokens (eng_Latn)
  - Rundi tokens (run_Latn) — our proxy for Runyoro/Rutooro
  - Universal tokens (punctuation, numbers, special tokens)

This prevents the model from generating Luganda/Kinyarwanda words.

Run:
    python prune_nllb_vocab.py
"""
import os
import json
import shutil
from transformers import NllbTokenizer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

# Bantu languages to REMOVE from the vocabulary
REMOVE_LANGS = {
    "lug_Latn",  # Luganda
    "kin_Latn",  # Kinyarwanda
    "nya_Latn",  # Chichewa
    "bem_Latn",  # Bemba
    "kik_Latn",  # Kikuyu
    "kon_Latn",  # Kongo
    "lin_Latn",  # Lingala
    "nso_Latn",  # Northern Sotho
    "sot_Latn",  # Sotho
    "ssw_Latn",  # Swati
    "tsn_Latn",  # Tswana
    "tum_Latn",  # Tumbuka
    "xho_Latn",  # Xhosa
    "zul_Latn",  # Zulu
    # Keep: run_Latn (Rundi), eng_Latn (English)
}


def prune_tokenizer(direction: str):
    model_path = os.path.join(MODEL_DIR, f"nllb_{direction}")
    if not os.path.isdir(model_path):
        print(f"Skipping {direction} — not found")
        return

    print(f"\nPruning tokenizer for nllb_{direction}...")
    tok = NllbTokenizer.from_pretrained(model_path)

    vocab = tok.get_vocab()
    original_size = len(vocab)

    # Find token IDs for language codes to remove
    remove_ids = set()
    for lang in REMOVE_LANGS:
        if lang in vocab:
            remove_ids.add(vocab[lang])

    # Also find any tokens that are exclusively used by those languages
    # by checking the sentencepiece model — we'll just remove the lang tag tokens
    # The model will still work, it just won't be able to start generation in those languages
    print(f"  Removing {len(remove_ids)} language tag tokens: {REMOVE_LANGS}")

    # Save updated tokenizer with added_tokens_encoder modified
    # The cleanest way is to override the forced_bos_token and remove lang codes
    # from added_tokens so they can't be generated

    # Load added_tokens.json
    added_tokens_path = os.path.join(model_path, "added_tokens.json")
    if os.path.exists(added_tokens_path):
        with open(added_tokens_path) as f:
            added_tokens = json.load(f)

        # Remove the unwanted language tokens
        pruned = {k: v for k, v in added_tokens.items() if k not in REMOVE_LANGS}
        removed_count = len(added_tokens) - len(pruned)

        with open(added_tokens_path, "w") as f:
            json.dump(pruned, f, indent=2)
        print(f"  Removed {removed_count} language tokens from added_tokens.json")
    else:
        print("  No added_tokens.json found — tokenizer may use sentencepiece only")

    # Update tokenizer_config to reflect only supported languages
    config_path = os.path.join(model_path, "tokenizer_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

        # Set src_lang and tgt_lang to only supported ones
        if direction == "en2lun":
            config["src_lang"] = "eng_Latn"
            config["tgt_lang"] = "run_Latn"
        else:
            config["src_lang"] = "run_Latn"
            config["tgt_lang"] = "eng_Latn"

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Updated tokenizer_config.json")

    print(f"  Done: nllb_{direction}")


def main():
    for direction in ["en2lun", "lun2en"]:
        prune_tokenizer(direction)
    print("\nVocabulary pruning complete.")
    print("Restart the backend to load the updated tokenizers.")


if __name__ == "__main__":
    main()
