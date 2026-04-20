"""
Cleans the back-translated pairs already in english_nyoro_clean.csv.
Removes hallucinated, repeated, too-short, or garbage synthetic pairs.
Keeps all original (non-synthetic) pairs untouched.

Run:
    python clean_backtranslated.py
"""
import os
import re
import pandas as pd
import unicodedata

DATA    = os.path.join(os.path.dirname(__file__), "data", "cleaned")
CSV     = os.path.join(DATA, "english_nyoro_clean.csv")


def is_bad_synthetic(en: str, lun: str) -> bool:
    """Return True if this back-translated pair should be discarded."""
    en  = str(en).strip()
    lun = str(lun).strip()

    # Too short
    if len(en) < 8 or len(lun) < 8:
        return True

    # Lunyoro side is just a single word or very short (dictionary entry artifact)
    if len(lun.split()) < 2:
        return True

    # Copy of source (model didn't translate)
    if en.lower() == lun.lower():
        return True

    # English suspiciously short vs Lunyoro (failed translation)
    en_words  = en.split()
    lun_words = lun.split()
    if len(en_words) < len(lun_words) * 0.25:
        return True

    # Repeated word runs — hallucination signal
    if len(en_words) > 3:
        bigrams = list(zip(en_words, en_words[1:]))
        repeat_ratio = sum(1 for a, b in bigrams if a.lower() == b.lower()) / len(bigrams)
        if repeat_ratio > 0.3:
            return True

    # Repeated n-gram phrases (e.g. "and and and" or "the the the")
    if re.search(r'\b(\w+)\s+\1\s+\1\b', en, re.IGNORECASE):
        return True

    # Non-ASCII garbage (wrong script output)
    ascii_ratio = sum(1 for c in en if ord(c) < 128) / max(len(en), 1)
    if ascii_ratio < 0.85:
        return True

    # English output contains Lunyoro-looking tokens (model code-switched)
    lunyoro_markers = ("oku", "omu", "aba", "ebi", "eki", "ama", "ngu", "nga")
    en_lower = en.lower()
    lun_token_count = sum(1 for w in en_lower.split() if any(w.startswith(m) for m in lunyoro_markers))
    if lun_token_count > len(en_words) * 0.3:
        return True

    return False


def main():
    print(f"Loading {CSV}...")
    df = pd.read_csv(CSV).fillna("")

    total_before = len(df)
    print(f"Total pairs before cleaning: {total_before:,}")

    mask_bad = df.apply(
        lambda row: is_bad_synthetic(row["english"], row["lunyoro"]), axis=1
    )

    removed = mask_bad.sum()
    df_clean = df[~mask_bad].reset_index(drop=True)

    print(f"Removed {removed:,} bad pairs ({removed/total_before*100:.1f}%)")

    # ── Round-trip consistency check ─────────────────────────────────────────
    # Translate synthetic English back to Lunyoro, check similarity to original
    print("\nRunning round-trip consistency check on synthetic pairs...")
    try:
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from translate import _mt_translate, _load_mt
        from rapidfuzz import fuzz
        from sentence_transformers import SentenceTransformer, util as st_util
        import numpy as np

        _load_mt("en2lun")

        # Load semantic model for meaning-level similarity
        sem_model_path = os.path.join(os.path.dirname(__file__), "model", "sem_model")
        sem_model = SentenceTransformer(sem_model_path)

        is_synthetic = ~df_clean["english"].str.startswith("[")
        synthetic_idx = df_clean[is_synthetic].index.tolist()
        print(f"  Checking {len(synthetic_idx):,} synthetic pairs...")

        bad_roundtrip = set()
        for i, idx in enumerate(synthetic_idx):
            en  = df_clean.at[idx, "english"]
            lun = df_clean.at[idx, "lunyoro"]

            # Translate English back to Lunyoro
            back = _mt_translate(en, "en2lun")
            if back:
                # 1. Surface similarity (catches obvious mismatches)
                surface_sim = fuzz.token_sort_ratio(back.lower(), lun.lower())

                # 2. Semantic similarity (catches fluent but wrong translations)
                orig_emb = sem_model.encode(lun,  convert_to_numpy=True)
                back_emb = sem_model.encode(back, convert_to_numpy=True)
                semantic_sim = float(st_util.cos_sim(orig_emb, back_emb)[0][0]) * 100

                # 3. Length ratio check (catches partial translations)
                en_words  = len(en.split())
                lun_words = len(lun.split())
                length_ok = en_words >= lun_words * 0.5

                # Discard if both surface AND semantic similarity are low,
                # or if length ratio suggests partial translation
                if (surface_sim < 25 and semantic_sim < 40) or not length_ok:
                    bad_roundtrip.add(idx)

            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(synthetic_idx)} checked, {len(bad_roundtrip)} flagged...")

        print(f"  Round-trip removed: {len(bad_roundtrip):,} more pairs")
        df_clean = df_clean.drop(index=list(bad_roundtrip)).reset_index(drop=True)
        removed += len(bad_roundtrip)

    except Exception as e:
        print(f"  Round-trip check skipped: {e}")

    print(f"\nTotal removed: {removed:,} ({removed/total_before*100:.1f}%)")
    print(f"Remaining: {len(df_clean):,} pairs")

    df_clean.to_csv(CSV, index=False)
    print(f"Saved cleaned data to {CSV}")
    print("\nNow run: python prepare_training_data.py to rebuild training splits")


if __name__ == "__main__":
    main()
