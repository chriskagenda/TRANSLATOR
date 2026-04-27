"""
Extracts English <-> Runyoro-Rutooro sentence pairs from OCR grammar data
and merges them into english_nyoro_clean.csv for retraining.

The grammar books contain hundreds of parallel examples like:
  "Omukazi tabohera bintu mukisiika."
  "A woman does not pack up things in the inner sitting room."

Run:
    python extract_ocr_pairs.py

Output:
    data/cleaned/english_nyoro_clean.csv  (updated with OCR-extracted pairs)
    data/cleaned/ocr_pairs_extracted.csv  (extracted pairs only, for review)
"""
import os, re, json
import pandas as pd

BASE      = os.path.dirname(__file__)
OCR_FILE  = os.path.join(BASE, "data", "OCR", "combined", "all_ocr_combined.json")
CLEAN_DIR = os.path.join(BASE, "data", "cleaned")
OUT_CSV   = os.path.join(CLEAN_DIR, "ocr_pairs_extracted.csv")
MAIN_CSV  = os.path.join(CLEAN_DIR, "english_nyoro_clean.csv")

# ── Runyoro-Rutooro detection heuristics ─────────────────────────────────────
# Common noun class prefixes and verb markers
RUNYORO_PREFIXES = (
    "oku", "omu", "aba", "eki", "ebi", "ama", "obu", "oru", "aka", "utu",
    "emi", "eri", "ery", "en", "em", "ni", "ba", "ka", "tu", "mu",
)
RUNYORO_PATTERNS = _re = re.compile(
    r'\b(oku|omu|aba|eki|ebi|ama|obu|oru|aka|utu|emi|eri|en[tk]|em[bp]|'
    r'omw|obw|okw|orw|ekw|abw|ngu|nga|hali|omu|kandi|baitu|nukwo|'
    r'obw|buli|hamu|muno|bwangu|mpola)\b',
    re.IGNORECASE
)

def looks_like_runyoro(text: str) -> bool:
    """Return True if text looks like Runyoro-Rutooro."""
    if not text or len(text) < 5:
        return False
    # Must have some Runyoro markers
    matches = RUNYORO_PATTERNS.findall(text)
    words = text.split()
    if not words:
        return False
    ratio = len(matches) / len(words)
    return ratio >= 0.15

def looks_like_english(text: str) -> bool:
    """Return True if text looks like English."""
    if not text or len(text) < 5:
        return False
    # Common English words
    en_words = re.compile(
        r'\b(the|a|an|is|are|was|were|has|have|had|will|would|can|could|'
        r'should|may|might|this|that|these|those|it|he|she|they|we|you|'
        r'of|in|to|for|with|on|at|from|by|not|be|do|does|did|said|'
        r'which|who|what|when|where|how|if|but|and|or|so|as|than)\b',
        re.IGNORECASE
    )
    words = text.split()
    if not words:
        return False
    matches = en_words.findall(text)
    ratio = len(matches) / len(words)
    return ratio >= 0.15

def clean_text(t: str) -> str:
    t = t.strip().strip('"').strip("'").strip()
    t = re.sub(r'\s+', ' ', t)
    # Remove leading page numbers / chapter refs
    t = re.sub(r'^[\d\s]+Chapter[^:]+:', '', t).strip()
    t = re.sub(r'^\d+\s+', '', t).strip()
    return t

def is_valid_pair(en: str, lun: str) -> bool:
    if len(en) < 8 or len(lun) < 8:
        return False
    if en.lower() == lun.lower():
        return False
    if len(en.split()) < 2 or len(lun.split()) < 2:
        return False
    # English side must look like English
    if not looks_like_english(en):
        return False
    # Lunyoro side must have some Runyoro markers
    if not looks_like_runyoro(lun):
        return False
    # Reject if Lunyoro side is mostly English
    if looks_like_english(lun) and not looks_like_runyoro(lun):
        return False
    return True

# ── Extraction strategies ─────────────────────────────────────────────────────

def extract_quoted_pairs(text: str) -> list[tuple[str, str]]:
    """
    Extract pairs where a Runyoro sentence is followed by its English translation
    in single quotes, e.g.:
      Omukazi tabohera bintu mukisiika.
      'A woman does not pack up things in the inner sitting room.'
    """
    pairs = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    for i, line in enumerate(lines):
        # Pattern: line in single quotes = English translation
        m = re.match(r"^['\u2018\u2019\u201c\u201d](.+)['\u2018\u2019\u201c\u201d]\.?$", line)
        if m:
            en = clean_text(m.group(1))
            # Look back for the Runyoro sentence (1-2 lines up)
            for back in range(1, 3):
                if i - back >= 0:
                    candidate = clean_text(lines[i - back])
                    if looks_like_runyoro(candidate) and not looks_like_english(candidate):
                        if is_valid_pair(en, candidate):
                            pairs.append((en, candidate))
                        break
    return pairs


def extract_inline_translation_pairs(text: str) -> list[tuple[str, str]]:
    """
    Extract pairs where Runyoro and English appear on the same line separated by
    common delimiters used in grammar books:
      word 'English meaning'
      Runyoro sentence. 'English translation.'
      Runyoro sentence (English translation)
    """
    pairs = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    for line in lines:
        # Pattern: text 'quoted translation'
        m = re.match(
            r"^(.+?)\s+['\u2018\u2019\u201c\u201d\"](.+?)['\u2018\u2019\u201c\u201d\"]\.?$",
            line
        )
        if m:
            part1 = clean_text(m.group(1))
            part2 = clean_text(m.group(2))
            # Determine which is Runyoro and which is English
            if looks_like_runyoro(part1) and looks_like_english(part2):
                if is_valid_pair(part2, part1):
                    pairs.append((part2, part1))
            elif looks_like_english(part1) and looks_like_runyoro(part2):
                if is_valid_pair(part1, part2):
                    pairs.append((part1, part2))

        # Pattern: Runyoro. (English translation)
        m2 = re.match(r"^(.+?)\s+\((.+?)\)\.?$", line)
        if m2:
            part1 = clean_text(m2.group(1))
            part2 = clean_text(m2.group(2))
            if looks_like_runyoro(part1) and looks_like_english(part2):
                if is_valid_pair(part2, part1):
                    pairs.append((part2, part1))

    return pairs


def extract_adjacent_pairs(text: str) -> list[tuple[str, str]]:
    """
    Extract pairs from adjacent lines where one is Runyoro and the next is English
    (or vice versa). Common pattern in grammar books.
    """
    pairs = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    for i in range(len(lines) - 1):
        a = clean_text(lines[i])
        b = clean_text(lines[i + 1])

        if len(a) < 8 or len(b) < 8:
            continue

        # Runyoro line followed by English line
        if looks_like_runyoro(a) and not looks_like_runyoro(b) and looks_like_english(b):
            if is_valid_pair(b, a):
                pairs.append((b, a))
        # English line followed by Runyoro line
        elif looks_like_english(a) and not looks_like_english(b) and looks_like_runyoro(b):
            if is_valid_pair(a, b):
                pairs.append((a, b))

    return pairs


def extract_from_text(text: str) -> list[tuple[str, str]]:
    """Run all extraction strategies on a page of text."""
    all_pairs = []
    all_pairs.extend(extract_quoted_pairs(text))
    all_pairs.extend(extract_inline_translation_pairs(text))
    all_pairs.extend(extract_adjacent_pairs(text))
    # Deduplicate within this page
    seen = set()
    unique = []
    for en, lun in all_pairs:
        key = (en.lower()[:60], lun.lower()[:60])
        if key not in seen:
            seen.add(key)
            unique.append((en, lun))
    return unique

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading OCR data from {OCR_FILE}...")
    with open(OCR_FILE, encoding="utf-8") as f:
        ocr_data = json.load(f)

    all_pairs: list[tuple[str, str]] = []
    source_counts: dict[str, int] = {}

    for source, pages in ocr_data.items():
        source_pairs = []
        for page_key, text in pages.items():
            if not isinstance(text, str):
                text = "\n".join(text)
            page_pairs = extract_from_text(text)
            source_pairs.extend(page_pairs)

        # Deduplicate per source
        seen = set()
        unique_source = []
        for en, lun in source_pairs:
            key = (en.lower()[:60], lun.lower()[:60])
            if key not in seen:
                seen.add(key)
                unique_source.append((en, lun))

        source_counts[source] = len(unique_source)
        all_pairs.extend(unique_source)
        print(f"  {source}: {len(unique_source)} pairs")

    # Global deduplication
    seen_global = set()
    unique_all = []
    for en, lun in all_pairs:
        key = (en.lower()[:60], lun.lower()[:60])
        if key not in seen_global:
            seen_global.add(key)
            unique_all.append({"english": en, "lunyoro": lun})

    print(f"\nTotal unique pairs extracted: {len(unique_all)}")

    # Save extracted pairs for review
    df_new = pd.DataFrame(unique_all)
    df_new.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Saved extracted pairs to {OUT_CSV}")

    # Merge into main training CSV
    df_main = pd.read_csv(MAIN_CSV)
    before = len(df_main)

    df_combined = pd.concat([df_main, df_new], ignore_index=True)
    df_combined["english"] = df_combined["english"].astype(str).str.strip()
    df_combined["lunyoro"] = df_combined["lunyoro"].astype(str).str.strip()
    df_combined = df_combined[
        (df_combined["english"].str.len() >= 8) &
        (df_combined["lunyoro"].str.len() >= 8)
    ]
    df_combined = df_combined.drop_duplicates(subset=["english", "lunyoro"])
    df_combined = df_combined.reset_index(drop=True)
    df_combined.to_csv(MAIN_CSV, index=False, encoding="utf-8")

    added = len(df_combined) - before
    print(f"Training pairs: {before:,} -> {len(df_combined):,} (+{added:,} new)")
    print(f"\nNext step: python prepare_training_data.py && python improve_and_retrain.py")


if __name__ == "__main__":
    main()
