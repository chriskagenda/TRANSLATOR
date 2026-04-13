"""
Runyoro/Rutooro R/L corrector.

Rules:
  1. L is used at word-initial position when followed by 'e' or 'i'
     e.g. leero, liiso, liino
  2. L is used when preceded by vowel a/o/u and followed by i/e
     e.g. a-liire, o-liire, tu-liire, mu-liire
  3. L does NOT follow e or i — use R instead
     e.g. eriire (not eliire), iriire (not iliire)
  4. All other positions use R (dominant sound)
"""
import re


def _fix_word(word: str) -> str:
    """Apply R/L rules to a single word."""
    if not word:
        return word

    chars = list(word.lower())
    result = list(word)  # preserve original case

    vowels_all  = set("aeiou")
    vowels_aou  = set("aou")   # trigger L before i/e
    vowels_ei   = set("ei")    # L does NOT follow these

    for i, ch in enumerate(chars):
        if ch not in ("r", "l"):
            continue

        prev_ch = chars[i - 1] if i > 0 else None
        next_ch = chars[i + 1] if i < len(chars) - 1 else None

        # Rule 1: word-initial L before e or i → keep L
        if i == 0 and next_ch in vowels_ei:
            result[i] = "L" if word[i].isupper() else "l"
            continue

        # Rule 3: L/R after e or i → must be R
        if prev_ch in vowels_ei:
            result[i] = "R" if word[i].isupper() else "r"
            continue

        # Rule 2: after a/o/u and before i/e → use L
        if prev_ch in vowels_aou and next_ch in vowels_ei:
            result[i] = "L" if word[i].isupper() else "l"
            continue

        # Rule 4: all other positions → use R
        if ch == "l":
            result[i] = "R" if word[i].isupper() else "r"

    return "".join(result)


def correct_rl(text: str) -> str:
    """Apply R/L rules to all words in a Lunyoro/Rutooro text."""
    if not text:
        return text

    # Split on word boundaries, preserving punctuation and spaces
    tokens = re.split(r"(\W+)", text)
    corrected = []
    for token in tokens:
        if re.match(r"[A-Za-z]+", token):
            corrected.append(_fix_word(token))
        else:
            corrected.append(token)
    return "".join(corrected)


if __name__ == "__main__":
    # Quick test
    tests = [
        ("leero",   "leero"),    # rule 1 — keep L
        ("liiso",   "liiso"),    # rule 1 — keep L
        ("liino",   "liino"),    # rule 1 — keep L
        ("aliire",  "aliire"),   # rule 2 — keep L
        ("oliire",  "oliire"),   # rule 2 — keep L
        ("tuliire", "tuliire"),  # rule 2 — keep L
        ("muliire", "muliire"),  # rule 2 — keep L
        ("eliire",  "eriire"),   # rule 3 — L→R after e
        ("iliire",  "iriire"),   # rule 3 — L→R after i
        ("eriire",  "eriire"),   # rule 3 — already correct
        ("orulimi", "orulimi"),  # rule 2 — u before i → keep L (tongue = orulimi)
    ]
    all_pass = True
    for inp, expected in tests:
        got = correct_rl(inp)
        status = "✓" if got == expected else "✗"
        if got != expected:
            all_pass = False
        print(f"  {status} {inp} → {got}  (expected: {expected})")
    print("All tests passed!" if all_pass else "Some tests failed.")
