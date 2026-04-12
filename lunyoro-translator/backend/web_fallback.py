"""
Web fallback for missing translations.
When a word/phrase is not found in the local dataset or models,
search runyorodictionary.com and other Runyoro sources via web scraping.
"""
import os
import re
import urllib.request
import urllib.parse

# Offline mode flag — respect the app's offline setting
_OFFLINE = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"

# Known static entries from runyorodictionary.com (homepage content)
# These supplement the dataset for common words not in training data
STATIC_ENTRIES: dict[str, str] = {
    # Numbers
    "one": "emu", "two": "ibiri", "three": "isatu", "four": "ina",
    "five": "itano", "six": "mukaaga", "seven": "musanju", "eight": "munaana",
    "nine": "mwenda", "ten": "ikumi", "hundred": "kikumi", "thousand": "rukumi",
    "million": "akakaikuru",
    # Interjections / common expressions
    "thank you": "webare", "thanks": "webare", "welcome": "karibu",
    "goodbye": "genda kurungi", "yes": "ego", "no": "nangwa",
    "enough": "bbaasi", "indeed": "weza", "my god": "ruhanga wange",
    "surprise": "mawe", "sympathy": "bambi", "satisfaction": "haakiri",
    # Common greetings
    "how are you": "oraire otya", "good morning": "osiibire otya",
    "good evening": "waliire otya", "peace": "mirembe",
    # Empaako
    "shining star": "atwooki", "beautiful": "ateenyi", "beloved": "abbooki",
    "warrior": "abaala", "royalty": "okaali",
    # Common words from idioms
    "early morning": "kuburorra mu rwigi",
    "as fast as possible": "maguru nkakwimaki",
    "satisfied": "omutima gwezire",
    "worried": "omutima guli enyuma",
    "shameless": "amaiso ga kimpenkirye",
}

# Reverse map: Lunyoro → English
STATIC_ENTRIES_REVERSE: dict[str, str] = {v: k for k, v in STATIC_ENTRIES.items()}


def lookup_static(word: str, direction: str = "en→lun") -> str | None:
    """Check static entries from runyorodictionary.com for a word."""
    w = word.lower().strip()
    if direction == "en→lun":
        return STATIC_ENTRIES.get(w)
    else:
        return STATIC_ENTRIES_REVERSE.get(w)


def web_search_fallback(word: str, direction: str = "en→lun") -> str | None:
    """
    Try to find a translation by searching runyorodictionary.com via Google.
    Only runs when not in offline mode.
    Returns the best candidate translation or None.
    """
    if _OFFLINE:
        return None

    try:
        site = "runyorodictionary.com"
        lang = "Runyoro Rutooro" if direction == "en→lun" else "English"
        query = f'site:{site} "{word}" {lang}'
        encoded = urllib.parse.quote(query)
        url = f"https://www.google.com/search?q={encoded}&num=3"

        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; LunyoroTranslator/1.0)"
        })
        with urllib.request.urlopen(req, timeout=5) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        # Extract snippets from Google results
        snippets = re.findall(r'<div[^>]*class="[^"]*BNeawe[^"]*"[^>]*>([^<]{5,200})</div>', html)
        for snippet in snippets[:5]:
            # Look for patterns like "word - translation" or "word: translation"
            match = re.search(
                rf'{re.escape(word)}\s*[-:–]\s*([A-Za-z\s\']+)',
                snippet, re.IGNORECASE
            )
            if match:
                candidate = match.group(1).strip()
                if 2 < len(candidate) < 60:
                    return candidate
    except Exception:
        pass

    return None
