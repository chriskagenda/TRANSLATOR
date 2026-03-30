"""
Translation logic:
1. Exact match lookup in sentence pairs
2. Semantic similarity search (embeddings)
3. Word-level dictionary lookup fallback
"""
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz, process
from preprocess import load_dictionary

INDEX_PATH = os.path.join(os.path.dirname(__file__), "model", "translation_index.pkl")

_index = None
_model = None
_dictionary = None


def _load():
    global _index, _model, _dictionary
    if _index is None:
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError("Translation index not found. Run train.py first.")
        with open(INDEX_PATH, "rb") as f:
            _index = pickle.load(f)
        _model = SentenceTransformer(_index["model_name"])
        _dictionary = _index["dictionary"]


def translate(text: str, top_k: int = 3) -> dict:
    _load()
    text = text.strip()

    # 1. Exact match
    english_sentences = _index["english_sentences"]
    lunyoro_sentences = _index["lunyoro_sentences"]

    lower_text = text.lower()
    for i, sent in enumerate(english_sentences):
        if sent.lower() == lower_text:
            return {
                "translation": lunyoro_sentences[i],
                "method": "exact_match",
                "confidence": 1.0,
                "alternatives": [],
            }

    # 2. Semantic similarity
    query_embedding = _model.encode(text, convert_to_tensor=True)
    corpus_embeddings = _index["embeddings"]

    scores = util.cos_sim(query_embedding, corpus_embeddings)[0].numpy()
    top_indices = np.argsort(scores)[::-1][:top_k]

    best_idx = top_indices[0]
    best_score = float(scores[best_idx])

    alternatives = [
        {
            "english": english_sentences[i],
            "lunyoro": lunyoro_sentences[i],
            "score": round(float(scores[i]), 3),
        }
        for i in top_indices[1:]
    ]

    # if confidence is reasonable, return semantic match
    if best_score > 0.5:
        return {
            "translation": lunyoro_sentences[best_idx],
            "method": "semantic_match",
            "confidence": round(best_score, 3),
            "matched_english": english_sentences[best_idx],
            "alternatives": alternatives,
        }

    # 3. Dictionary word lookup fallback
    words = text.lower().split()
    dict_words = [d["word"] for d in _dictionary]
    found = []
    for word in words:
        match = process.extractOne(word, dict_words, scorer=fuzz.ratio, score_cutoff=80)
        if match:
            entry = next((d for d in _dictionary if d["word"] == match[0]), None)
            if entry:
                found.append({"english_word": word, "lunyoro_word": entry["word"],
                               "definition": entry.get("definitionNative", "")})

    return {
        "translation": None,
        "method": "dictionary_fallback",
        "confidence": round(best_score, 3),
        "matched_english": english_sentences[best_idx],
        "alternatives": alternatives,
        "dictionary_matches": found,
        "message": "No close translation found. Showing closest matches.",
    }


def lookup_word(word: str) -> list:
    _load()
    word_lower = word.lower()
    results = []
    for entry in _dictionary:
        if entry["word"].lower() == word_lower or word_lower in entry.get("definitionEnglish", "").lower():
            results.append(entry)
    return results[:5]


def translate_to_english(text: str, top_k: int = 3) -> dict:
    """Reverse translation: Lunyoro/Rutooro → English"""
    _load()
    text = text.strip()

    english_sentences = _index["english_sentences"]
    lunyoro_sentences = _index["lunyoro_sentences"]

    # 1. Exact match in lunyoro sentences
    lower_text = text.lower()
    for i, sent in enumerate(lunyoro_sentences):
        if sent.lower() == lower_text:
            return {
                "translation": english_sentences[i],
                "method": "exact_match",
                "confidence": 1.0,
                "alternatives": [],
            }

    # 2. Semantic similarity against lunyoro corpus
    if "lunyoro_embeddings" not in _index:
        # build lunyoro embeddings on first reverse call and cache in memory
        _index["lunyoro_embeddings"] = _model.encode(lunyoro_sentences, show_progress_bar=False, batch_size=64)

    query_embedding = _model.encode(text, convert_to_tensor=True)
    lunyoro_embeddings = _index["lunyoro_embeddings"]

    scores = util.cos_sim(query_embedding, lunyoro_embeddings)[0].numpy()
    top_indices = np.argsort(scores)[::-1][:top_k]

    best_idx = top_indices[0]
    best_score = float(scores[best_idx])

    alternatives = [
        {
            "lunyoro": lunyoro_sentences[i],
            "english": english_sentences[i],
            "score": round(float(scores[i]), 3),
        }
        for i in top_indices[1:]
    ]

    if best_score > 0.5:
        return {
            "translation": english_sentences[best_idx],
            "method": "semantic_match",
            "confidence": round(best_score, 3),
            "matched_lunyoro": lunyoro_sentences[best_idx],
            "alternatives": alternatives,
        }

    # 3. Dictionary fallback — match against lunyoro words
    words = text.lower().split()
    dict_words = [d["word"] for d in _dictionary]
    found = []
    for word in words:
        match = process.extractOne(word, dict_words, scorer=fuzz.ratio, score_cutoff=75)
        if match:
            entry = next((d for d in _dictionary if d["word"] == match[0]), None)
            if entry and entry.get("definitionEnglish"):
                found.append({
                    "lunyoro_word": entry["word"],
                    "english_definition": entry["definitionEnglish"],
                })

    return {
        "translation": None,
        "method": "dictionary_fallback",
        "confidence": round(best_score, 3),
        "matched_lunyoro": lunyoro_sentences[best_idx],
        "alternatives": alternatives,
        "dictionary_matches": found,
        "message": "No close translation found. Showing closest matches.",
    }


def get_index_and_model():
    """Expose loaded index and model for batch operations."""
    _load()
    return _index, _model


def spellcheck(text: str) -> list:
    """
    Check each word in the Lunyoro/Rutooro text against the dictionary.
    Only flags words that have no reasonable match in the known corpus.
    """
    _load()
    import re

    # build known word set from dictionary entries + all individual words in lunyoro sentences
    known_words = set(d["word"].lower() for d in _dictionary if d["word"])
    for sent in _index["lunyoro_sentences"]:
        for w in re.findall(r"[a-zA-Z']+", sent):
            if len(w) >= 3:
                known_words.add(w.lower())

    dict_word_list = list(known_words)

    tokens = re.findall(r"[a-zA-Z']+", text)
    misspelled = []

    for token in tokens:
        lower = token.lower()

        # skip very short words — too ambiguous to check
        if len(lower) < 4:
            continue

        # exact match — definitely correct
        if lower in known_words:
            continue

        # fuzzy match — if best score >= 85 treat as a known variant (morphology, prefix, etc.)
        best = process.extractOne(lower, dict_word_list, scorer=fuzz.ratio)
        if best and best[1] >= 85:
            continue

        # genuinely unknown — suggest closest matches above 65
        suggestions = process.extract(lower, dict_word_list, scorer=fuzz.ratio, limit=3, score_cutoff=65)
        misspelled.append({
            "word": token,
            "suggestions": [s[0] for s in suggestions],
        })

    return misspelled
