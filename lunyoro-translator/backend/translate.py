"""
Translation logic:
  Primary  — fine-tuned MarianMT models (en2lun / lun2en) when available
  Fallback — semantic similarity retrieval + dictionary lookup
"""
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz, process
from preprocess import load_dictionary

INDEX_PATH = os.path.join(os.path.dirname(__file__), "model", "translation_index.pkl")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "model")
SEM_MODEL_DIR = os.path.join(MODEL_DIR, "sem_model")

# Force fully offline mode — no network calls to HuggingFace
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ── cached singletons ────────────────────────────────────────────────────────
_index      = None
_sem_model  = None
_dictionary = None
_corpus_vocab = None

_mt_models     = {}   # {"en2lun": (tokenizer, model), "lun2en": (tokenizer, model)}
_mt_available  = {}   # {"en2lun": bool, "lun2en": bool}


# ── loaders ──────────────────────────────────────────────────────────────────

def _load_retrieval():
    global _index, _sem_model, _dictionary, _corpus_vocab
    if _index is not None:
        return
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("Translation index not found. Run train.py first.")
    with open(INDEX_PATH, "rb") as f:
        _index = pickle.load(f)
    # Load from local copy first; fall back to cached name if local copy missing
    sem_path = SEM_MODEL_DIR if os.path.isdir(SEM_MODEL_DIR) else _index["model_name"]
    _sem_model  = SentenceTransformer(sem_path)
    _dictionary = _index["dictionary"]


def _load_mt(direction: str):
    """Lazy-load a fine-tuned MarianMT model. Returns True if available."""
    if direction in _mt_available:
        return _mt_available[direction]

    path = os.path.join(MODEL_DIR, direction)
    if not os.path.isdir(path):
        _mt_available[direction] = False
        return False

    try:
        from transformers import MarianMTModel, MarianTokenizer
        import torch
        tokenizer = MarianTokenizer.from_pretrained(path)
        model     = MarianMTModel.from_pretrained(path)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        _mt_models[direction]    = (tokenizer, model, device)
        _mt_available[direction] = True
        print(f"[translate] Loaded fine-tuned model: {direction}")
        return True
    except Exception as e:
        print(f"[translate] Could not load {direction} model: {e}")
        _mt_available[direction] = False
        return False


def _mt_translate(text: str, direction: str) -> str | None:
    """Run inference with a fine-tuned MarianMT model."""
    if not _load_mt(direction):
        return None
    import torch
    tokenizer, model, device = _mt_models[direction]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            num_beams=4,
            max_length=256,
            early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ── public API ───────────────────────────────────────────────────────────────

def translate(text: str, top_k: int = 3) -> dict:
    """English → Lunyoro/Rutooro"""
    text = text.strip()

    # 1. Fine-tuned model (best quality)
    mt_result = _mt_translate(text, "en2lun")
    if mt_result:
        return {
            "translation": mt_result,
            "method":      "neural_mt",
            "confidence":  1.0,
            "alternatives": [],
        }

    # 2. Retrieval fallback
    _load_retrieval()
    english_sentences  = _index["english_sentences"]
    lunyoro_sentences  = _index["lunyoro_sentences"]

    # exact match
    lower = text.lower()
    for i, sent in enumerate(english_sentences):
        if sent.lower() == lower:
            return {"translation": lunyoro_sentences[i], "method": "exact_match",
                    "confidence": 1.0, "alternatives": []}

    # semantic similarity
    q_emb   = _sem_model.encode(text, convert_to_tensor=True)
    scores  = util.cos_sim(q_emb, _index["embeddings"])[0].numpy()
    top_idx = np.argsort(scores)[::-1][:top_k]
    best, best_score = top_idx[0], float(scores[top_idx[0]])

    alternatives = [{"english": english_sentences[i], "lunyoro": lunyoro_sentences[i],
                     "score": round(float(scores[i]), 3)} for i in top_idx[1:]]

    if best_score > 0.5:
        return {"translation": lunyoro_sentences[best], "method": "semantic_match",
                "confidence": round(best_score, 3),
                "matched_english": english_sentences[best], "alternatives": alternatives}

    # 3. Dictionary fallback
    return _dict_fallback(text, best_score, english_sentences[best], alternatives, "en→lun")


def translate_to_english(text: str, top_k: int = 3) -> dict:
    """Lunyoro/Rutooro → English"""
    text = text.strip()

    # 1. Fine-tuned model
    mt_result = _mt_translate(text, "lun2en")
    if mt_result:
        return {
            "translation": mt_result,
            "method":      "neural_mt",
            "confidence":  1.0,
            "alternatives": [],
        }

    # 2. Retrieval fallback
    _load_retrieval()
    english_sentences = _index["english_sentences"]
    lunyoro_sentences = _index["lunyoro_sentences"]

    lower = text.lower()
    for i, sent in enumerate(lunyoro_sentences):
        if sent.lower() == lower:
            return {"translation": english_sentences[i], "method": "exact_match",
                    "confidence": 1.0, "alternatives": []}

    if "lunyoro_embeddings" not in _index:
        _index["lunyoro_embeddings"] = _sem_model.encode(
            lunyoro_sentences, show_progress_bar=False, batch_size=64
        )

    q_emb   = _sem_model.encode(text, convert_to_tensor=True)
    scores  = util.cos_sim(q_emb, _index["lunyoro_embeddings"])[0].numpy()
    top_idx = np.argsort(scores)[::-1][:top_k]
    best, best_score = top_idx[0], float(scores[top_idx[0]])

    alternatives = [{"lunyoro": lunyoro_sentences[i], "english": english_sentences[i],
                     "score": round(float(scores[i]), 3)} for i in top_idx[1:]]

    if best_score > 0.5:
        return {"translation": english_sentences[best], "method": "semantic_match",
                "confidence": round(best_score, 3),
                "matched_lunyoro": lunyoro_sentences[best], "alternatives": alternatives}

    return _dict_fallback_reverse(text, best_score, lunyoro_sentences[best], alternatives)


def _dict_fallback(text, best_score, matched_english, alternatives, direction):
    _load_retrieval()
    words      = text.lower().split()
    dict_words = [d["word"] for d in _dictionary]
    found = []
    for word in words:
        match = process.extractOne(word, dict_words, scorer=fuzz.ratio, score_cutoff=80)
        if match:
            entry = next((d for d in _dictionary if d["word"] == match[0]), None)
            if entry:
                found.append({"english_word": word, "lunyoro_word": entry["word"],
                               "definition": entry.get("definitionNative", "")})
    return {"translation": None, "method": "dictionary_fallback",
            "confidence": round(best_score, 3), "matched_english": matched_english,
            "alternatives": alternatives, "dictionary_matches": found,
            "message": "No close translation found. Showing closest matches."}


def _dict_fallback_reverse(text, best_score, matched_lunyoro, alternatives):
    _load_retrieval()
    words      = text.lower().split()
    dict_words = [d["word"] for d in _dictionary]
    found = []
    for word in words:
        match = process.extractOne(word, dict_words, scorer=fuzz.ratio, score_cutoff=75)
        if match:
            entry = next((d for d in _dictionary if d["word"] == match[0]), None)
            if entry and entry.get("definitionEnglish"):
                found.append({"lunyoro_word": entry["word"],
                               "english_definition": entry["definitionEnglish"]})
    return {"translation": None, "method": "dictionary_fallback",
            "confidence": round(best_score, 3), "matched_lunyoro": matched_lunyoro,
            "alternatives": alternatives, "dictionary_matches": found,
            "message": "No close translation found. Showing closest matches."}


def lookup_word(word: str) -> list:
    _load_retrieval()
    word_lower = word.lower()
    results = []
    for entry in _dictionary:
        if entry["word"].lower() == word_lower or word_lower in entry.get("definitionEnglish", "").lower():
            results.append(entry)
    return results[:5]


def get_index_and_model():
    _load_retrieval()
    return _index, _sem_model


def _build_corpus_vocab() -> set:
    """
    Build a vocabulary of known Lunyoro/Rutooro words from:
    1. The cleaned training corpus sentences (lunyoro_sentences from the index)
    2. The MarianMT lun2en tokenizer vocabulary (SentencePiece surface forms)
    3. The dictionary word list
    """
    import re
    _load_retrieval()
    known: set[str] = set()

    # 1. All words from the cleaned corpus sentences
    for sent in _index["lunyoro_sentences"]:
        for w in re.findall(r"[a-zA-Z']+", sent):
            if len(w) >= 2:
                known.add(w.lower())

    # 2. MarianMT lun2en tokenizer vocab — surface forms that are real words
    lun2en_path = os.path.join(MODEL_DIR, "lun2en")
    if os.path.isdir(lun2en_path):
        try:
            from transformers import MarianTokenizer
            tok = MarianTokenizer.from_pretrained(lun2en_path)
            for token in tok.get_vocab().keys():
                # strip sentencepiece prefix ▁ and keep only alphabetic tokens
                clean = token.lstrip("▁").lower()
                if clean.isalpha() and len(clean) >= 2:
                    known.add(clean)
        except Exception:
            pass

    # 3. Dictionary words
    for d in _dictionary:
        if d.get("word"):
            known.add(d["word"].lower())

    return known


# cache the vocab so it's only built once
_corpus_vocab: set | None = None


def spellcheck(text: str) -> list:
    global _corpus_vocab
    import re
    _load_retrieval()

    if _corpus_vocab is None:
        _corpus_vocab = _build_corpus_vocab()

    vocab_list = list(_corpus_vocab)
    tokens = re.findall(r"[a-zA-Z']+", text)
    misspelled = []

    for token in tokens:
        lower = token.lower()
        if len(lower) < 3 or lower in _corpus_vocab:
            continue

        # Check if the MarianMT lun2en model can encode it cleanly (known subwords)
        lun2en_path = os.path.join(MODEL_DIR, "lun2en")
        model_knows = False
        if os.path.isdir(lun2en_path) and _load_mt("lun2en"):
            try:
                tokenizer, _, _ = _mt_models["lun2en"]
                pieces = tokenizer.tokenize(lower)
                # If encoded as a single piece (no UNK), the model recognises it
                if pieces and "<unk>" not in pieces and len(pieces) == 1:
                    model_knows = True
            except Exception:
                pass

        if model_knows:
            continue

        # Generate suggestions: fuzzy match against corpus vocab
        # Prefer words that share the same prefix (common in Bantu languages)
        prefix = lower[:3]
        prefix_words = [w for w in vocab_list if w.startswith(prefix)]
        candidate_pool = prefix_words if len(prefix_words) >= 10 else vocab_list

        suggestions = process.extract(
            lower, candidate_pool,
            scorer=fuzz.ratio,
            limit=5,
            score_cutoff=55,
        )
        # deduplicate and take top 3
        seen: set[str] = set()
        top: list[str] = []
        for s in sorted(suggestions, key=lambda x: -x[1]):
            if s[0] not in seen:
                seen.add(s[0])
                top.append(s[0])
            if len(top) == 3:
                break

        misspelled.append({"word": token, "suggestions": top})

    return misspelled
