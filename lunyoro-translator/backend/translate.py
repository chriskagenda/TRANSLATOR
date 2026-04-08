"""
Translation logic:
  Primary  — fine-tuned MarianMT models (en2lun / lun2en) when available
  Fallback — semantic similarity retrieval + dictionary lookup
"""
import os
import pickle
import re
import unicodedata
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

_APOSTROPHE_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'",
    "\u201C": '"', "\u201D": '"',
    "\u02BC": "'", "\u0060": "'",
})


def _normalise(text: str) -> str:
    """NFC normalise + apostrophe standardisation for consistent matching."""
    text = unicodedata.normalize("NFC", text)
    return text.translate(_APOSTROPHE_MAP)


# ── cached singletons ────────────────────────────────────────────────────────
_index        = None
_sem_model    = None
_dictionary   = None
_corpus_vocab = None
_dict_word_map: dict = {}   # lowercase word → entry, for O(1) lookup

_mt_models     = {}   # {"en2lun": (tokenizer, model), "lun2en": (tokenizer, model)}
_mt_available  = {}   # {"en2lun": bool, "lun2en": bool}
_nllb_models   = {}   # {"en2lun": (tokenizer, model), "lun2en": (tokenizer, model)}
_nllb_available = {}  # {"en2lun": bool, "lun2en": bool}

NLLB_LANG_EN  = "eng_Latn"
NLLB_LANG_LUN = "lug_Latn"


# ── loaders ──────────────────────────────────────────────────────────────────

def _load_retrieval():
    global _index, _sem_model, _dictionary, _corpus_vocab, _dict_word_map
    if _index is not None:
        return
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("Translation index not found. Run train.py first.")
    with open(INDEX_PATH, "rb") as f:
        _index = pickle.load(f)
    sem_path = SEM_MODEL_DIR if os.path.isdir(SEM_MODEL_DIR) else _index["model_name"]
    _sem_model  = SentenceTransformer(sem_path)
    _dictionary = _index["dictionary"]
    # build O(1) lookup map
    _dict_word_map = {d["word"].lower(): d for d in _dictionary}
    # also map by lowercased definitionEnglish for en→lun searches
    _dict_def_map: dict = {}
    for d in _dictionary:
        key = (d.get("definitionEnglish") or "").lower()
        if key:
            _dict_def_map[key] = d
    _index["_dict_def_map"] = _dict_def_map


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


def _mt_translate(text: str, direction: str, context: str = "") -> str | None:
    """Run inference with a fine-tuned MarianMT model.
    Optionally prepend a context sentence separated by ' ||| '.
    """
    if not _load_mt(direction):
        return None
    import torch
    tokenizer, model, device = _mt_models[direction]
    input_text = f"{context} ||| {text}" if context else text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            num_beams=4,
            max_length=512,
            early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def _load_nllb(direction: str) -> bool:
    """Lazy-load a fine-tuned NLLB model. Returns True if available."""
    if direction in _nllb_available:
        return _nllb_available[direction]

    path = os.path.join(MODEL_DIR, f"nllb_{direction}")
    if not os.path.isdir(path):
        _nllb_available[direction] = False
        return False

    try:
        from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
        import torch
        tokenizer = NllbTokenizer.from_pretrained(path)
        model = AutoModelForSeq2SeqLM.from_pretrained(path)
        model.eval()
        if torch.cuda.device_count() >= 2:
            device = "cuda:1"
        elif torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        model.to(device)
        _nllb_models[direction] = (tokenizer, model, device)
        _nllb_available[direction] = True
        print(f"[translate] Loaded NLLB model: {direction} on {device}")
        return True
    except Exception as e:
        print(f"[translate] Could not load NLLB {direction}: {e}")
        _nllb_available[direction] = False
        return False


def _nllb_translate(text: str, direction: str, context: str = "") -> str | None:
    """Run inference with a fine-tuned NLLB model."""
    if not _load_nllb(direction):
        return None
    import torch
    tokenizer, model, device = _nllb_models[direction]
    src_lang = NLLB_LANG_EN  if direction == "en2lun" else NLLB_LANG_LUN
    tgt_lang = NLLB_LANG_LUN if direction == "en2lun" else NLLB_LANG_EN
    tokenizer.src_lang = src_lang
    input_text = f"{context} ||| {text}" if context else text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256).to(device)
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            num_beams=4,
            max_length=512,
            early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ── public API ───────────────────────────────────────────────────────────────

def translate(text: str, top_k: int = 3, context: str = "") -> dict:
    """English → Lunyoro/Rutooro — uses both MarianMT and NLLB if available."""
    text = _normalise(text.strip())

    marian = _mt_translate(text, "en2lun", context=context)
    nllb   = _nllb_translate(text, "en2lun", context=context)

    if marian or nllb:
        return {
            "translation":        marian or nllb,
            "translation_nllb":   nllb,
            "translation_marian": marian,
            "method":             "neural_mt",
            "confidence":         1.0,
            "alternatives":       [],
        }

    # 2. Retrieval fallback
    _load_retrieval()
    english_sentences  = _index["english_sentences"]
    lunyoro_sentences  = _index["lunyoro_sentences"]

    lower = text.lower()
    for i, sent in enumerate(english_sentences):
        if sent.lower() == lower:
            return {"translation": lunyoro_sentences[i], "method": "exact_match",
                    "confidence": 1.0, "alternatives": []}

    q_emb   = _sem_model.encode(text, convert_to_numpy=True)
    scores  = util.cos_sim(q_emb, _index["embeddings"])[0].numpy()
    top_idx = np.argsort(scores)[::-1][:top_k]
    best, best_score = top_idx[0], float(scores[top_idx[0]])

    alternatives = [{"english": english_sentences[i], "lunyoro": lunyoro_sentences[i],
                     "score": round(float(scores[i]), 3)} for i in top_idx[1:]]

    if best_score > 0.5:
        return {"translation": lunyoro_sentences[best], "method": "semantic_match",
                "confidence": round(best_score, 3),
                "matched_english": english_sentences[best], "alternatives": alternatives}

    return _dict_fallback(text, best_score, english_sentences[best], alternatives, "en→lun")


def translate_to_english(text: str, top_k: int = 3, context: str = "") -> dict:
    """Lunyoro/Rutooro → English — uses both MarianMT and NLLB if available."""
    text = _normalise(text.strip())

    marian = _mt_translate(text, "lun2en", context=context)
    nllb   = _nllb_translate(text, "lun2en", context=context)

    if marian or nllb:
        return {
            "translation":        marian or nllb,
            "translation_nllb":   nllb,
            "translation_marian": marian,
            "method":             "neural_mt",
            "confidence":         1.0,
            "alternatives":       [],
        }

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
            lunyoro_sentences, show_progress_bar=False, batch_size=64, convert_to_numpy=True
        )

    q_emb   = _sem_model.encode(text, convert_to_numpy=True)
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
    words      = re.findall(r"[a-zA-Z']+", text.lower())
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
    words      = re.findall(r"[a-zA-Z']+", text.lower())
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


def _infer_pos(word: str) -> str | None:
    """
    Heuristically infer the likely POS of a Lunyoro/Rutooro word from its prefix.
    Based on Bantu noun class and verb prefix patterns in the corpus.
    """
    w = word.lower().strip()
    if w.startswith(("oku", "okw", "ok-")):
        return "V"
    noun_prefixes = ("om", "ab", "ob", "eb", "ek", "ak", "en", "em", "in", "im",
                     "oru", "ama", "obu", "otu", "oku", "eri", "aga", "ege")
    if any(w.startswith(p) for p in noun_prefixes):
        return "N"
    if w.startswith(("nk", "ng", "mbi", "ndi", "nge")):
        return "ADJ"
    return None


def lookup_word(word: str, direction: str = "en→lun") -> list:
    """
    Dictionary lookup powered by MarianMT + semantic search + POS context.
    """
    _load_retrieval()
    word = _normalise(word.strip())
    word_lower = word.lower()
    results = []
    seen_words: set[str] = set()

    mt_direction = "en2lun" if direction == "en→lun" else "lun2en"
    mt_translation = _mt_translate(word, mt_direction)

    lunyoro_side = mt_translation if direction == "en→lun" else word
    inferred_pos = _infer_pos(lunyoro_side) if lunyoro_side else None

    en_side = word if direction == "en→lun" else (mt_translation or "")
    if not inferred_pos and en_side:
        en_lower = en_side.lower()
        if en_lower.startswith("to ") or en_lower.endswith(("ing", "ed", "ify", "ize", "ise")):
            inferred_pos = "V"
        elif en_lower.endswith(("ness", "tion", "ment", "ity", "er", "or", "ist")):
            inferred_pos = "N"
        elif en_lower.endswith(("ful", "less", "ous", "ive", "al", "ic", "able", "ible")):
            inferred_pos = "ADJ"

    q_emb = _sem_model.encode(word, convert_to_numpy=True)

    if direction == "en→lun":
        scores = util.cos_sim(q_emb, _index["embeddings"])[0].numpy()
    else:
        if "lunyoro_embeddings" not in _index:
            _index["lunyoro_embeddings"] = _sem_model.encode(
                _index["lunyoro_sentences"], show_progress_bar=False, batch_size=64, convert_to_numpy=True
            )
        scores = util.cos_sim(q_emb, _index["lunyoro_embeddings"])[0].numpy()

    top_idx = np.argsort(scores)[::-1][:5]
    corpus_hits = []
    for i in top_idx:
        score = float(scores[i])
        if score < 0.35:
            break
        corpus_hits.append({
            "word": _index["lunyoro_sentences"][i] if direction == "en→lun" else _index["english_sentences"][i],
            "definitionEnglish": _index["english_sentences"][i] if direction == "en→lun" else _index["lunyoro_sentences"][i],
            "definitionNative": "",
            "exampleSentence1": _index["lunyoro_sentences"][i],
            "exampleSentence1English": _index["english_sentences"][i],
            "dialect": "",
            "pos": "",
            "source": "corpus",
            "confidence": round(score, 3),
        })

    if direction == "en→lun":
        dict_search_keys = [
            (d["word"], (d.get("definitionEnglish") or "").lower())
            for d in _dictionary
        ]
        exact_matches = [
            (d, 100) for d in _dictionary
            if word_lower == (d.get("definitionEnglish") or "").lower().strip()
            or word_lower in (d.get("definitionEnglish") or "").lower().split()
        ]
        fuzzy_def_matches = process.extract(
            word_lower,
            [key for _, key in dict_search_keys],
            scorer=fuzz.token_sort_ratio,
            limit=10,
            score_cutoff=65,
        )
        fuzzy_matches = []
        for match in fuzzy_def_matches:
            entry = _index["_dict_def_map"].get(match[0])
            if entry:
                fuzzy_matches.append((entry, match[1]))
        for d, score in exact_matches:
            if d not in [e for e, _ in fuzzy_matches]:
                fuzzy_matches.insert(0, (d, score))
    else:
        dict_words = [d["word"] for d in _dictionary]
        exact_lun = [
            (d, 100) for d in _dictionary
            if word_lower == d["word"].lower()
            or word_lower in d["word"].lower()
            or d["word"].lower() in word_lower
        ]
        raw_matches = process.extract(
            word_lower,
            [w.lower() for w in dict_words],
            scorer=fuzz.token_sort_ratio,
            limit=10,
            score_cutoff=65,
        )
        fuzzy_matches = list(exact_lun)
        seen_fm = {d["word"] for d, _ in exact_lun}
        for match in raw_matches:
            entry = _dict_word_map.get(match[0])
            if entry and entry["word"] not in seen_fm:
                seen_fm.add(entry["word"])
                fuzzy_matches.append((entry, match[1]))
        if mt_translation:
            mt_def_matches = process.extract(
                mt_translation.lower(),
                [(d.get("definitionEnglish") or "").lower() for d in _dictionary],
                scorer=fuzz.token_sort_ratio,
                limit=5,
                score_cutoff=65,
            )
            for match in mt_def_matches:
                entry = _index["_dict_def_map"].get(match[0])
                if entry and entry not in [e for e, _ in fuzzy_matches]:
                    fuzzy_matches.append((entry, match[1]))

    dict_results = []
    for entry, score in fuzzy_matches:
        if entry["word"] in seen_words:
            continue
        seen_words.add(entry["word"])
        base_score = score / 100.0
        entry_pos = (entry.get("pos") or "").strip().upper()
        pos_boost = 0.0
        if inferred_pos and entry_pos:
            if entry_pos == inferred_pos:
                pos_boost = 0.15
            elif (inferred_pos == "N" and entry_pos == "ADJ") or \
                 (inferred_pos == "ADJ" and entry_pos == "N"):
                pos_boost = 0.05
        dict_results.append({
            **entry,
            "source": "dictionary",
            "confidence": round(min(base_score + pos_boost, 1.0), 3),
            "pos_matched": entry_pos == inferred_pos if inferred_pos and entry_pos else False,
        })

    dict_results.sort(key=lambda x: (-int(x.get("pos_matched", False)), -x["confidence"]))

    for entry in _dictionary:
        w = entry["word"]
        if w in seen_words:
            continue
        def_en = (entry.get("definitionEnglish") or "").lower()
        lun_word = w.lower()
        match = (
            (direction == "en→lun" and (def_en == word_lower or word_lower in def_en.split()))
            or (direction == "lun→en" and (lun_word == word_lower or word_lower in lun_word))
        )
        if match:
            seen_words.add(w)
            entry_pos = (entry.get("pos") or "").strip().upper()
            dict_results.append({
                **entry,
                "source": "dictionary",
                "confidence": 1.0,
                "pos_matched": entry_pos == inferred_pos if inferred_pos else False,
            })

    if mt_translation:
        mt_lower = mt_translation.lower()
        mt_dict_entry = _dict_word_map.get(mt_lower)
        results.append({
            "word": mt_translation if direction == "en→lun" else word,
            "definitionEnglish": word if direction == "en→lun" else mt_translation,
            "definitionNative": mt_dict_entry.get("definitionNative", "") if mt_dict_entry else "",
            "exampleSentence1": mt_dict_entry.get("exampleSentence1", "") if mt_dict_entry else "",
            "exampleSentence1English": mt_dict_entry.get("exampleSentence1English", "") if mt_dict_entry else "",
            "dialect": mt_dict_entry.get("dialect", "") if mt_dict_entry else "",
            "pos": mt_dict_entry.get("pos", inferred_pos or "") if mt_dict_entry else (inferred_pos or ""),
            "source": "neural_mt",
            "confidence": 1.0,
            "pos_matched": True if inferred_pos else False,
        })

    results.extend(dict_results[:5])
    results.extend(corpus_hits[:3])

    results.sort(key=lambda x: -x.get("confidence", 0))
    return results[:8]


def get_index_and_model():
    _load_retrieval()
    return _index, _sem_model


def _build_corpus_vocab() -> set:
    """Build vocabulary of known Lunyoro/Rutooro words from corpus, tokenizer, and dictionary."""
    _load_retrieval()
    known: set[str] = set()

    for sent in _index["lunyoro_sentences"]:
        for w in re.findall(r"[a-zA-Z']+", sent):
            if len(w) >= 2:
                known.add(w.lower())

    lun2en_path = os.path.join(MODEL_DIR, "lun2en")
    if os.path.isdir(lun2en_path):
        try:
            from transformers import MarianTokenizer
            tok = MarianTokenizer.from_pretrained(lun2en_path)
            for token in tok.get_vocab().keys():
                clean = token.lstrip("▁").lower()
                if clean.isalpha() and len(clean) >= 2:
                    known.add(clean)
        except Exception:
            pass

    for d in _dictionary:
        if d.get("word"):
            known.add(d["word"].lower())

    return known


_corpus_vocab: set | None = None


def spellcheck(text: str) -> list:
    global _corpus_vocab
    _load_retrieval()

    text = _normalise(text)

    if _corpus_vocab is None:
        _corpus_vocab = _build_corpus_vocab()

    vocab_list = list(_corpus_vocab)
    tokens = re.findall(r"[a-zA-Z']+", text)
    misspelled = []

    for token in tokens:
        lower = token.lower()
        if len(lower) < 3 or lower in _corpus_vocab:
            continue

        lun2en_path = os.path.join(MODEL_DIR, "lun2en")
        model_knows = False
        if os.path.isdir(lun2en_path) and _load_mt("lun2en"):
            try:
                tokenizer, _, _ = _mt_models["lun2en"]
                pieces = tokenizer.tokenize(lower)
                if pieces and "<unk>" not in pieces and len(pieces) == 1:
                    model_knows = True
            except Exception:
                pass

        if model_knows:
            continue

        prefix = lower[:3]
        prefix_words = [w for w in vocab_list if w.startswith(prefix)]
        candidate_pool = prefix_words if len(prefix_words) >= 10 else vocab_list

        suggestions = process.extract(
            lower, candidate_pool,
            scorer=fuzz.ratio,
            limit=5,
            score_cutoff=55,
        )
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
