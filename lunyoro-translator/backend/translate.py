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
_nllb_models   = {}   # {"en2lun": (tokenizer, model, device), "lun2en": ...}
_nllb_available = {}  # {"en2lun": bool, "lun2en": bool}
_nllb_whitelist: list | None = None  # token ID whitelist loaded once

NLLB_LANG_EN  = "eng_Latn"
NLLB_LANG_LUN = "run_Latn"  # Rundi — closest supported code to Lunyoro/Rutooro


def _load_nllb_whitelist() -> list | None:
    """Load the Lunyoro token whitelist, build it if missing."""
    global _nllb_whitelist
    if _nllb_whitelist is not None:
        return _nllb_whitelist
    whitelist_path = os.path.join(MODEL_DIR, "lunyoro_token_whitelist.json")
    if os.path.exists(whitelist_path):
        import json
        with open(whitelist_path) as f:
            _nllb_whitelist = json.load(f)
        print(f"[translate] Loaded token whitelist: {len(_nllb_whitelist):,} allowed tokens")
    else:
        print("[translate] Token whitelist not found — run build_lunyoro_vocab.py to generate it")
        _nllb_whitelist = []
    return _nllb_whitelist


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
    """Lazy-load a fine-tuned MarianMT model. Auto-downloads from HuggingFace if missing."""
    if direction in _mt_available:
        return _mt_available[direction]

    path = os.path.join(MODEL_DIR, direction)

    # Auto-download from HuggingFace if not present locally
    if not os.path.isdir(path) or not any(
        f.endswith((".safetensors", ".bin")) for f in os.listdir(path) if os.path.isdir(path)
    ):
        hf_repos = {
            "en2lun": "keithtwesigye/lunyoro-en2lun",
            "lun2en": "keithtwesigye/lunyoro-lun2en",
        }
        repo_id = hf_repos.get(direction)
        if repo_id:
            try:
                print(f"[translate] Downloading {repo_id} from HuggingFace...")
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=path,
                    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
                )
                print(f"[translate] Downloaded {direction} model.")
            except Exception as e:
                print(f"[translate] Could not download {direction} from HuggingFace: {e}")
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
    """Run inference with a fine-tuned MarianMT model."""
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
            no_repeat_ngram_size=3,
            repetition_penalty=1.3,
            length_penalty=1.0,
        )
    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return result


def _load_nllb(direction: str) -> bool:
    """Lazy-load a fine-tuned NLLB model. Auto-downloads from HuggingFace if missing."""
    if direction in _nllb_available:
        return _nllb_available[direction]

    path = os.path.join(MODEL_DIR, f"nllb_{direction}")

    # Auto-download from HuggingFace if not present locally
    if not os.path.isdir(path) or not any(
        f.endswith((".safetensors", ".bin")) for f in os.listdir(path) if os.path.isdir(path)
    ):
        hf_repos = {
            "en2lun": "keithtwesigye/lunyoro-nllb_en2lun",
            "lun2en": "keithtwesigye/lunyoro-nllb_lun2en",
        }
        repo_id = hf_repos.get(direction)
        if repo_id:
            try:
                print(f"[translate] Downloading {repo_id} from HuggingFace...")
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=path,
                    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
                )
                print(f"[translate] Downloaded nllb_{direction} model.")
            except Exception as e:
                print(f"[translate] Could not download nllb_{direction} from HuggingFace: {e}")
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
    src_lang = NLLB_LANG_EN if direction == "en2lun" else NLLB_LANG_LUN
    tgt_lang = NLLB_LANG_LUN if direction == "en2lun" else NLLB_LANG_EN
    tokenizer.src_lang = src_lang
    input_text = f"{context} ||| {text}" if context else text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256).to(device)

    generate_kwargs: dict = dict(
        num_beams=4,
        max_length=512,
        early_stopping=True,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
        length_penalty=1.0,
    )
    # Force the target language token so NLLB generates in the correct language
    generate_kwargs["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids(tgt_lang)

    # For en2lun: suppress non-Lunyoro tokens to prevent Swahili/Kinyarwanda bleed
    if direction == "en2lun":
        whitelist = _load_nllb_whitelist()
        if whitelist:
            vocab_size = getattr(getattr(model, "module", model).config, "vocab_size", 256204)
            allowed_set = set(whitelist)
            suppress = [i for i in range(vocab_size) if i not in allowed_set]
            if suppress:
                generate_kwargs["suppress_tokens"] = suppress

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generate_kwargs)
    nllb_result = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Discard output that looks like raw dictionary notation rather than a real translation
    if _is_notation_garbage(nllb_result):
        return None

    return nllb_result


def _is_notation_garbage(text: str) -> bool:
    """Return True if the text looks like raw dictionary notation, not a real translation."""
    if not text:
        return True
    import re
    t = text.strip()
    # Patterns that indicate dictionary notation artifacts
    notation_patterns = [
        r'\bn\.\s*(cl\.|v\.|adj\.)',          # "n. cl.", "n. v."
        r'\(pl\.\s*(nil|same|\w+)\)',          # "(pl. nil)", "(pl. same)"
        r',\s*o-\s*,',                         # ", o-,"  (noun class marker)
        r'\bcl\.\s*\d+',                       # "cl. 11"
        r'^\s*[a-z]{1,3}\.\s*\(',             # starts with "n. (" or "v. ("
        r'\(pl\.\s*\w*\)\s*$',                # ends with "(pl. X)"
    ]
    for pat in notation_patterns:
        if re.search(pat, t, re.IGNORECASE):
            return True
    # Also reject if >50% of tokens are abbreviations/punctuation with no real words
    real_words = re.findall(r'[a-zA-Z]{4,}', t)
    tokens = t.split()
    if tokens and len(real_words) / len(tokens) < 0.3:
        return True
    return False


# ── public API ───────────────────────────────────────────────────────────────

def translate(text: str, top_k: int = 3, context: str = "") -> dict:
    """English → Lunyoro/Rutooro — uses both MarianMT and NLLB if available."""
    text = _normalise(text.strip())

    marian = _mt_translate(text, "en2lun", context=context)
    nllb   = _nllb_translate(text, "en2lun", context=context)

    if marian or nllb:
        return {
            "translation":        marian or nllb,  # MarianMT is primary
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
            "translation":        marian or nllb,  # MarianMT is primary for lun→en
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
        # Check static web entries first
        from web_fallback import lookup_static
        static = lookup_static(word, "en→lun")
        if static:
            found.append({"english_word": word, "lunyoro_word": static, "definition": ""})
            continue
        match = process.extractOne(word, dict_words, scorer=fuzz.ratio, score_cutoff=80)
        if match:
            entry = next((d for d in _dictionary if d["word"] == match[0]), None)
            if entry:
                found.append({"english_word": word, "lunyoro_word": entry["word"],
                               "definition": entry.get("definitionNative", "")})

    # If still nothing found, try web fallback for the full phrase
    if not found:
        from web_fallback import web_search_fallback
        web_result = web_search_fallback(text, "en→lun")
        if web_result:
            return {"translation": web_result, "method": "web_fallback",
                    "confidence": 0.4, "alternatives": alternatives}

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
    Dictionary lookup: exact match → fuzzy dictionary → neural MT → corpus.
    """
    _load_retrieval()
    word = _normalise(word.strip())
    word_lower = word.lower()
    results: list = []
    seen_words: set[str] = set()

    def clean_mt(text: str | None) -> str | None:
        """Strip domain tags and notation garbage from MT output."""
        if not text:
            return None
        import re as _re
        t = _re.sub(r'^\[.*?\]\s*', '', text).strip()            # remove [GENERAL] etc.
        t = _re.sub(r',\s*[a-z]-\s*,.*$', '', t, flags=_re.I).strip()   # ", o-, n. cl..."
        t = _re.sub(r'\(pl\.\s*\w*\)', '', t).strip()            # (pl. nil)
        t = _re.sub(r'\bn\.\s*cl\.\s*\d+.*$', '', t, flags=_re.I).strip()
        t = _re.sub(r',\s*n\.\s*,.*$', '', t, flags=_re.I).strip()      # ", n., ekisisani"
        t = _re.sub(r',\s*v\.\s*,.*$', '', t, flags=_re.I).strip()      # ", v., ..."
        t = _re.sub(r',\s*adj\.\s*,.*$', '', t, flags=_re.I).strip()
        t = _re.sub(r'\s*,\s*ekisisani.*$', '', t, flags=_re.I).strip()  # ", ekisisani" suffix
        t = t.strip('.,; ')
        if not t or len(t) < 2:
            return None
        return t

    mt_direction = "en2lun" if direction == "en→lun" else "lun2en"
    raw_mt = _mt_translate(word, mt_direction)
    mt_translation = clean_mt(raw_mt)

    # ── 1. Exact dictionary match (highest priority) ──────────────────────
    if direction == "en→lun":
        exact = [d for d in _dictionary
                 if word_lower == (d.get("definitionEnglish") or "").lower().strip()
                 or word_lower in (d.get("definitionEnglish") or "").lower().split()]
    else:
        exact = [d for d in _dictionary
                 if word_lower == d["word"].lower()
                 or d["word"].lower() == word_lower]

    for d in exact:
        if d["word"] not in seen_words:
            seen_words.add(d["word"])
            results.append({**d, "source": "dictionary", "confidence": 1.0, "pos_matched": False})

    # ── 2. Fuzzy dictionary match ─────────────────────────────────────────
    if direction == "en→lun":
        fuzzy_raw = process.extract(
            word_lower,
            [(d.get("definitionEnglish") or "").lower() for d in _dictionary],
            scorer=fuzz.token_sort_ratio,
            limit=10,
            score_cutoff=70,
        )
        for match_text, score, _ in fuzzy_raw:
            entry = _index["_dict_def_map"].get(match_text)
            if entry and entry["word"] not in seen_words:
                seen_words.add(entry["word"])
                results.append({**entry, "source": "dictionary",
                                 "confidence": round(score / 100, 3), "pos_matched": False})
    else:
        dict_words_lower = [d["word"].lower() for d in _dictionary]
        fuzzy_raw = process.extract(
            word_lower,
            dict_words_lower,
            scorer=fuzz.ratio,       # stricter scorer for Lunyoro words
            limit=10,
            score_cutoff=80,         # higher threshold — Lunyoro words are similar-looking
        )
        for match_text, score, _ in fuzzy_raw:
            entry = _dict_word_map.get(match_text)
            if entry and entry["word"] not in seen_words:
                seen_words.add(entry["word"])
                results.append({**entry, "source": "dictionary",
                                 "confidence": round(score / 100, 3), "pos_matched": False})

    # ── 3. Neural MT result ───────────────────────────────────────────────
    if mt_translation and mt_translation.lower() not in seen_words:
        seen_words.add(mt_translation.lower())
        # Try to enrich with dictionary entry for the MT word
        mt_dict = _dict_word_map.get(mt_translation.lower())
        results.append({
            "word":                    mt_translation if direction == "en→lun" else word,
            "definitionEnglish":       word if direction == "en→lun" else mt_translation,
            "definitionNative":        mt_dict.get("definitionNative", "") if mt_dict else "",
            "exampleSentence1":        mt_dict.get("exampleSentence1", "") if mt_dict else "",
            "exampleSentence1English": mt_dict.get("exampleSentence1English", "") if mt_dict else "",
            "dialect":                 mt_dict.get("dialect", "") if mt_dict else "",
            "pos":                     mt_dict.get("pos", "") if mt_dict else "",
            "source":                  "neural_mt",
            "confidence":              0.95,
            "pos_matched":             False,
        })

    # ── 4. Corpus semantic search (only for multi-word queries) ──────────────
    # Single words get poor corpus matches — skip unless query is a phrase
    is_phrase = len(word.split()) > 1
    if is_phrase:
        q_emb = _sem_model.encode(word, convert_to_numpy=True)
        if direction == "en→lun":
            scores = util.cos_sim(q_emb, _index["embeddings"])[0].numpy()
        else:
            if "lunyoro_embeddings" not in _index:
                _index["lunyoro_embeddings"] = _sem_model.encode(
                    _index["lunyoro_sentences"], show_progress_bar=False,
                    batch_size=64, convert_to_numpy=True
                )
            scores = util.cos_sim(q_emb, _index["lunyoro_embeddings"])[0].numpy()

        top_idx = np.argsort(scores)[::-1][:5]
        for i in top_idx:
            score = float(scores[i])
            if score < 0.45:
                break
            lun = _index["lunyoro_sentences"][i]
            en  = _index["english_sentences"][i]
            if _is_notation_garbage(lun) or _is_notation_garbage(en):
                continue
            display_word = lun if direction == "en→lun" else en
            if display_word not in seen_words:
                seen_words.add(display_word)
                results.append({
                    "word":                    display_word,
                    "definitionEnglish":       en if direction == "en→lun" else lun,
                    "definitionNative":        "",
                    "exampleSentence1":        lun,
                    "exampleSentence1English": en,
                    "dialect": "", "pos": "",
                    "source":     "corpus",
                    "confidence": round(score, 3),
                    "pos_matched": False,
                })

    # Sort: exact dict first, then by confidence
    results.sort(key=lambda x: (
        0 if (x["source"] == "dictionary" and x["confidence"] == 1.0) else
        1 if x["source"] == "dictionary" else
        2 if x["source"] == "neural_mt" else 3,
        -x.get("confidence", 0)
    ))
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
        # Add interjections as known valid words
        from language_rules import INTERJECTIONS, IDIOMS
        for word in INTERJECTIONS:
            for w in word.split():
                if len(w) >= 2:
                    _corpus_vocab.add(w.lower())
        for phrase in IDIOMS:
            for w in phrase.split():
                if len(w) >= 2:
                    _corpus_vocab.add(w.lower())

    vocab_list = list(_corpus_vocab)
    tokens = re.findall(r"[a-zA-Z']+", text)
    misspelled = []

    for token in tokens:
        lower = token.lower()

        # Skip English-looking words (lun→en input may contain proper nouns, code-switching)
        if lower in {"the", "a", "an", "is", "are", "was", "were", "and", "or", "of", "in", "to"}:
            continue

        # Valid Bantu morphological prefixes — never flag these as misspelled
        _BANTU_PREFIXES = (
            "oku", "okw", "omu", "aba", "obu", "otu", "ama", "eri",
            "ebi", "eki", "aka", "aga", "oru", "en", "em", "in", "im",
            "ni", "ba", "ka", "ku", "mu", "bu", "tu", "bi", "ki", "ga",
        )
        if any(lower.startswith(p) for p in _BANTU_PREFIXES) and len(lower) >= 4:
            continue

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
            score_cutoff=75,
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
