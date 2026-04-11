from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import json
import os
import io

# Ensure all HuggingFace/transformers calls stay fully offline
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from translate import translate, translate_to_english, lookup_word, spellcheck, get_index_and_model

app = FastAPI(title="Lunyoro/Rutooro Translator API")


@app.on_event("startup")
def preload_model():
    """Load retrieval index and all neural MT models at startup."""
    get_index_and_model()
    from translate import _load_mt, _load_nllb
    _load_mt("en2lun")
    _load_mt("lun2en")
    _load_nllb("en2lun")
    _load_nllb("lun2en")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "history.json")


def save_history(entry: dict):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    history.insert(0, entry)
    history = history[:500]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


class TranslateRequest(BaseModel):
    text: str
    context: str = ""  # optional previous sentence for context-aware translation


class WordLookupRequest(BaseModel):
    word: str
    direction: str = "en→lun"

class SpellCheckRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "Lunyoro/Rutooro Translator API is running"}


@app.post("/translate")
def translate_text(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(req.text) > 1000:
        raise HTTPException(status_code=400, detail="Text too long (max 1000 chars)")
    result = translate(req.text, context=req.context)
    save_history({
        "input": req.text,
        "direction": "en→lun",
        "translation": result.get("translation"),
        "method": result.get("method"),
        "confidence": result.get("confidence"),
        "timestamp": datetime.utcnow().isoformat(),
    })
    return result


@app.post("/translate-reverse")
def translate_reverse(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(req.text) > 1000:
        raise HTTPException(status_code=400, detail="Text too long (max 1000 chars)")
    result = translate_to_english(req.text, context=req.context)
    save_history({
        "input": req.text,
        "direction": "lun→en",
        "translation": result.get("translation"),
        "method": result.get("method"),
        "confidence": result.get("confidence"),
        "timestamp": datetime.utcnow().isoformat(),
    })
    return result


@app.post("/lookup")
def word_lookup(req: WordLookupRequest):
    if not req.word.strip():
        raise HTTPException(status_code=400, detail="Word cannot be empty")
    results = lookup_word(req.word, req.direction)
    return {"word": req.word, "results": results}


@app.post("/spellcheck")
def spellcheck_text(req: SpellCheckRequest):
    if not req.text.strip():
        return {"misspelled": []}
    results = spellcheck(req.text)
    return {"misspelled": results}


@app.get("/history")
def get_history():
    if not os.path.exists(HISTORY_FILE):
        return {"history": []}
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
    return {"history": history}


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}


def extract_text_from_file(filename: str, contents: bytes) -> str:
    """Extract plain text from PDF, DOCX, DOC, or TXT files."""
    import re
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(contents))
        text = " ".join(page.extract_text() or "" for page in reader.pages)

    elif ext in (".docx", ".doc"):
        from docx import Document
        doc = Document(io.BytesIO(contents))
        text = " ".join(p.text for p in doc.paragraphs if p.text.strip())

    elif ext == ".txt":
        text = contents.decode("utf-8", errors="ignore")

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return re.sub(r'\s+', ' ', text).strip()


def validate_upload(filename: str):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )


@app.post("/summarize-pdf")
async def summarize_pdf(file: UploadFile = File(...)):
    """Upload a PDF, DOCX, DOC, or TXT and get an English summary."""
    validate_upload(file.filename)

    import re
    from translate import _mt_translate, _load_retrieval, _dictionary

    contents = await file.read()
    try:
        full_text = extract_text_from_file(file.filename, contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not full_text:
        raise HTTPException(status_code=400, detail="No text found in document")

    # Split into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', full_text) if len(s.strip()) > 10]
    total_sentences = len(sentences)

    # Detect language — if majority of words match Lunyoro dictionary, translate first
    _load_retrieval()
    known_lunyoro = set(d["word"].lower() for d in _dictionary if d.get("word"))
    sample_words = " ".join(sentences[:20]).lower().split()
    lunyoro_hits = sum(1 for w in sample_words if w in known_lunyoro)
    is_lunyoro = lunyoro_hits / max(len(sample_words), 1) > 0.1

    # Translate Lunyoro → English if needed
    if is_lunyoro:
        english_sentences = []
        for sent in sentences:
            translated = _mt_translate(sent, "lun2en") or sent
            english_sentences.append(translated)
    else:
        english_sentences = sentences

    # Extractive summarization — score sentences by position + keyword frequency
    from collections import Counter
    all_words = " ".join(english_sentences).lower().split()
    stopwords = {"the","a","an","and","or","but","in","on","at","to","for","of","with","is","was","are","were","be","been","it","this","that","as","by","from","have","has","had","not","he","she","they","we","i","you","his","her","their","its","my","our","your"}
    word_freq = Counter(w for w in all_words if w not in stopwords and len(w) > 3)

    def score_sentence(sent: str, idx: int, total: int) -> float:
        words = sent.lower().split()
        freq_score = sum(word_freq.get(w, 0) for w in words) / max(len(words), 1)
        # Boost first and last sentences
        position_score = 1.5 if idx < total * 0.15 else (1.2 if idx > total * 0.85 else 1.0)
        return freq_score * position_score

    scored = [(score_sentence(s, i, len(english_sentences)), s)
              for i, s in enumerate(english_sentences)]
    scored.sort(key=lambda x: -x[0])

    # Pick top sentences — roughly 20% of document or max 10
    top_n = max(3, min(10, len(english_sentences) // 5))
    top_sentences = [s for _, s in scored[:top_n]]

    # Re-order by original position for coherent reading
    order = {s: i for i, s in enumerate(english_sentences)}
    top_sentences.sort(key=lambda s: order.get(s, 0))

    summary = " ".join(top_sentences)

    # Translate the English summary to Lunyoro using both models
    from translate import _mt_translate, _nllb_translate
    summary_lunyoro_marian = _mt_translate(summary, "en2lun") or ""
    summary_lunyoro_nllb   = _nllb_translate(summary, "en2lun") or ""
    # Primary = NLLB-200, fallback to MarianMT
    summary_lunyoro = summary_lunyoro_nllb or summary_lunyoro_marian

    save_history({
        "input": f"[DOC Summary] {file.filename}",
        "direction": "en→lun",
        "translation": summary_lunyoro[:200] + "..." if len(summary_lunyoro) > 200 else summary_lunyoro,
        "method": "extractive_summary",
        "confidence": None,
        "timestamp": datetime.utcnow().isoformat(),
    })

    return {
        "filename": file.filename,
        "total_pages": full_text.count("\f") + 1 if file.filename.lower().endswith(".pdf") else 1,
        "total_sentences": total_sentences,
        "language_detected": "lunyoro" if is_lunyoro else "english",
        "summary": summary,
        "summary_lunyoro": summary_lunyoro,
        "summary_lunyoro_marian": summary_lunyoro_marian,
        "summary_lunyoro_nllb": summary_lunyoro_nllb,
        "sentences_used": top_n,
    }


class ChatRequest(BaseModel):
    message: str
    history: list = []
    sector: str | None = None
    conversation_mode: bool = False


@app.post("/chat")
def chat(req: ChatRequest):
    """AI Language Assistant — answers questions about Runyoro-Rutooro using the translation models."""
    from translate import _mt_translate, _nllb_translate, _load_retrieval, _normalise
    _load_retrieval()
    from translate import _dictionary, _index

    msg = _normalise(req.message.strip())
    msg_lower = msg.lower()
    sector = (req.sector or "").upper()

    # ── Sector-specific vocabulary response ─────────────────────────────────
    SECTOR_LABELS = {
        "CUL": "Culture & Traditions", "ART": "Arts & Music",
        "AGR": "Agriculture",          "ENV": "Environment & Nature",
        "EDU": "Education",            "SPR": "Spirituality",
        "DLY": "Daily Life",           "NAR": "Storytelling",
        "ECO": "Economy & Trade",      "GOV": "Governance",
        "HIS": "History",              "HLT": "Health",
        "POL": "Politics",             "ALL": "All Sectors",
    }
    if sector and sector in SECTOR_LABELS:
        if sector == "ALL":
            sector_entries = [
                d for d in _dictionary
                if d.get("word") and d.get("definitionEnglish")
            ][:10]
        else:
            sector_entries = [
                d for d in _dictionary
                if (d.get("domain") or "").upper() == sector and d.get("word") and d.get("definitionEnglish")
            ][:8]
        label = SECTOR_LABELS[sector]
        if sector_entries:
            reply = f"Here are key **{label}** terms in Runyoro-Rutooro:\n\n"
            for e in sector_entries:
                word = e["word"]
                defn = e.get("definitionEnglish", "")
                ex   = e.get("exampleSentence1", "")
                ex_en = e.get("exampleSentence1English", "")
                reply += f"• **{word}** — {defn}"
                if ex:
                    reply += f"\n  *{ex}*"
                    if ex_en:
                        reply += f" ({ex_en})"
                reply += "\n"
            # Also translate the user's message with sector context
            context = f"[{sector}]"
            marian = _mt_translate(f"{context} {msg}", "en2lun") or _mt_translate(msg, "en2lun") or ""
            nllb   = _nllb_translate(f"{context} {msg}", "en2lun") or _nllb_translate(msg, "en2lun") or ""
            if marian or nllb:
                reply += f"\nTranslation of your query: **{marian or nllb}**"
            return {"reply": reply}
        else:
            reply = f"I don't have specific **{label}** entries yet, but here's a translation:\n\n"
            marian = _mt_translate(msg, "en2lun") or ""
            nllb   = _nllb_translate(msg, "en2lun") or ""
            reply += f"**{marian or nllb or 'No translation available'}**"
            return {"reply": reply}

    # ── Conversation mode — generative replies in Runyoro/Rutooro ────────────
    if req.conversation_mode:
        import random

        # Use NLLB first for lun→en — it's more accurate than MarianMT for Lunyoro
        msg_en = _nllb_translate(msg, "lun2en") or _mt_translate(msg, "lun2en") or msg
        msg_en_lower = msg_en.lower()

        # Step 2: Generate a thoughtful English response based on the question
        # Build context from conversation history
        history_context = ""
        if req.history:
            recent = req.history[-4:]  # last 4 turns
            for turn in recent:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role == "user":
                    en = _mt_translate(content, "lun2en") or content
                    history_context += f"User: {en}\n"
                else:
                    history_context += f"Assistant: {content}\n"

        # Generate a contextual English reply
        if any(w in msg_en_lower for w in ["who are you", "what are you", "your name"]):
            reply_en = "I am an AI language assistant for Runyoro-Rutooro. I can help you learn the language, answer questions, and have conversations."

        elif any(w in msg_en_lower for w in ["hello", "hi", "good morning", "good afternoon", "good evening", "how are you", "greetings", "how do you do", "how are things", "how is it", "how is life", "fine", "well", "doing"]):
            replies = [
                "I am fine, thank you for asking! How are you doing today?",
                "I am doing very well! It is great to hear from you. How can I help?",
                "I am well, thank you! I am happy to speak with you today.",
            ]
            reply_en = random.choice(replies)

        elif any(w in msg_en_lower for w in ["thank", "grateful", "appreciate"]):
            replies = [
                "You are very welcome! It is my pleasure to help you.",
                "I am glad I could be of help. Feel free to ask anything else.",
                "Thank you for the kind words! I am always here for you.",
            ]
            reply_en = random.choice(replies)

        elif any(w in msg_en_lower for w in ["bye", "goodbye", "see you", "farewell", "leaving"]):
            replies = [
                "Goodbye! It was wonderful speaking with you. Come back anytime.",
                "Safe travels! I will be here whenever you need me.",
                "Until next time! Take care of yourself.",
            ]
            reply_en = random.choice(replies)

        elif any(w in msg_en_lower for w in ["what", "how", "why", "when", "where", "who", "which", "explain", "tell me"]):
            # Question — generate a relevant answer using corpus knowledge
            from sentence_transformers import util as st_util
            import numpy as np
            q_emb = _sem_model.encode(msg_en, convert_to_numpy=True)
            scores = st_util.cos_sim(q_emb, _index["embeddings"])[0].numpy()
            top_idx = np.argsort(scores)[::-1][:3]
            top_en = [_index["english_sentences"][i] for i in top_idx if float(scores[i]) > 0.3]
            if top_en:
                reply_en = f"That is a great question. {top_en[0]} I hope that helps answer what you were asking."
            else:
                reply_en = f"That is an interesting question about '{msg_en}'. In Runyoro-Rutooro culture and language, this is an important topic worth exploring further."

        elif any(w in msg_en_lower for w in ["yes", "no", "agree", "disagree", "correct", "wrong", "true", "false"]):
            replies = [
                "I understand your point. Could you tell me more about what you think?",
                "That is a valid perspective. I would love to hear more of your thoughts.",
                "Interesting! Let us explore this topic further together.",
            ]
            reply_en = random.choice(replies)

        else:
            # General conversational reply
            replies = [
                f"I understand you are talking about '{msg_en}'. That is a fascinating topic. Tell me more about what you mean.",
                f"Thank you for sharing that. Regarding '{msg_en}', I think this is something worth discussing in depth.",
                f"I hear you. '{msg_en}' is indeed an important subject. What specifically would you like to know?",
                "That is very interesting! Please continue, I am listening carefully.",
                "I appreciate you sharing that with me. Could you elaborate a little more?",
            ]
            reply_en = random.choice(replies)

        # Step 3: Translate to Lunyoro — MarianMT primary (authentic Lunyoro), NLLB fallback
        reply_lun = _mt_translate(reply_en, "en2lun") or _nllb_translate(reply_en, "en2lun") or reply_en

        return {"reply": reply_lun}

    # ── Detect intent ────────────────────────────────────────────────────────

    # 1. Translation request: "how do I say X in Runyoro/Rutooro"
    import re
    say_match = re.search(
        r"how (?:do i|to) say ['\"]?(.+?)['\"]? in (?:runyoro|rutooro|lunyoro)",
        msg_lower
    )
    if say_match:
        phrase = say_match.group(1).strip()
        marian = _mt_translate(phrase, "en2lun") or ""
        nllb   = _nllb_translate(phrase, "en2lun") or ""
        primary = marian or nllb
        reply = f'"{phrase}" in Runyoro-Rutooro is: **{primary}**'
        if marian and nllb and marian.lower() != nllb.lower():
            reply += f'\n\nMarianMT says: {marian}\nNLLB-200 says: {nllb}'
        return {"reply": reply}

    # 2. Dictionary lookup: "what does X mean" / "define X"
    define_match = re.search(
        r"(?:what does|what is|define|meaning of) ['\"]?(\w[\w\s']+?)['\"]?(?: mean| in (?:runyoro|rutooro|english))?[?]?$",
        msg_lower
    )
    if define_match:
        word = define_match.group(1).strip()
        from translate import lookup_word
        results = lookup_word(word, "en→lun")
        if results:
            r = results[0]
            reply = f'**{r.get("word", word)}** — {r.get("definitionEnglish", "")}'
            if r.get("definitionNative"):
                reply += f'\nNative: {r["definitionNative"]}'
            if r.get("exampleSentence1"):
                reply += f'\nExample: {r["exampleSentence1"]}'
                if r.get("exampleSentence1English"):
                    reply += f' ({r["exampleSentence1English"]})'
        else:
            reply = f'Sorry, I couldn\'t find a definition for "{word}" in the dictionary.'
        return {"reply": reply}

    # 3. Grammar / explain question
    if any(w in msg_lower for w in ["explain", "difference between", "grammar", "prefix", "verb", "noun", "how do you use"]):
        # Use NLLB to translate the question to get context, then give a structured answer
        reply = (
            "Runyoro-Rutooro is a Bantu language with rich morphology. "
            "Verbs typically start with **oku-** (infinitive prefix), e.g. *okugenda* (to go). "
            "Nouns follow Bantu noun classes — *om-/ab-* for people, *en-/em-* for animals/things, *ama-* for plurals.\n\n"
        )
        # Try to translate the specific terms mentioned
        words = re.findall(r"'([^']+)'|\"([^\"]+)\"", msg)
        for w_tuple in words[:2]:
            w = (w_tuple[0] or w_tuple[1]).strip()
            if w:
                t = _mt_translate(w, "en2lun") or _nllb_translate(w, "en2lun") or ""
                if t:
                    reply += f'**{w}** → {t}\n'
        if not words:
            reply += "Could you give me a specific word or phrase to explain?"
        return {"reply": reply}

    # 4. Culture question
    if any(w in msg_lower for w in ["culture", "tradition", "empaako", "bunyoro", "tooro", "kingdom", "custom", "ceremony"]):
        entries = [d for d in _dictionary if any(
            w in (d.get("domain") or "").upper() for w in ["CUL", "ART", "SPR"]
        )][:3]
        reply = (
            "Runyoro-Rutooro culture is rich with traditions. "
            "The **Empaako** naming system gives each person a praise name shared across the community. "
            "Traditional music includes instruments like the *enanga* (harp) and *engoma* (drum).\n\n"
        )
        if entries:
            reply += "Some cultural terms:\n"
            for e in entries:
                reply += f'• **{e["word"]}** — {e.get("definitionEnglish", "")}\n'
        return {"reply": reply}

    # 5. Story request
    if any(w in msg_lower for w in ["story", "tell me", "short story", "tale"]):
        # Translate a simple story sentence by sentence
        story_en = [
            "Once upon a time, there was a wise king in Bunyoro.",
            "He ruled his people with kindness and justice.",
            "The people loved him and called him by his Empaako name.",
            "And they lived in peace forever."
        ]
        story_lun = [_mt_translate(s, "en2lun") or s for s in story_en]
        reply = "Here is a short story in Rutooro:\n\n"
        for en, lun in zip(story_en, story_lun):
            reply += f"{lun}\n*({en})*\n\n"
        return {"reply": reply}

    # 6. General translation fallback — translate the message itself
    marian = _mt_translate(msg, "en2lun") or ""
    nllb   = _nllb_translate(msg, "en2lun") or ""
    primary = marian or nllb

    if primary:
        reply = f'Translation of your message to Runyoro-Rutooro:\n\n**{primary}**'
        if marian and nllb and marian.lower() != nllb.lower():
            reply += f'\n\nMarianMT: {marian}\nNLLB-200: {nllb}'
    else:
        reply = (
            "I'm your Runyoro-Rutooro language assistant. You can ask me to:\n"
            "• Translate phrases: *\"How do I say 'good morning' in Runyoro?\"*\n"
            "• Look up words: *\"What does 'omukama' mean?\"*\n"
            "• Explain grammar or culture\n"
            "• Tell a short story in Rutooro"
        )

    return {"reply": reply}
