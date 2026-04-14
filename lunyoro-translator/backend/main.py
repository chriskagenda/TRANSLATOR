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
    allow_origins=["*"],
    allow_credentials=False,
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
    # Primary = MarianMT (fine-tuned on Runyoro-Rutooro), fallback to NLLB
    summary_lunyoro = summary_lunyoro_marian or summary_lunyoro_nllb

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
    """AI Language Assistant — LLM-powered generative replies about Runyoro-Rutooro."""
    import re, requests as _requests
    from translate import _mt_translate, _load_retrieval, _normalise, _index, _sem_model
    from language_rules import get_grammar_context, EMPAAKO, PROVERBS, NUMBERS
    import numpy as np
    from sentence_transformers import util as st_util

    _load_retrieval()
    from translate import _dictionary

    msg    = _normalise(req.message.strip())
    sector = (req.sector or "").upper()

    SECTOR_LABELS = {
        "CUL": "Culture & Traditions", "ART": "Arts & Music",
        "AGR": "Agriculture",          "ENV": "Environment & Nature",
        "EDU": "Education",            "SPR": "Spirituality",
        "DLY": "Daily Life",           "NAR": "Storytelling",
        "ECO": "Economy & Trade",      "GOV": "Governance",
        "HIS": "History",              "HLT": "Health",
        "POL": "Politics",             "ALL": "All Sectors",
    }

    def to_runyoro(text: str) -> str:
        return _mt_translate(text, "en2lun") or text

    # ── Retrieve relevant corpus context ─────────────────────────────────────
    def corpus_context(query: str, k: int = 2) -> str:
        q_emb  = _sem_model.encode(query, convert_to_numpy=True)
        scores = st_util.cos_sim(q_emb, _index["embeddings"])[0].numpy()
        top    = np.argsort(scores)[::-1][:k]
        pairs  = []
        for i in top:
            if float(scores[i]) > 0.2:
                en  = _index["english_sentences"][i][:120]
                lun = _index["lunyoro_sentences"][i][:120]
                pairs.append(f'  "{en}" → "{lun}"')
        return "\n".join(pairs)

    def dict_context(code: str, n: int = 4) -> str:
        if code == "ALL":
            entries = [d for d in _dictionary if d.get("word") and d.get("definitionEnglish")][:n]
        else:
            entries = [d for d in _dictionary
                       if (d.get("domain") or "").upper() == code
                       and d.get("word") and d.get("definitionEnglish")][:n]
        return "\n".join(f'  {e["word"]} = {e["definitionEnglish"]}' for e in entries)

    # ── Build system prompt ───────────────────────────────────────────────────
    corpus_ctx   = corpus_context(msg)
    sector_label = SECTOR_LABELS.get(sector, "")
    dict_ctx     = dict_context(sector) if sector else ""

    system_prompt = (
        "You are an expert AI assistant for the Runyoro-Rutooro language of the Bunyoro-Kitara and Tooro kingdoms in Uganda.\n"
        "Answer questions about the language, grammar, culture, vocabulary, and translation. "
        "Be conversational and accurate.\n"
        "IMPORTANT: Write in short, simple sentences. Each sentence should be clear and direct. "
        "Avoid complex clauses, passive voice, and long compound sentences. "
        "This helps with accurate translation.\n"
    )
    if corpus_ctx:
        system_prompt += f"\nRelevant examples (English → Runyoro-Rutooro):\n{corpus_ctx}\n"
    if sector_label:
        system_prompt += f"\nSector focus: {sector_label}\n"
    if dict_ctx:
        system_prompt += f"Vocabulary:\n{dict_ctx}\n"
    system_prompt += "\nAlways reply in English. Be clear and concise."

    # ── Build message history for Ollama ─────────────────────────────────────
    messages = [{"role": "system", "content": system_prompt}]
    for turn in (req.history or [])[-10:]:
        role    = turn.get("role", "")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": msg})

    # ── Call Ollama ───────────────────────────────────────────────────────────
    try:
        resp = _requests.post(
            "http://localhost:11434/api/chat",
            json={"model": "llama3.2:3b", "messages": messages, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        reply_en = resp.json()["message"]["content"].strip()
    except Exception as e:
        reply_en = "I am your Runyoro-Rutooro language assistant. I can help you translate, explain grammar, and discuss culture."

    # ── Always translate the reply to Runyoro-Rutooro ────────────────────────
    from language_rules import apply_rl_rule_to_text
    translated = to_runyoro(reply_en)
    if translated:
        translated = apply_rl_rule_to_text(translated)
    reply = translated or reply_en

    return {"reply": reply}

@app.get("/language-rules")
def get_language_rules():
    """Return language rules, interjections, idioms, numbers and proverbs."""
    from language_rules import (
        RL_RULE, EMPAAKO, INTERJECTIONS, IDIOMS, NUMBERS, PROVERBS, get_grammar_context
    )
    return {
        "rl_rule": RL_RULE.strip(),
        "grammar_summary": get_grammar_context().strip(),
        "empaako": EMPAAKO,
        "interjections": INTERJECTIONS,
        "idioms": IDIOMS,
        "numbers": {str(k): v for k, v in NUMBERS.items()},
        "proverbs": PROVERBS,
    }


@app.get("/language-rules/interjections")
def get_interjections():
    from language_rules import INTERJECTIONS
    return {"interjections": INTERJECTIONS}


@app.get("/language-rules/idioms")
def get_idioms():
    from language_rules import IDIOMS
    return {"idioms": IDIOMS}


@app.get("/language-rules/proverbs")
def get_proverbs():
    from language_rules import PROVERBS
    import random
    return {"proverbs": PROVERBS, "random": random.choice(PROVERBS)}
