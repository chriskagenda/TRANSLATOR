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
    """Load retrieval index and neural MT models at startup."""
    get_index_and_model()
    # preload fine-tuned MT models so first request is instant
    from translate import _load_mt
    _load_mt("en2lun")
    _load_mt("lun2en")

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


class WordLookupRequest(BaseModel):
    word: str

class SpellCheckRequest(BaseModel):
    text: str

class GenerateRequest(BaseModel):
    prompt: str


@app.get("/")
def root():
    return {"message": "Lunyoro/Rutooro Translator API is running"}


@app.post("/translate")
def translate_text(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(req.text) > 1000:
        raise HTTPException(status_code=400, detail="Text too long (max 1000 chars)")
    result = translate(req.text)
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
    result = translate_to_english(req.text)
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
    results = lookup_word(req.word)
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


@app.post("/generate")
def generate_lunyoro(req: GenerateRequest):
    """
    Takes an English prompt and generates a Lunyoro/Rutooro response.
    Translates the prompt to Lunyoro, then builds a contextual reply
    by finding semantically related Lunyoro sentences from the corpus.
    """
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    if len(req.prompt) > 500:
        raise HTTPException(status_code=400, detail="Prompt too long (max 500 chars)")

    import numpy as np
    from sentence_transformers import util
    from translate import _mt_translate

    # 1. Translate the prompt to Lunyoro
    translated_prompt = _mt_translate(req.prompt, "en2lun")
    if not translated_prompt:
        result = translate(req.prompt)
        translated_prompt = result.get("translation") or req.prompt

    # 2. Find related Lunyoro sentences from the corpus for context
    _index, _sem_model = get_index_and_model()
    english_sentences = _index["english_sentences"]
    lunyoro_sentences = _index["lunyoro_sentences"]

    q_emb = _sem_model.encode(req.prompt, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, _index["embeddings"])[0].numpy()
    top_idx = np.argsort(scores)[::-1][:5]

    related = [
        {"english": english_sentences[i], "lunyoro": lunyoro_sentences[i], "score": round(float(scores[i]), 3)}
        for i in top_idx if float(scores[i]) > 0.3
    ]

    # 3. Build a generated Lunyoro response by composing related sentences
    if related:
        # Use the top match as the primary response, append supporting sentences
        primary = related[0]["lunyoro"]
        supporting = [r["lunyoro"] for r in related[1:3] if r["score"] > 0.4]
        generated = primary
        if supporting:
            generated = generated + " " + " ".join(supporting)
    else:
        generated = translated_prompt

    save_history({
        "input": req.prompt,
        "direction": "en→lun",
        "translation": generated,
        "method": "generate",
        "confidence": round(float(scores[top_idx[0]]), 3) if len(top_idx) else 0,
        "timestamp": datetime.utcnow().isoformat(),
    })

    return {
        "prompt": req.prompt,
        "prompt_translated": translated_prompt,
        "generated": generated,
        "related": related,
    }


@app.post("/translate-pdf")
async def translate_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise HTTPException(status_code=500, detail="pypdf2 not installed")

    import re
    from translate import get_index_and_model
    _index, _model = get_index_and_model()

    contents = await file.read()
    reader = PdfReader(io.BytesIO(contents))

    # extract all sentences across all pages first
    page_sentences: list[list[str]] = []
    all_sentences: list[str] = []
    for page in reader.pages:
        raw = page.extract_text() or ""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw) if len(s.strip()) >= 3]
        page_sentences.append(sentences)
        all_sentences.extend(sentences)

    if not all_sentences:
        raise HTTPException(status_code=400, detail="No text found in PDF")

    # batch encode all sentences in one shot — much faster than one by one
    from sentence_transformers import util
    import numpy as np

    query_embeddings = _model.encode(all_sentences, batch_size=64, show_progress_bar=False)
    corpus_embeddings = _index["embeddings"]
    english_sentences = _index["english_sentences"]
    lunyoro_sentences = _index["lunyoro_sentences"]

    # cosine similarity for all queries at once
    scores_matrix = util.cos_sim(query_embeddings, corpus_embeddings).numpy()

    # build results
    flat_results: list[dict] = []
    for i, sentence in enumerate(all_sentences):
        scores = scores_matrix[i]
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        flat_results.append({
            "original": sentence,
            "translation": lunyoro_sentences[best_idx] if best_score > 0.4 else sentence,
            "confidence": round(best_score, 3),
            "method": "semantic_match" if best_score > 0.4 else "no_match",
        })

    # reassemble into pages
    pages_translated = []
    idx = 0
    for page_num, sentences in enumerate(page_sentences, start=1):
        pages_translated.append({
            "page": page_num,
            "sentences": flat_results[idx: idx + len(sentences)],
        })
        idx += len(sentences)

    save_history({
        "input": f"[PDF] {file.filename}",
        "direction": "en→lun",
        "translation": f"[{len(all_sentences)} sentences translated]",
        "method": "pdf",
        "confidence": None,
        "timestamp": datetime.utcnow().isoformat(),
    })

    return {
        "filename": file.filename,
        "total_pages": len(reader.pages),
        "pages": pages_translated,
    }
