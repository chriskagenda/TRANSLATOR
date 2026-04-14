# Lunyoro-Rutooro Translator

An AI-powered translation system for the Runyoro-Rutooro language of the Bunyoro-Kitara and Tooro kingdoms in Uganda.

## Features

- English ↔ Lunyoro/Rutooro translation (MarianMT + NLLB-200)
- Dictionary lookup with example sentences
- AI chat assistant powered by LLaMA 3.2 (via Ollama)
- PDF/DOCX document summarization and translation
- Voice translation
- Spellcheck with R/L rule enforcement
- Domain-aware translation (Medical, Education, Agriculture, etc.)

## Requirements

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.com) (for AI chat)

## Quick Setup

### Linux / macOS
```bash
cd lunyoro-translator
bash setup.sh
```

### Windows
```bat
cd lunyoro-translator
setup.bat
```

Or manually:

```bash
# 1. Python backend
pip install -r backend/requirements.txt

# 2. Frontend
cd frontend && npm install

# 3. Ollama — download from https://ollama.com/download
ollama pull llama3.2:3b
```

## Running the App

Open 3 terminals:

```bash
# Terminal 1 — Backend API (port 8000)
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend (port 3002)
cd frontend
npm run dev

# Terminal 3 — Ollama (if not running as a service)
ollama serve
```

Then open **http://localhost:3002**

## Training

To rebuild training data and retrain models:

```bash
cd backend

# 1. Merge new submissions and rebuild training splits
python clean_new_submissions.py

# 2. Retrain MarianMT models
python fine_tune.py --direction both --epochs 10 --batch_size 32

# 3. Retrain NLLB models
python fine_tune_nllb.py --direction both --epochs 10 --batch_size 4
```

## Architecture

```
backend/
  main.py                  — FastAPI server
  translate.py             — Translation logic (MarianMT + NLLB + retrieval)
  language_rules.py        — R/L rule, grammar, idioms, proverbs
  prepare_training_data.py — Corpus builder with domain tagging + R/L augmentation
  clean_new_submissions.py — Merges new crowd-sourced submissions
  fine_tune.py             — MarianMT fine-tuning
  fine_tune_nllb.py        — NLLB-200 fine-tuning
  model/
    en2lun/                — MarianMT English→Lunyoro
    lun2en/                — MarianMT Lunyoro→English
    nllb_en2lun/           — NLLB-200 English→Lunyoro
    nllb_lun2en/           — NLLB-200 Lunyoro→English
    sem_model/             — Sentence transformer for semantic search

frontend/
  components/
    Translator.tsx         — Main translation UI
    Dictionary.tsx         — Dictionary lookup
    ChatPage.tsx           — AI chat assistant
    PdfTranslator.tsx      — Document summarization
    VoiceTranslator.tsx    — Voice input/output
    History.tsx            — Translation history
```

## Chat (LLM)

The chat assistant uses **LLaMA 3.2 3B** running locally via Ollama. It generates responses in English, which are then translated to Runyoro-Rutooro by the fine-tuned MarianMT model. No internet connection required after setup.
