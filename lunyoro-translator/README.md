# Lunyoro-Rutooro Translator

An AI-powered translation system for the Runyoro-Rutooro language of the Bunyoro-Kitara and Tooro kingdoms in Uganda.

## Features

- English ↔ Lunyoro/Rutooro translation (MarianMT + NLLB-200)
- Dictionary lookup with example sentences
- AI chat assistant powered by LLaMA 3.2 (via Ollama)
- PDF/DOCX document summarization and translation
- Voice translation
- Spellcheck
- Domain-aware translation (Medical, Education, Agriculture, etc.)
- Model comparison view (MarianMT vs NLLB-200)

## Dataset

- ~53,948 English-Lunyoro sentence pairs
- Sources:
  - `english_nyoro_clean.csv` — main sentence pairs with domain tagging
  - `runyoro_english_sentences_clean.csv` — April crowd-sourced sentence submissions
  - `rutooro_dictionary_clean.csv` — word/definition/example pairs from the Rutooro dictionary
  - `word_entries_clean.csv` — dictionary example sentences and word-level definition pairs
  - `empaako_pairs.csv`, `idioms_pairs.csv`, `numbers_pairs.csv`, `interjections_pairs_clean.csv`, `proverbs_pairs_clean.csv` — small cultural/linguistic extras
- Augmented via back-translation using the fine-tuned lun2en model
- Cleaned with quality filters: deduplication, length checks, hallucination detection, round-trip consistency

## Models

All models are hosted on HuggingFace under [keithtwesigye](https://huggingface.co/keithtwesigye):

| Model | Repo | Description |
|-------|------|-------------|
| MarianMT en→lun | `keithtwesigye/lunyoro-en2lun` | English to Lunyoro |
| MarianMT lun→en | `keithtwesigye/lunyoro-lun2en` | Lunyoro to English |
| NLLB-200 en→lun | `keithtwesigye/lunyoro-nllb_en2lun` | English to Lunyoro (NLLB) |
| NLLB-200 lun→en | `keithtwesigye/lunyoro-nllb_lun2en` | Lunyoro to English (NLLB) |

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

# 2. Download models from HuggingFace
cd backend
python download_models.py

# 3. Frontend
cd ../frontend && npm install

# 4. Ollama — download from https://ollama.com/download
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

# (Alternative) Clean and merge previously unprocessed raw files
# (word_submissions_rows, word_entries_rows_root, sentence_submissions_rows)
# then rebuild training splits
python clean_unprocessed_raw.py

# 2. (Optional) Back-translation augmentation
python back_translate.py
python clean_backtranslated.py

# (Alternative) Full improvement pipeline in one shot:
# back-translates sentence_submissions, quality-filters, rebuilds splits,
# then fine-tunes both MarianMT (15 epochs, LR=2e-5) and NLLB (8 epochs, label_smoothing=0.2) from existing checkpoints,
# NLLB uses DDP across multiple GPUs if available (gloo backend for Windows compatibility),
# then pushes all 4 models + updated dataset to HuggingFace and commits/pushes to GitHub
python improve_and_retrain.py

# Skip data preparation steps (back-translation, filtering, splits) and go straight to fine-tuning:
python improve_and_retrain.py --skip-data

# Skip training entirely — only push existing models and dataset to HuggingFace:
python improve_and_retrain.py --skip-train

# 3. Retrain MarianMT models
python fine_tune.py --direction both --epochs 10 --batch_size 32

# 4. Retrain NLLB models
python fine_tune_nllb.py --direction both --epochs 10 --batch_size 4

# 5. (Optional) Evaluate all 4 models on the test set
# Single-process (any hardware):
python eval_models.py
# Results saved to eval_results_full.json (BLEU, token F1, exact match)

# Sequential single-GPU evaluation (auto-selects GPU with most free memory):
# Useful when Ollama or another process occupies a GPU — avoids hardcoded cuda:0/cuda:1
python run_eval.py
# Results saved to eval_results_all.json (BLEU, token F1, exact match, ms/sample)

# MarianMT-only evaluation (runs on CPU, leaving both GPUs free for NLLB training):
# Useful for a quick check of just the two MarianMT models
python eval_marian.py
# Results saved to eval_marian_results.json (BLEU, token F1, exact match)

# Multi-GPU parallel evaluation (requires 2 GPUs):
# GPU 0 runs MarianMT en2lun + lun2en; GPU 1 runs NLLB en2lun + lun2en simultaneously
python eval_all_parallel.py
# Results saved to eval_results_all.json (BLEU, token F1, exact match, ms/sample)
```

## Publishing Models to HuggingFace

To push README files to the 4 HuggingFace model repos, set your HuggingFace token as an environment variable before running the script:

```bash
# Linux / macOS
export HF_TOKEN=your_token_here
python backend/_push_hf_readmes.py

# Windows
set HF_TOKEN=your_token_here
python backend\_push_hf_readmes.py
```

You can generate a token at https://huggingface.co/settings/tokens (needs write access to the target repos).

## Architecture

```
backend/
  main.py                    — FastAPI server
  translate.py               — Translation logic (MarianMT + NLLB + retrieval)
  language_rules.py          — Orthography constants (alphabet, vowels, diphthongs, apostrophe contexts), R/L rule, noun class system (classes 1–15 with prefixes), get_noun_class() for morphological prefix analysis, concordial agreement table (subject/object/adjective concords per class) with get_subject_concord()/get_object_concord() helpers, plural sound changes (class 11→10), get_class6_prefix() for class 5→6 plural prefix selection, verb structure constants (infinitive prefixes, subject prefixes, tense/aspect markers, negative markers, verb suffixes, derivative suffixes), full tense system reference (TENSES dict + CONDITIONAL_PARTICLES), adjective/adverb constants (COMPARISON degrees, ADJECTIVE_STEMS, ADVERBS_OF_MANNER), number system (NUMBERS 1–1B, NUMERAL_CONCORDS per noun class, ORDINAL_NOTE), particles/conjunctions/prepositions (CONJUNCTIONS, PREPOSITIONS, NEGATION_WORDS, NYA_PARTICLE), pronouns (PERSONAL_PRONOUNS, OBJECT_PRONOUNS), language names (LANGUAGE_NAMES), augmentative/pejorative prefix examples (AUGMENTATIVE_EXAMPLES, MAGNITUDE_EXAMPLES, MAGNITUDE_ERI_EXAMPLES), honorific names (EMPAAKO), interjections (INTERJECTIONS), idiomatic expressions (IDIOMS), and proverbs (PROVERBS)
  prepare_training_data.py   — Corpus builder with domain tagging
  clean_new_submissions.py   — Merges new crowd-sourced submissions
  clean_unprocessed_raw.py   — Cleans and merges previously unprocessed raw files (word_submissions_rows, word_entries_rows_root, sentence_submissions_rows) into english_nyoro_clean.csv and word_entries_clean.csv, then rebuilds train/val/test splits
  extract_ocr_pairs.py       — Extracts English ↔ Runyoro-Rutooro sentence pairs from OCR grammar data (data/OCR/combined/all_ocr_combined.json) and merges them into english_nyoro_clean.csv; uses heuristic language detection (Runyoro prefix patterns + English function words) to validate pairs before merging; outputs data/cleaned/ocr_pairs_extracted.csv for review; run directly with `python extract_ocr_pairs.py`
  push_models_hf.py            — Uploads all 4 fine-tuned model folders (en2lun, lun2en, nllb_en2lun, nllb_lun2en) to their respective HuggingFace repos; requires HF_TOKEN env var with write access
  clean_sentence_submission.py — Cleans the April sentence submission Excel file (Runyoro-English_Translation.xlsx); standardises columns, strips whitespace, drops empty rows and duplicates, writes data/cleaned/runyoro_english_sentences_clean.csv
  clean_remaining_raw.py     — Extracts translation pairs from remaining raw CSVs (word_submissions_rows*.csv, corpus_sentences_rows (1).csv); merges new pairs into english_nyoro_clean.csv after deduplication
  clean_extra.py             — Merges Excel dictionary datasets
  clean_dictionaries.py      — Cleans and converts Rutooro/Runyoro Excel dictionary files to CSV; normalises column names, strips definition noise (grammar notation, cross-references, OCR-duplicated phrases), deduplicates entries, and writes data/cleaned/rutooro_dictionary_clean.csv
  inspect_raw.py             — Inspects raw CSV files: prints row counts, column names, null counts, and sample rows for word/sentence submission and corpus files
  audit_csvs.py              — Audits all CSV files in data/: reports row counts, nulls, duplicates, and whether a cleaned version exists
  check_dups.py              — Checks whether pairs of raw CSV files are identical (e.g. versioned duplicates like word_entries_rows.csv vs word_entries_rows (1).csv)
  verify_dict.py             — Verifies the cleaned dictionary CSV: prints row count, null counts per column, and sample rows that have both a definition and a Runyoro example sentence
  back_translate.py          — Back-translation augmentation
  clean_backtranslated.py    — Quality filtering for synthetic pairs
  improve_and_retrain.py     — Full improvement pipeline: back-translates sentence_submissions, quality-filters synthetic pairs, rebuilds training splits, fine-tunes MarianMT (15 epochs, LR=2e-5, label_smoothing=0.2) and NLLB (8 epochs, label_smoothing=0.2) from existing checkpoints, then pushes all 4 models + updated dataset to HuggingFace and commits/pushes code and data to GitHub. NLLB training uses DistributedDataParallel (DDP) via torchrun when 2+ GPUs are available (gloo backend, compatible with Windows); falls back to single-GPU automatically. Pass --skip-data to skip steps 1-3; pass --skip-train to skip training and only push existing models to HuggingFace
  fine_tune.py               — MarianMT fine-tuning
  fine_tune_nllb.py          — NLLB-200 fine-tuning
  eval_models.py             — Evaluates all 4 models on the test set (BLEU, token F1, exact match)
  eval_marian.py             — MarianMT-only evaluation: runs on CPU, leaving both GPUs free for concurrent NLLB training; outputs eval_marian_results.json (BLEU, token F1, exact match)
  run_eval.py                — Sequential single-GPU evaluation: auto-selects the GPU with the most free memory (avoids conflicts with Ollama); outputs eval_results_all.json
  eval_all_parallel.py       — Evaluates all 4 models in parallel across 2 GPUs (GPU 0: MarianMT, GPU 1: NLLB); outputs eval_results_all.json
  download_models.py         — Downloads all models from HuggingFace
  model/
    en2lun/                  — MarianMT English→Lunyoro
    lun2en/                  — MarianMT Lunyoro→English
    nllb_en2lun/             — NLLB-200 English→Lunyoro
    nllb_lun2en/             — NLLB-200 Lunyoro→English
    sem_model/               — Sentence transformer for semantic search

frontend/
  components/
    Translator.tsx           — Main translation UI
    Dictionary.tsx           — Dictionary lookup
    ChatPage.tsx             — AI chat assistant
    PdfTranslator.tsx        — Document summarization
    VoiceTranslator.tsx      — Voice input/output
    History.tsx              — Translation history
```

## Chat (LLM)

The chat assistant uses **LLaMA 3.2 3B** running locally via Ollama. It generates responses in English, which are then translated to Runyoro-Rutooro by the fine-tuned MarianMT model. No internet connection required after setup.
