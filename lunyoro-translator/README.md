# Lunyoro / Rutooro Translator

AI-powered translation between English and Lunyoro/Rutooro using fine-tuned MarianMT neural models.

## Quick Start (after cloning)

> Requires: Python 3.10+, Node.js 18+, Git LFS

### 1. Pull model files (Git LFS)
```bash
git lfs pull
```

### 2. Backend setup
```bash
cd lunyoro-translator/backend
python setup.py
```

### 3. Start backend
```bash
uvicorn main:app --reload --port 8000
```

### 4. Start frontend (new terminal)
```bash
cd lunyoro-translator/frontend
npm install
npm run dev
```

### 5. Open the app
```
http://localhost:3002
```

---

## Models

The fine-tuned models are stored in `backend/model/` via Git LFS — no training required after cloning.

| Model | Direction | Epochs | Best val_loss |
|-------|-----------|--------|---------------|
| `en2lun` | English → Lunyoro/Rutooro | 10 | 2.12 |
| `lun2en` | Lunyoro/Rutooro → English | 10 | 2.12 |

## Notes for other machines

- GPU (NVIDIA CUDA): translations run instantly, models load in ~5 seconds on startup
- CPU only: translations take ~3-5 seconds each, startup takes ~30 seconds — still works fine
- Git LFS must be installed before cloning, otherwise model files will be empty pointers
```bash
python prepare_training_data.py
python fine_tune.py --direction both --epochs 10 --batch_size 32
```

## Dataset

- 6,200 parallel sentence pairs (cleaned)
- 1,142 dictionary entries with definitions and examples
- 8,521 augmented training pairs (includes dictionary examples)
