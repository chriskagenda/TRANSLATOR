#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Lunyoro-Rutooro Translator — Full Setup Script (Linux/macOS)
# ─────────────────────────────────────────────────────────────
set -e

echo "=== Lunyoro-Rutooro Translator Setup ==="

# ── 1. Python dependencies ────────────────────────────────────
echo ""
echo "[1/5] Installing Python dependencies..."
pip install -r backend/requirements.txt

# ── 2. Node dependencies ──────────────────────────────────────
echo ""
echo "[2/5] Installing frontend dependencies..."
cd frontend && npm install && cd ..

# ── 3. Download models from HuggingFace ──────────────────────
echo ""
echo "[3/5] Downloading translation models from HuggingFace..."
python backend/download_models.py

# ── 4. Ollama ─────────────────────────────────────────────────
echo ""
echo "[4/5] Setting up Ollama (LLM for chat)..."
if command -v ollama &> /dev/null; then
    echo "  Ollama already installed."
else
    echo "  Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "  Pulling qwen3.5:latest model (~6.6GB)..."
ollama pull qwen3.5:latest

# ── 5. Done ───────────────────────────────────────────────────
echo ""
echo "[5/5] Setup complete!"
echo ""
echo "To run the app:"
echo "  Terminal 1 (backend):  cd backend && uvicorn main:app --reload --port 8000"
echo "  Terminal 2 (frontend): cd frontend && npm run dev"
echo ""
echo "Then open http://localhost:3002"
