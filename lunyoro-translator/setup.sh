#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Lunyoro-Rutooro Translator — Full Setup Script
# ─────────────────────────────────────────────────────────────

set -e

echo "=== Lunyoro-Rutooro Translator Setup ==="

# ── 1. Python dependencies ────────────────────────────────────
echo ""
echo "[1/4] Installing Python dependencies..."
pip install -r backend/requirements.txt

# ── 2. Node dependencies ──────────────────────────────────────
echo ""
echo "[2/4] Installing frontend dependencies..."
cd frontend && npm install && cd ..

# ── 3. Ollama ─────────────────────────────────────────────────
echo ""
echo "[3/4] Setting up Ollama (LLM for chat)..."

if command -v ollama &> /dev/null; then
    echo "  Ollama already installed."
else
    echo "  Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "  Pulling llama3.2:3b model (~2GB)..."
ollama pull llama3.2:3b

# ── 4. Done ───────────────────────────────────────────────────
echo ""
echo "[4/4] Setup complete!"
echo ""
echo "To run the app:"
echo "  Terminal 1 (backend):  cd backend && uvicorn main:app --reload --port 8000"
echo "  Terminal 2 (frontend): cd frontend && npm run dev"
echo "  Terminal 3 (ollama):   ollama serve   (if not already running as a service)"
echo ""
echo "Then open http://localhost:3002"
