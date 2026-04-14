@echo off
REM ─────────────────────────────────────────────────────────────
REM Lunyoro-Rutooro Translator — Windows Setup Script
REM ─────────────────────────────────────────────────────────────

echo === Lunyoro-Rutooro Translator Setup ===

echo.
echo [1/5] Installing Python dependencies...
pip install -r backend\requirements.txt
IF %ERRORLEVEL% NEQ 0 ( echo ERROR: pip install failed & pause & exit /b 1 )

echo.
echo [2/5] Installing frontend dependencies...
cd frontend && npm install && cd ..
IF %ERRORLEVEL% NEQ 0 ( echo ERROR: npm install failed & pause & exit /b 1 )

echo.
echo [3/5] Downloading translation models from HuggingFace...
python backend\download_models.py
IF %ERRORLEVEL% NEQ 0 ( echo ERROR: model download failed & pause & exit /b 1 )

echo.
echo [4/5] Setting up Ollama (LLM for chat)...
where ollama >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo   Ollama not found. Downloading installer...
    powershell -Command "Invoke-WebRequest -Uri 'https://ollama.com/download/OllamaSetup.exe' -OutFile 'OllamaSetup.exe'"
    echo   Running Ollama installer (follow the prompts)...
    start /wait OllamaSetup.exe
    del OllamaSetup.exe
)
echo   Pulling qwen3.5:latest model (~6.6GB)...
ollama pull qwen3.5:latest
IF %ERRORLEVEL% NEQ 0 ( echo WARNING: Could not pull qwen3.5:latest. Start Ollama manually and run: ollama pull qwen3.5:latest )

echo.
echo [5/5] Setup complete!
echo.
echo To run the app:
echo   Terminal 1 (backend):  cd backend ^&^& uvicorn main:app --reload --port 8000
echo   Terminal 2 (frontend): cd frontend ^&^& npm run dev
echo.
echo Then open http://localhost:3002
pause
