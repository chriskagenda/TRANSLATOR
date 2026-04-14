@echo off
REM ─────────────────────────────────────────────────────────────
REM Lunyoro-Rutooro Translator — Windows Setup Script
REM ─────────────────────────────────────────────────────────────

echo === Lunyoro-Rutooro Translator Setup ===

echo.
echo [1/4] Installing Python dependencies...
pip install -r backend\requirements.txt

echo.
echo [2/4] Installing frontend dependencies...
cd frontend && npm install && cd ..

echo.
echo [3/4] Setting up Ollama...
where ollama >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo   Ollama not found. Please download and install from:
    echo   https://ollama.com/download/windows
    echo   Then re-run this script.
    pause
    exit /b 1
) ELSE (
    echo   Ollama found. Pulling llama3.2:3b model (~2GB)...
    ollama pull llama3.2:3b
)

echo.
echo [4/4] Setup complete!
echo.
echo To run the app:
echo   Terminal 1 (backend):  cd backend ^&^& uvicorn main:app --reload --port 8000
echo   Terminal 2 (frontend): cd frontend ^&^& npm run dev
echo   Ollama runs as a background service automatically on Windows.
echo.
echo Then open http://localhost:3002
pause
