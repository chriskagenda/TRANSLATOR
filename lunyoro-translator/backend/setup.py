"""
One-time setup script for new machines.
Run this after cloning the repo:
    python setup.py

It will:
  1. Install all Python dependencies
  2. Verify the fine-tuned models are present (pulled via Git LFS)
  3. Verify the translation index exists
  4. Print instructions to start the app
"""
import subprocess
import sys
import os

BASE = os.path.dirname(__file__)


def run(cmd):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: command failed: {cmd}")
        sys.exit(1)


def check_file(path, label):
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        print(f"  ✓ {label}")
        return True
    print(f"  ✗ {label} — NOT FOUND or empty")
    return False


print("=" * 60)
print("Lunyoro / Rutooro Translator — Setup")
print("=" * 60)

# 1. Install dependencies
run(f"{sys.executable} -m pip install -r requirements.txt")

# 2. Check model files
print("\nChecking model files...")
all_ok = True
all_ok &= check_file(os.path.join(BASE, "model", "en2lun", "model.safetensors"), "en2lun model")
all_ok &= check_file(os.path.join(BASE, "model", "lun2en", "model.safetensors"), "lun2en model")
all_ok &= check_file(os.path.join(BASE, "model", "translation_index.pkl"),        "translation index")

if not all_ok:
    print("""
  Some model files are missing. Make sure Git LFS is installed and run:
    git lfs pull
  Then re-run this script.
""")
    sys.exit(1)

print("\n✓ All model files present. No training needed.")
print("""
==============================================
Setup complete. To run the app:

  Backend (terminal 1):
    cd lunyoro-translator/backend
    uvicorn main:app --reload --port 8000

  Frontend (terminal 2):
    cd lunyoro-translator/frontend
    npm install
    npm run dev

  Then open: http://localhost:3002
==============================================
""")
