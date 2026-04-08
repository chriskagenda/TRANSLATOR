"""
One-time setup script for new machines.
Run this after cloning the repo:
    python setup.py

It will:
  1. Install all Python dependencies (CUDA torch if GPU detected, CPU otherwise)
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
    print(f"  ✗ {label} — NOT FOUND or empty (run: git lfs pull)")
    return False


def detect_cuda():
    """Check if an NVIDIA GPU is available via nvidia-smi."""
    result = subprocess.run("nvidia-smi", shell=True, capture_output=True)
    return result.returncode == 0


print("=" * 60)
print("Lunyoro / Rutooro Translator — Setup")
print("=" * 60)

# 1. Install base dependencies (everything except torch)
run(f"{sys.executable} -m pip install fastapi uvicorn pandas scikit-learn "
    f"sentence-transformers transformers datasets sacrebleu sentencepiece "
    f"sacremoses python-dotenv pydantic rapidfuzz pypdf2 python-multipart")

# 2. Install correct torch build
if detect_cuda():
    print("\n✓ NVIDIA GPU detected — installing CUDA torch (cu124)...")
    run(f"{sys.executable} -m pip install torch torchvision torchaudio "
        f"--index-url https://download.pytorch.org/whl/cu124")
else:
    print("\n  No GPU detected — installing CPU torch...")
    run(f"{sys.executable} -m pip install torch torchvision torchaudio")

# 3. Verify GPU is usable after install
try:
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  ✓ GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("  Running on CPU")
except Exception:
    pass

# 4. Check model files
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

# 5. Download NLLB models from HuggingFace if not present
print("\nChecking NLLB models...")
run(f"{sys.executable} -m pip install huggingface_hub -q")
for direction, repo in [("nllb_en2lun", "chriskagenda/lunyoro-nllb_en2lun"),
                         ("nllb_lun2en", "chriskagenda/lunyoro-nllb_lun2en")]:
    safetensors = os.path.join(BASE, "model", direction, "model.safetensors")
    if not os.path.exists(safetensors) or os.path.getsize(safetensors) < 1000:
        print(f"  Downloading {direction} from HuggingFace...")
        run(f"{sys.executable} -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download(repo_id='{repo}', local_dir='model/{direction}')\"")
    else:
        print(f"  ✓ {direction} model")

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
