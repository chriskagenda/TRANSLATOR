"""
Full automated pipeline:
  1. Wait for _retrain_augmented.py to finish
  2. Clean back-translated data (round-trip + semantic filter)
  3. Rebuild training splits from clean data
  4. Retrain all models on clean corpus
  5. Rebuild index + vocab whitelist
  6. Upload all 4 models to HuggingFace (replaces old models)
  7. git add -A + commit + push (LFS + code)

Set HF_TOKEN env var before running:
  $env:HF_TOKEN = "your_token_here"
"""
import subprocess, sys, time, os

BASE      = os.path.dirname(os.path.abspath(__file__))
REPO      = os.path.join(BASE, "..")
PY        = sys.executable
# Token read from environment — never hardcoded
HF_TOKEN  = os.environ.get("HF_TOKEN", "")

HF_MODELS = [
    ("model/en2lun",      "keithtwesigye/lunyoro-en2lun"),
    ("model/lun2en",      "keithtwesigye/lunyoro-lun2en"),
    ("model/nllb_en2lun", "keithtwesigye/lunyoro-nllb_en2lun"),
    ("model/nllb_lun2en", "keithtwesigye/lunyoro-nllb_lun2en"),
]


def is_running(script: str) -> bool:
    result = subprocess.run(
        ["powershell", "-Command",
         "Get-WmiObject Win32_Process -Filter \"name='python.exe'\" "
         "| Select-Object -ExpandProperty CommandLine"],
        capture_output=True, text=True
    )
    return script in result.stdout


def run(cmd, cwd=BASE, check=False):
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    r = subprocess.run(cmd, cwd=cwd)
    if check and r.returncode != 0:
        print(f"ERROR: command failed (exit {r.returncode})")
        sys.exit(r.returncode)
    return r.returncode


# ── Step 1: wait for current training ────────────────────────────────────────
print("=" * 60)
print("Step 1: Waiting for _retrain_augmented.py to finish...")
print("=" * 60)
while is_running("_retrain_augmented.py"):
    print("  still training...")
    time.sleep(60)
print("Training done.")


# ── Step 2: clean back-translated data ───────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: Cleaning back-translated data...")
print("=" * 60)
run([PY, os.path.join(BASE, "clean_backtranslated.py")], check=True)


# ── Step 3: rebuild training splits ──────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: Rebuilding training splits from clean corpus...")
print("=" * 60)
run([PY, os.path.join(BASE, "prepare_training_data.py")], check=True)


# ── Step 4: retrain all models on clean data ─────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: Retraining all models on clean corpus...")
print("=" * 60)
run([PY, os.path.join(BASE, "fine_tune.py"), "--direction", "both", "--epochs", "5"], check=True)
run([PY, os.path.join(BASE, "fine_tune_nllb.py"), "--direction", "both"], check=True)


# ── Step 5: rebuild index + whitelist ────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 5: Rebuilding index and vocab whitelist...")
print("=" * 60)
for script in ["train.py", "build_lunyoro_vocab.py", "patch_index.py"]:
    run([PY, os.path.join(BASE, script)])


# ── Step 6: upload to HuggingFace ────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 6: Uploading models to HuggingFace...")
print("=" * 60)

if not HF_TOKEN:
    print("  WARNING: HF_TOKEN not set — skipping HuggingFace upload")
    print("  Set it with: $env:HF_TOKEN = 'your_token'")
else:
    upload_script = os.path.join(BASE, "_hf_upload.py")
    with open(upload_script, "w") as f:
        f.write(f"""
import os, glob
from huggingface_hub import HfApi

api   = HfApi(token=os.environ['HF_TOKEN'])
BASE  = r'{BASE}'
models = {HF_MODELS!r}

for local, repo in models:
    print(f'\\nUploading {{local}} -> {{repo}}...')
    files = [f for f in glob.glob(os.path.join(BASE, local, '**'), recursive=True) if os.path.isfile(f)]
    for fpath in files:
        path_in_repo = os.path.relpath(fpath, os.path.join(BASE, local)).replace(os.sep, '/')
        size_mb = round(os.path.getsize(fpath) / 1024 / 1024, 1)
        print(f'  {{path_in_repo}} ({{size_mb}} MB)...')
        try:
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=path_in_repo,
                repo_id=repo,
                repo_type='model',
            )
        except Exception as e:
            print(f'  ERROR: {{e}}')
    print(f'Done: {{repo}}')

print('All HuggingFace uploads complete.')
""")
    env = os.environ.copy()
    env["HF_TOKEN"] = HF_TOKEN
    subprocess.run([PY, upload_script], cwd=BASE, env=env)


# ── Step 7: git commit + push ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 7: Pushing to git (LFS + code)...")
print("=" * 60)
run(["git", "add", "-A"], cwd=REPO)
run(["git", "commit", "-m",
     "Retrain on clean 59k corpus (round-trip + semantic filtered) — replaces dirty models"],
    cwd=REPO)
run(["git", "push"], cwd=REPO)

print("\n" + "=" * 60)
print("ALL DONE. Clean models pushed to HuggingFace, LFS, and git.")
print("=" * 60)
