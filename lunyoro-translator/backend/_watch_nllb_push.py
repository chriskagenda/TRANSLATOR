"""
Waits for fine_tune_nllb.py to finish, then rebuilds index,
uploads to HuggingFace and pushes to git.
"""
import subprocess, sys, time, os

BASE     = os.path.dirname(os.path.abspath(__file__))
REPO     = os.path.join(BASE, "..")
PY       = sys.executable
HF_TOKEN = os.environ.get("HF_TOKEN", "")

HF_MODELS = [
    ("model/en2lun",      "keithtwesigye/lunyoro-en2lun"),
    ("model/lun2en",      "keithtwesigye/lunyoro-lun2en"),
    ("model/nllb_en2lun", "keithtwesigye/lunyoro-nllb_en2lun"),
    ("model/nllb_lun2en", "keithtwesigye/lunyoro-nllb_lun2en"),
]


def is_running(script):
    r = subprocess.run(
        ["powershell", "-Command",
         "Get-WmiObject Win32_Process -Filter \"name='python.exe'\" | Select-Object -ExpandProperty CommandLine"],
        capture_output=True, text=True)
    return script in r.stdout


def run(cmd, cwd=BASE):
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, cwd=cwd)


print("Watcher: waiting for fine_tune_nllb.py to finish...")
while is_running("fine_tune_nllb"):
    print("  still training NLLB...")
    time.sleep(60)
print("NLLB training done.")

# Rebuild index + whitelist
for script in ["train.py", "build_lunyoro_vocab.py", "patch_index.py"]:
    run([PY, os.path.join(BASE, script)])

# Upload to HuggingFace
if HF_TOKEN:
    upload_script = os.path.join(BASE, "_hf_upload.py")
    with open(upload_script, "w") as f:
        f.write(f"""
import os, glob
from huggingface_hub import HfApi
api   = HfApi(token=os.environ['HF_TOKEN'])
BASE  = r'{BASE}'
models = {HF_MODELS!r}
for local, repo in models:
    print(f'Uploading {{local}} -> {{repo}}...')
    files = [f for f in glob.glob(os.path.join(BASE, local, '**'), recursive=True) if os.path.isfile(f)]
    for fpath in files:
        path_in_repo = os.path.relpath(fpath, os.path.join(BASE, local)).replace(os.sep, '/')
        try:
            api.upload_file(path_or_fileobj=fpath, path_in_repo=path_in_repo, repo_id=repo, repo_type='model')
        except Exception as e:
            print(f'  ERROR: {{e}}')
    print(f'Done: {{repo}}')
print('HuggingFace upload complete.')
""")
    env = os.environ.copy()
    env["HF_TOKEN"] = HF_TOKEN
    subprocess.run([PY, upload_script], cwd=BASE, env=env)

# Git push — force push, never pull
run(["git", "add", "-A"], cwd=REPO)
run(["git", "commit", "-m", "Retrain NLLB on clean 46k corpus + index rebuild"], cwd=REPO)
run(["git", "push", "--force-with-lease"], cwd=REPO)
print("\nAll done. Everything pushed.")
