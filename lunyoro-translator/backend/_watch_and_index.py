"""
Polls until fine_tune_nllb.py is no longer running,
then runs train.py → build_lunyoro_vocab.py → patch_index.py.
"""
import subprocess
import sys
import time
import os

BACKEND = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def nllb_still_running() -> bool:
    # Use PowerShell to get full command lines of running python processes
    result = subprocess.run(
        ["powershell", "-Command",
         "Get-WmiObject Win32_Process -Filter \"name='python.exe'\" | Select-Object -ExpandProperty CommandLine"],
        capture_output=True, text=True
    )
    return "fine_tune_nllb" in result.stdout


print("Watcher: waiting for fine_tune_nllb.py to finish...")
while nllb_still_running():
    en2lun = os.path.join(BACKEND, "model", "nllb_en2lun", "model.safetensors")
    lun2en = os.path.join(BACKEND, "model", "nllb_lun2en", "model.safetensors")
    print(f"  still training... nllb_en2lun saved={os.path.exists(en2lun)}  nllb_lun2en saved={os.path.exists(lun2en)}")
    time.sleep(60)

print("Watcher: training done. Starting index build...")

for script in ["train.py", "build_lunyoro_vocab.py", "patch_index.py"]:
    print(f"\n>>> python {script}")
    r = subprocess.run([PY, os.path.join(BACKEND, script)], cwd=BACKEND)
    if r.returncode != 0:
        print(f"ERROR: {script} failed (exit {r.returncode})")
        sys.exit(r.returncode)
    print(f"Done: {script}")

print("\nAll done. Index + vocab whitelist rebuilt.")
