"""
Waits for the new lun2en model to be saved, then runs back_translate.py on CPU.
Runs in parallel with NLLB GPU training.
"""
import subprocess, sys, time, os

BASE = os.path.dirname(os.path.abspath(__file__))
PY   = sys.executable
LUN2EN_MODEL = os.path.join(BASE, "model", "lun2en", "model.safetensors")

import datetime

def model_mtime():
    if os.path.exists(LUN2EN_MODEL):
        return os.path.getmtime(LUN2EN_MODEL)
    return 0

# Record current mtime — wait for it to be updated by the new training run
baseline = model_mtime()
print(f"Watcher: baseline lun2en mtime = {datetime.datetime.fromtimestamp(baseline)}")
print("Waiting for new lun2en checkpoint to be saved...")

while True:
    current = model_mtime()
    if current > baseline:
        print(f"New lun2en checkpoint detected at {datetime.datetime.fromtimestamp(current)}")
        break
    time.sleep(30)

# Small buffer to ensure file write is complete
time.sleep(10)

print("\nStarting back-translation on CPU with new lun2en model...")
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = ""  # force CPU so GPUs stay free for NLLB
r = subprocess.run([PY, os.path.join(BASE, "back_translate.py")], cwd=BASE, env=env)
if r.returncode != 0:
    print(f"ERROR: back_translate.py failed (exit {r.returncode})")
    sys.exit(r.returncode)

print("Back-translation complete.")
