"""
Waits for fine_tune.py (MarianMT) to finish, then runs nllb_en2lun,
then rebuilds the index and pushes everything.
"""
import subprocess, sys, time, os

BACKEND = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def is_running(script: str) -> bool:
    result = subprocess.run(
        ["powershell", "-Command",
         f"Get-WmiObject Win32_Process -Filter \"name='python.exe'\" | Select-Object -ExpandProperty CommandLine"],
        capture_output=True, text=True
    )
    return script in result.stdout


print("Watcher: waiting for fine_tune.py (MarianMT) to finish...")
while is_running("fine_tune.py"):
    print("  MarianMT still training...")
    time.sleep(60)

print("MarianMT done. Starting nllb_en2lun...")
r = subprocess.run([PY, os.path.join(BACKEND, "fine_tune_nllb.py"), "--direction", "en2lun"], cwd=BACKEND)
if r.returncode != 0:
    print(f"ERROR: fine_tune_nllb.py failed (exit {r.returncode})")
    sys.exit(r.returncode)

print("nllb_en2lun done. Rebuilding index...")
for script in ["train.py", "build_lunyoro_vocab.py", "patch_index.py"]:
    subprocess.run([PY, os.path.join(BACKEND, script)], cwd=BACKEND)

print("All training complete. Pushing to git...")
subprocess.run(
    ["git", "add", "-A"],
    cwd=os.path.join(BACKEND, "..")
)
subprocess.run(
    ["git", "commit", "-m", "Retrain all models with new dataset: MarianMT both + NLLB en2lun"],
    cwd=os.path.join(BACKEND, "..")
)
subprocess.run(["git", "push"], cwd=os.path.join(BACKEND, ".."))
print("Done. All models retrained and pushed.")
