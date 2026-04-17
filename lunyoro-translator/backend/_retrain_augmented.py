"""
Full retraining pipeline on the augmented dataset (with back-translated pairs).
"""
import subprocess, sys, os

BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

scripts = [
    ("prepare_training_data.py", []),
    ("fine_tune.py", ["--direction", "both", "--epochs", "5"]),
    ("fine_tune_nllb.py", ["--direction", "both"]),
    # Back-translate with the newly trained lun2en model to further augment data
    ("back_translate.py", []),
    # Rebuild splits with the new back-translated pairs
    ("prepare_training_data.py", []),
    # Final retrain on fully augmented corpus
    ("fine_tune.py", ["--direction", "both", "--epochs", "5"]),
    ("fine_tune_nllb.py", ["--direction", "both"]),
    ("train.py", []),
    ("build_lunyoro_vocab.py", []),
    ("patch_index.py", []),
]

for script, args in scripts:
    cmd = [PY, os.path.join(BASE, script)] + args
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    r = subprocess.run(cmd, cwd=BASE)
    if r.returncode != 0:
        print(f"ERROR: {script} failed (exit {r.returncode})")
        sys.exit(r.returncode)

print("\nAll done. Committing and pushing...")
subprocess.run(["git", "add", "-A"], cwd=os.path.join(BASE, ".."))
subprocess.run(["git", "commit", "-m", "Retrain all models on augmented dataset (51k pairs)"], cwd=os.path.join(BASE, ".."))
subprocess.run(["git", "push"], cwd=os.path.join(BASE, ".."))
print("Complete.")
