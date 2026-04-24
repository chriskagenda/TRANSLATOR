"""Push all 4 models to HuggingFace repos."""
from huggingface_hub import HfApi
from pathlib import Path
import os

TOKEN = os.environ.get("HF_TOKEN", "")
api = HfApi(token=TOKEN)

MODEL_DIR = Path(__file__).parent / "model"

models = [
    ("en2lun",      "keithtwesigye/lunyoro-en2lun"),
    ("lun2en",      "keithtwesigye/lunyoro-lun2en"),
    ("nllb_en2lun", "keithtwesigye/lunyoro-nllb_en2lun"),
    ("nllb_lun2en", "keithtwesigye/lunyoro-nllb_lun2en"),
]

for folder, repo_id in models:
    local_dir = MODEL_DIR / folder
    if not local_dir.exists():
        print(f"SKIP {folder} — folder not found")
        continue
    print(f"\nUploading {folder} -> {repo_id} ...")
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload fine-tuned model weights",
        ignore_patterns=["*.cache", ".cache/**", "__pycache__/**"],
    )
    print(f"Done: {repo_id}")

print("\nAll models pushed to HuggingFace.")
