"""
Downloads all fine-tuned Runyoro-Rutooro models from HuggingFace.
Run this once after cloning the repo:

    python download_models.py

Models pulled:
    keithtwesigye/lunyoro-en2lun      → model/en2lun/
    keithtwesigye/lunyoro-lun2en      → model/lun2en/
    keithtwesigye/lunyoro-nllb_en2lun → model/nllb_en2lun/
    keithtwesigye/lunyoro-nllb_lun2en → model/nllb_lun2en/
"""
import os
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "model"

HF_MODELS = {
    "en2lun":      "keithtwesigye/lunyoro-en2lun",
    "lun2en":      "keithtwesigye/lunyoro-lun2en",
    "nllb_en2lun": "keithtwesigye/lunyoro-nllb_en2lun",
    "nllb_lun2en": "keithtwesigye/lunyoro-nllb_lun2en",
}


def download_all(force: bool = False):
    from huggingface_hub import snapshot_download

    for local_name, repo_id in HF_MODELS.items():
        dest = MODEL_DIR / local_name
        if dest.exists() and not force:
            # Check if model weights are present
            has_weights = any(dest.glob("*.safetensors")) or any(dest.glob("*.bin"))
            if has_weights:
                print(f"  ✓ {local_name} already exists — skipping (use --force to re-download)")
                continue

        print(f"  ↓ Downloading {repo_id} → model/{local_name}/")
        dest.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        print(f"  ✓ {local_name} downloaded")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-download even if model exists")
    args = parser.parse_args()

    print("=== Downloading Runyoro-Rutooro models from HuggingFace ===")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install huggingface_hub")
        from huggingface_hub import snapshot_download

    download_all(force=args.force)
    print("\nAll models ready.")
