"""
FileSense — One-Time Model Setup
=================================
Run this script ONCE, on a machine with internet access, before using FileSense.

What it does:
    Downloads the embedding model from HuggingFace into the local `models/` folder
    inside this project. After this runs successfully, FileSense is fully offline
    forever. This script never needs to run again unless you delete the models/ folder.

How to run:
    python setup_models.py

You should see a success message at the end. After that, start the app normally
with `streamlit run app.py`.
"""

import sys
from pathlib import Path

# Make sure config is importable even when run from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from sentence_transformers import SentenceTransformer

def download_embedding_model():
    print("=" * 60)
    print("FileSense — Model Setup")
    print("=" * 60)
    print(f"\nModel       : {config.MODEL_NAME}")
    print(f"Destination : {config.MODEL_CACHE_DIR}")
    print("\nDownloading... (this only happens once)\n")

    try:
        # Explicitly download into the project's own models/ folder.
        # local_files_only is NOT set here — this is the one intentional download.
        model = SentenceTransformer(
            config.MODEL_NAME,
            cache_folder=str(config.MODEL_CACHE_DIR)
        )

        # Smoke test: make sure the model actually works before declaring success.
        test_embedding = model.encode("FileSense setup test.")
        if len(test_embedding) == 0:
            raise ValueError("Model loaded but produced empty embedding — something is wrong.")

        print("\n✅ Setup complete.")
        print(f"   Model is saved in: {config.MODEL_CACHE_DIR}")
        print("   FileSense is now fully offline. Run `streamlit run app.py` to start.\n")

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("\nCheck your internet connection and try again.")
        print("If the problem persists, verify the model name in config.py:")
        print(f"    MODEL_NAME = \"{config.MODEL_NAME}\"")
        sys.exit(1)


if __name__ == "__main__":
    download_embedding_model()