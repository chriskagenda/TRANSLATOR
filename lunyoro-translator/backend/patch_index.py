"""
One-time script: adds lunyoro_embeddings to the existing translation_index.pkl
so lun→en dictionary search is instant on startup.
"""
import pickle, os
from sentence_transformers import SentenceTransformer

INDEX_PATH = os.path.join(os.path.dirname(__file__), "model", "translation_index.pkl")
SEM_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model", "sem_model")

print("Loading index...")
with open(INDEX_PATH, "rb") as f:
    index = pickle.load(f)

if "lunyoro_embeddings" in index:
    print("lunyoro_embeddings already present — nothing to do.")
else:
    print(f"Encoding {len(index['lunyoro_sentences'])} Lunyoro sentences...")
    model = SentenceTransformer(SEM_MODEL_DIR)
    index["lunyoro_embeddings"] = model.encode(
        index["lunyoro_sentences"], show_progress_bar=True, batch_size=64
    )
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
    print("Done — index updated.")
