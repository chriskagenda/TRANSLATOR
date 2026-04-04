"""
Build translation index using sentence embeddings.
Since dataset is small (~6k pairs), we use semantic similarity search
with sentence-transformers instead of training a seq2seq model from scratch.
This gives much better results for low-resource languages.
"""
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from preprocess import load_sentence_pairs, load_dictionary

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
INDEX_PATH = os.path.join(os.path.dirname(__file__), "model", "translation_index.pkl")


def build_index():
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    print("Loading data...")
    pairs = load_sentence_pairs()
    dictionary = load_dictionary()

    print(f"Encoding {len(pairs)} sentence pairs with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    english_sentences = pairs["english"].tolist()
    lunyoro_sentences = pairs["lunyoro"].tolist()

    print(f"Encoding {len(pairs)} English sentences...")
    embeddings = model.encode(english_sentences, show_progress_bar=True, batch_size=64)

    print(f"Encoding {len(pairs)} Lunyoro sentences...")
    lunyoro_embeddings = model.encode(lunyoro_sentences, show_progress_bar=True, batch_size=64)

    index = {
        "model_name": MODEL_NAME,
        "english_sentences": english_sentences,
        "lunyoro_sentences": lunyoro_sentences,
        "embeddings": embeddings,
        "lunyoro_embeddings": lunyoro_embeddings,
        "dictionary": dictionary.to_dict(orient="records"),
    }

    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)

    print(f"Index saved to {INDEX_PATH}")
    print(f"Total sentence pairs indexed: {len(english_sentences)}")
    print(f"Total dictionary entries: {len(dictionary)}")


if __name__ == "__main__":
    build_index()
