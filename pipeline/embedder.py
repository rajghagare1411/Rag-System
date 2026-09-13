from sentence_transformers import SentenceTransformer
import numpy as np

# Load once at module level (downloads on first run, ~90MB)
MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[Embedder] Loading model '{MODEL_NAME}'...")
        _model = SentenceTransformer(MODEL_NAME)
        print(f"[Embedder] Model loaded.")
    return _model


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add an 'embedding' (numpy array) to each chunk dict.
    Returns the same list with embeddings attached.
    """
    model = get_model()
    texts = [c["content"] for c in chunks]

    print(f"[Embedder] Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    print(f"[Embedder] Done. Embedding shape: {embeddings[0].shape}")
    return chunks


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string for similarity search."""
    model = get_model()
    return model.encode(query, convert_to_numpy=True)


# ---------- quick test ----------
if __name__ == "__main__":
    sample_chunks = [
        {"source": "test.pdf", "content": "Machine learning is a subset of AI."},
        {"source": "test.pdf", "content": "Python is a popular programming language."},
    ]

    embedded = embed_chunks(sample_chunks)
    print(f"\nEmbedding for first chunk: {embedded[0]['embedding'][:5]}...")  # first 5 values

    q_embedding = embed_query("What is machine learning?")
    print(f"\nQuery embedding: {q_embedding[:5]}...")