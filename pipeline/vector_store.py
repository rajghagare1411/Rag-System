import faiss
import numpy as np
import pickle
import os

INDEX_FILE = "vector_store.index"
METADATA_FILE = "vector_store_metadata.pkl"


class VectorStore:
    def __init__(self, dimension: int = 384):
        """
        dimension: must match your embedding model's output size
                   all-MiniLM-L6-v2 → 384
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance search
        self.metadata = []  # stores chunk dicts (without embeddings)

    def add(self, chunks: list[dict]):
        """Add embedded chunks to the FAISS index."""
        embeddings = np.array([c["embedding"] for c in chunks]).astype("float32")
        self.index.add(embeddings)

        # store metadata without the embedding array (saves memory)
        for chunk in chunks:
            meta = {k: v for k, v in chunk.items() if k != "embedding"}
            self.metadata.append(meta)

        print(f"[VectorStore] Added {len(chunks)} chunks. Total: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Search for top_k most similar chunks.
        Returns list of chunk dicts with an added 'score' field.
        """
        query = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            result = dict(self.metadata[idx])
            result["score"] = float(dist)
            results.append(result)

        return results

    def save(self, directory: str = "."):
        """Save FAISS index and metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, INDEX_FILE))
        with open(os.path.join(directory, METADATA_FILE), "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[VectorStore] Saved to '{directory}'")

    def load(self, directory: str = "."):
        """Load FAISS index and metadata from disk."""
        self.index = faiss.read_index(os.path.join(directory, INDEX_FILE))
        with open(os.path.join(directory, METADATA_FILE), "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[VectorStore] Loaded {self.index.ntotal} vectors from '{directory}'")


# ---------- quick test ----------
if __name__ == "__main__":
    from embedder import embed_chunks, embed_query

    sample_chunks = [
        {"source": "doc1.pdf", "content": "FAISS is a library for efficient similarity search."},
        {"source": "doc2.pdf", "content": "Python is widely used in data science."},
        {"source": "doc3.pdf", "content": "Neural networks power modern AI applications."},
    ]

    embedded = embed_chunks(sample_chunks)

    store = VectorStore(dimension=384)
    store.add(embedded)

    q = embed_query("How does similarity search work?")
    results = store.search(q, top_k=2)

    print("\nTop results:")
    for r in results:
        print(f"  [{r['score']:.4f}] {r['content']} (source: {r['source']})")