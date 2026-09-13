def chunk_text(documents: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Split document text into overlapping chunks.

    Args:
        documents : list of dicts with at least a 'content' key
        chunk_size: max number of characters per chunk
        overlap   : number of characters to overlap between chunks

    Returns:
        list of chunk dicts with content + metadata
    """
    chunks = []

    for doc in documents:
        text = doc["content"]
        metadata = {k: v for k, v in doc.items() if k != "content"}

        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    **metadata,
                    "chunk_index": chunk_index,
                    "content": chunk_text
                })
                chunk_index += 1

            start += chunk_size - overlap  # slide window with overlap

    print(f"[Chunker] Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


# ---------- quick test ----------
if __name__ == "__main__":
    sample_docs = [
        {
            "source": "test.pdf",
            "page": 1,
            "content": "This is a long document. " * 100
        }
    ]

    chunks = chunk_text(sample_docs, chunk_size=200, overlap=30)
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"\nFirst chunk:\n{chunks[0]['content']}")
    print(f"\nSecond chunk (should overlap):\n{chunks[1]['content']}")