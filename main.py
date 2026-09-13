import sys
import os

# so Python finds sibling folders
sys.path.append(os.path.dirname(__file__))

from ingestion.pdf_loader import load_pdf
from ingestion.yt_loader import load_youtube
from ingestion.github_loader import load_github
from pipeline.chunker import chunk_text
from pipeline.embedder import embed_chunks, embed_query
from pipeline.vector_store import VectorStore
from llm.provider import ask

STORE_DIR = "saved_store"


def ingest(sources: list[str], store: VectorStore):
    """Load, chunk, embed, and store all provided sources."""
    all_docs = []

    for source in sources:
        if source.endswith(".pdf"):
            all_docs.extend(load_pdf(source))
        elif "youtube.com" in source or "youtu.be" in source:
            all_docs.extend(load_youtube(source))
        elif "github.com" in source:
            all_docs.extend(load_github(source))
        else:
            print(f"[Main] Unknown source type, skipping: {source}")

    if not all_docs:
        print("[Main] No documents loaded.")
        return

    chunks = chunk_text(all_docs)
    embedded_chunks = embed_chunks(chunks)
    store.add(embedded_chunks)
    store.save(STORE_DIR)
    print(f"\n✅ Ingestion complete. {store.index.ntotal} vectors stored.\n")


def query_loop(store: VectorStore):
    """Interactive terminal Q&A loop."""
    print("\n🔍 RAG Q&A ready! Type your question (or 'exit' to quit)\n")

    while True:
        query = input("You: ").strip()

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("Bye!")
            break

        q_embedding = embed_query(query)
        top_chunks = store.search(q_embedding, top_k=5)

        if not top_chunks:
            print("Assistant: No relevant context found.\n")
            continue

        answer = ask(query, top_chunks)
        print(f"\nAssistant: {answer}\n")
        print("-" * 60)


def main():
    store = VectorStore(dimension=384)

    # 1. Load existing knowledge if available
    if os.path.exists(os.path.join(STORE_DIR, "vector_store.index")):
        print("[Main] Loading existing vector store...")
        store.load(STORE_DIR)
    else:
        print("[Main] No existing vector store found.")

    # 2. Ask to add new sources
    choice = input("\nWould you like to add new sources to your knowledge base? (y/n): ").strip().lower()
    if choice == 'y':
        print("\nEnter source paths or URLs (one per line, or press Enter on an empty line to finish):")
        sources = []
        while True:
            line = input("> ").strip().strip('"').strip("'")
            if not line:
                break
            sources.append(line)

        if sources:
            print(f"[Main] Ingesting {len(sources)} source(s)...")
            ingest(sources, store)
        else:
            print("[Main] No sources provided, skipping ingestion.")

    # 3. Validate and Query
    if store.index.ntotal > 0:
        query_loop(store)
    else:
        print("\n[Main] Error: The vector store is empty. You must add at least one source to use the RAG system.")
        print("Please run the program again and provide valid source paths or URLs.")


if __name__ == "__main__":
    main()
