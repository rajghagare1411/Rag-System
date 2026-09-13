import fitz  # PyMuPDF


def load_pdf(file_path: str) -> list[dict]:
    """
    Extract text from a PDF file, page by page.
    Returns a list of dicts with page number and text content.
    """
    doc = fitz.open(file_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        if text.strip():  # skip empty pages
            pages.append({
                "source": file_path,
                "page": page_num + 1,
                "content": text.strip()
            })

    doc.close()
    print(f"[PDF] Loaded {len(pages)} pages from '{file_path}'")
    return pages


# ---------- quick test ----------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_loader.py <path_to_pdf>")
        sys.exit(1)

    pages = load_pdf(sys.argv[1])
    for p in pages[:3]:  # preview first 3 pages
        print(f"\n--- Page {p['page']} ---")
        print(p["content"][:300])