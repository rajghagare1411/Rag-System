# 🚀 RAG Universal

A lightweight Retrieval-Augmented Generation (RAG) system that ingests documents from PDFs, YouTube videos, and GitHub repositories, converts them into embeddings, stores them in a FAISS vector database, and answers questions using an LLM provider.

## ✨ Features

- PDF ingestion using PyMuPDF
- YouTube transcript extraction using `youtube-transcript-api`
- GitHub repository source ingestion for text-based project files
- Text chunking and embedding generation with `sentence-transformers`
- Vector search with FAISS
- Question answering through Gemini, NVIDIA, or OpenRouter
- Local persistence of the vector index in the `saved_store` directory

## 🗂️ Project structure

```text
Rag_Universal/
├── .env.example
├── .gitignore
├── main.py
├── README.md
├── requirements.txt
├── ingestion/
│   ├── github_loader.py
│   ├── pdf_loader.py
│   └── yt_loader.py
├── llm/
│   ├── gemini_client.py
│   ├── nvidia_client.py
│   ├── openrouter_client.py
│   └── provider.py
├── pipeline/
│   ├── chunker.py
│   ├── embedder.py
│   └── vector_store.py
├── saved_store/
│   └── vector_store.index
└── .venv/
```

## ⚙️ How it works

1. A source is added in `main.py`.
2. The relevant loader reads content from:
   - PDF files
   - YouTube links
   - GitHub repository URLs
3. The text is split into chunks in `pipeline/chunker.py`.
4. Embeddings are generated in `pipeline/embedder.py` using the `all-MiniLM-L6-v2` model.
5. Chunks are stored in a FAISS vector index (`pipeline/vector_store.py`).
6. A user question is embedded and matched against the stored vector store.
7. Relevant chunks are passed to the configured LLM provider for a grounded answer.

## 🧰 Setup

### 1) Create a virtual environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure environment variables

Copy the example file and add your API keys:

```bash
copy .env.example .env
```

Then edit `.env` with the needed values.

Example:

```env
GEMINI_API_KEY=your_gemini_key_here
LLM_PROVIDER=gemini
```

Supported provider options:

- `gemini` — uses `GEMINI_API_KEY`
- `nvidia` — uses `NVIDIA_API_KEY`
- `openrouter` — uses `OPENROUTER_API_KEY`

## ▶️ Running the app

```bash
python main.py
```

When the program starts:

- it checks whether a vector store already exists in `saved_store`
- it asks whether you want to add new sources
- you can enter one or more source paths or URLs
- after ingestion, you can ask questions in the terminal

## 📚 Supported sources

### PDF

```text
path/to/file.pdf
```

### YouTube video

```text
https://www.youtube.com/watch?v=VIDEO_ID
https://youtu.be/VIDEO_ID
```

### GitHub repository

```text
https://github.com/owner/repository
```

## 📝 Notes

- The vector store is saved in `saved_store/` to keep the RAG context persistent across runs.
- `saved_store` and `.env` are excluded from Git in `.gitignore`.
- Model downloads may happen on first run, depending on the embedding model and provider setup.

## ✅ Requirements

The project depends on:

- `faiss-cpu`
- `sentence-transformers`
- `numpy`
- `PyMuPDF`
- `yt-dlp`
- `youtube-transcript-api`
- `GitPython`
- `google-genai`
- `python-dotenv`
- `openai`

## 📜 License

This project is for educational and personal use unless otherwise specified.

## 👨‍💻 Author

Designed and built by Raj Ghagare.

> Built with Python, FAISS, and modern LLM integrations for intelligent document retrieval.
