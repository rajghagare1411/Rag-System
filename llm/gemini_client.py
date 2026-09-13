import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Missing GEMINI_API_KEY. Add it to your .env file or environment before running this script.")

client = genai.Client(api_key=api_key)


def build_prompt(query: str, context_chunks: list[dict]) -> str:
    """Build a prompt using retrieved chunks as context."""
    context_parts = []

    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("source", "unknown")
        # show page for PDFs, timestamp for YouTube, file for GitHub
        detail = (
            f"page {chunk['page']}" if "page" in chunk else
            f"timestamp {chunk['timestamp']}s" if "timestamp" in chunk else
            f"file: {chunk.get('file', '')}"
        )
        context_parts.append(f"[Source {i} — {source} | {detail}]\n{chunk['content']}")

    context_text = "\n\n".join(context_parts)

    return f"""You are a helpful assistant. Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I couldn't find that in the provided sources."
Always mention which source(s) you used.

--- CONTEXT ---
{context_text}
--- END CONTEXT ---

Question: {query}

Answer:"""


def ask(query: str, context_chunks: list[dict]) -> str:
    """
    Send query + retrieved context to Gemini and return the answer.
    Includes automatic retries for transient server errors (503, 429).
    """
    prompt = build_prompt(query, context_chunks)
    max_retries = 3
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            err_msg = str(e)
            # Retry on 503 (Unavailable/Overloaded) or 429 (Resource Exhausted/Quota)
            if "503" in err_msg or "429" in err_msg:
                if attempt < max_retries - 1:
                    print(f"[Gemini] Server overloaded or quota reached. Retrying in {retry_delay}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
            return f"[Gemini Error] {e}"


# ---------- quick test ----------
if __name__ == "__main__":
    sample_chunks = [
        {
            "source": "ml_notes.pdf",
            "page": 3,
            "content": "Gradient descent is an optimization algorithm used to minimize the loss function in machine learning models by iteratively updating weights."
        },
        {
            "source": "ml_notes.pdf",
            "page": 5,
            "content": "The learning rate controls how large each step is during gradient descent. Too high and it overshoots; too low and training is very slow."
        }
    ]

    answer = ask("What is gradient descent and how does learning rate affect it?", sample_chunks)
    print(f"\nAnswer:\n{answer}")
