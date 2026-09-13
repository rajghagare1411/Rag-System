import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

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
    Send query + retrieved context to NVIDIA/DeepSeek and return the answer.
    """
    prompt = build_prompt(query, context_chunks)

    try:
        completion = client.chat.completions.create(
            model="deepseek-ai/deepseek-v4-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            top_p=0.95,
            max_tokens=16384,
            extra_body={"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}},
            stream=False
        )

        content = completion.choices[0].message.content
        # DeepSeek reasoning content is usually in 'reasoning' or 'reasoning_content'
        reasoning = getattr(completion.choices[0].message, "reasoning", None) or \
                    getattr(completion.choices[0].message, "reasoning_content", None)

        if reasoning:
            return f"Thinking:\n{reasoning}\n\nAnswer:\n{content}"

        return content
    except Exception as e:
        return f"[NVIDIA Error] {e}"


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

    print("Testing NVIDIA client...")
    answer = ask("What is gradient descent and how does learning rate affect it?", sample_chunks)
    print(f"\nAnswer:\n{answer}")
