import os
from dotenv import load_dotenv

load_dotenv()

def ask(query, context_chunks):
    """
    Provider wrapper that selects the LLM based on the LLM_PROVIDER env var.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "nvidia":
        try:
            from llm.nvidia_client import ask as nvidia_ask
            return nvidia_ask(query, context_chunks)
        except ImportError as e:
            return f"[Provider Error] Failed to load NVIDIA client: {e}"
    elif provider == "openrouter":
        try:
            from llm.openrouter_client import ask as openrouter_ask
            return openrouter_ask(query, context_chunks)
        except ImportError as e:
            return f"[Provider Error] Failed to load OpenRouter client: {e}"
    else:
        try:
            from llm.gemini_client import ask as gemini_ask
            return gemini_ask(query, context_chunks)
        except ImportError as e:
            return f"[Provider Error] Failed to load Gemini client: {e}"
