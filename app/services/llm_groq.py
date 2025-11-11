import os
from groq import Groq

_GROQ_CLIENT = None


def get_groq_client() -> Groq:
    global _GROQ_CLIENT
    if _GROQ_CLIENT is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Falta GROQ_API_KEY en el entorno")
        _GROQ_CLIENT = Groq(api_key=api_key)
    return _GROQ_CLIENT


def chat_complete(
    messages,
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    client = get_groq_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()
