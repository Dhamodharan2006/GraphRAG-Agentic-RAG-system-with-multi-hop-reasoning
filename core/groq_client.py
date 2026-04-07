from groq import Groq
from core.config import GROQ_API_KEY, GROQ_MODEL

_client = None

def get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client

def groq_chat(prompt: str, system: str = "You are a helpful research assistant.", max_tokens: int = 1024) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content