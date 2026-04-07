from sentence_transformers import SentenceTransformer
from core.config import EMBEDDING_MODEL

_model = None

def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def embed(texts: list[str]) -> list:
    return get_embedding_model().encode(texts).tolist()