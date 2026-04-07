import chromadb
from chromadb.config import Settings
from core.config import CHROMA_PATH

_client = None
_collection = None

def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _client.get_or_create_collection(
            name="papers",
            metadata={"hnsw:space": "cosine"}
        )
    return _collection

def vector_search(query: str, top_k: int = 5, paper_filter: list = None):
    col = get_collection()
    where = {"paper_id": {"$in": paper_filter}} if paper_filter else None
    results = col.query(
        query_texts=[query],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"]
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    return [{"text": d, "metadata": m, "score": 1 - s}
            for d, m, s in zip(docs, metas, dists)]