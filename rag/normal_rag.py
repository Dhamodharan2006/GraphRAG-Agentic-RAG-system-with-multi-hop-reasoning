from agents.tools import tool_vector_search
from core.groq_client import groq_chat


def normal_rag_answer(question: str) -> str:
    results = tool_vector_search(question, top_k=4)
    if not results:
        return "No relevant documents found. Please upload PDFs first."

    context = "\n\n".join(
        [f"[{i+1}] From {r['paper_id']}: {r['text'][:500]}"
         for i, r in enumerate(results)]
    )
    prompt = f"""Answer the question using ONLY the context below. Cite sources using [number] format.

Context:
{context}

Question: {question}
Answer:"""
    return groq_chat(prompt)