import fitz
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_chunks(pdf_path: str, chunk_size=800, chunk_overlap=100):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(full_text)
    paper_id = os.path.basename(pdf_path).replace(".pdf", "")[:50]
    title = doc.metadata.get("title", paper_id)
    return paper_id, title, chunks