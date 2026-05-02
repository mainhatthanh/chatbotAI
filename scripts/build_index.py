#!/usr/bin/env python3
"""Build vectorstore/faiss.index va vectorstore/documents.pkl."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag.data_loader import load_all_documents
from rag.embedder import Embedder
from rag.pipeline import DOCUMENTS_PATH, FAISS_INDEX_PATH, VECTORSTORE_DIR
from rag.vector_store import VectorStore

BOOKS_PATH = "data/books.csv"
FAQ_PATH = "data/faq.json"


def build_index():
    all_docs = load_all_documents(BOOKS_PATH, FAQ_PATH)
    documents = all_docs["books"] + all_docs["faq"]
    texts = [doc["text"] for doc in documents]

    embedder = Embedder()
    embeddings = embedder.encode_text(texts)

    store = VectorStore()
    store.add_documents(embeddings, documents)

    VECTORSTORE_DIR.mkdir(exist_ok=True)
    store.save(FAISS_INDEX_PATH, DOCUMENTS_PATH)
    print(f"Saved {len(documents)} documents to {FAISS_INDEX_PATH} and {DOCUMENTS_PATH}")


if __name__ == "__main__":
    build_index()
