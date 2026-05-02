import pickle

import faiss
import numpy as np


class VectorStore:
    """FAISS vector store dung cosine search tren embeddings da normalize."""

    def __init__(self):
        self.index = None
        self.documents = []
        self.embeddings = None

    def add_documents(self, embeddings, documents):
        """Build FAISS index moi moi khi pipeline load lai data."""
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")

        self.embeddings = embeddings
        self.documents = documents
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query_embedding, top_k=3):
        if not self.documents or self.index is None:
            return []

        k = min(top_k, len(self.documents))
        query_embedding = np.asarray([query_embedding], dtype="float32")

        scores, indices = self.index.search(query_embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0:
                continue
            cosine_score = float(scores[0][i])
            results.append({
                "document": self.documents[idx],
                "distance": 1.0 - cosine_score,
            })
        return results

    def save(self, index_path, docs_path):
        """Luu FAISS index va documents ra disk de app co the preload nhanh."""
        if self.index is None:
            raise ValueError("Cannot save an empty vector store")

        faiss.write_index(self.index, str(index_path))
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)

    @classmethod
    def load(cls, index_path, docs_path):
        """Load FAISS index va documents da build truoc do."""
        store = cls()
        store.index = faiss.read_index(str(index_path))
        with open(docs_path, "rb") as f:
            store.documents = pickle.load(f)
        return store
