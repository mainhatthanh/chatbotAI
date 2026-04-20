import pickle
from sklearn.neighbors import NearestNeighbors


class VectorStore:
    """Wrapper nho quanh NearestNeighbors de search embedding bang cosine distance."""

    def __init__(self):
        self.model = NearestNeighbors(metric="cosine")
        self.documents = []
        self.embeddings = None

    def add_documents(self, embeddings, documents):
        """Fit index moi moi khi pipeline load lai data."""
        self.embeddings = embeddings
        self.documents = documents
        self.model.fit(embeddings)

    def search(self, query_embedding, top_k=3):
        if not self.documents:
            return []

        k = min(top_k, len(self.documents))

        distances, indices = self.model.kneighbors(
            [query_embedding], n_neighbors=k
        )

        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "document": self.documents[idx],
                "distance": float(distances[0][i])
            })
        return results

    def save(self, model_path, docs_path):
        """Luu index neu sau nay muon preload vector store ra file."""
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)

    @classmethod
    def load(cls, model_path, docs_path):
        """Load index da save truoc do."""
        store = cls()

        with open(model_path, "rb") as f:
            store.model = pickle.load(f)

        with open(docs_path, "rb") as f:
            store.documents = pickle.load(f)

        return store
