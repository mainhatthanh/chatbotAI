from rag.embedder import Embedder
from rag.vector_store import VectorStore

class Retriever:
    def __init__(self, store: VectorStore, documents=None):
        self.embedder = Embedder()
        self.store = store
        self.documents = documents or []

    def retrieve(self, query: str, top_k=3):
        query_embedding = self.embedder.encode_query(query)
        results = self.store.search(query_embedding, top_k=top_k)
        return results

    def keyword_search_books(self, query: str, top_k=3):
        query_lower = query.lower()
        matched = []

        for doc in self.documents:
            if doc.get("source") != "books":
                continue

            metadata = doc.get("metadata", {})
            score = 0

            if metadata.get("title", "") in query_lower or query_lower in metadata.get("title", ""):
                score += 5
            if metadata.get("author", "") in query_lower or query_lower in metadata.get("author", ""):
                score += 4
            if metadata.get("category", "") in query_lower or query_lower in metadata.get("category", ""):
                score += 3

            # match từng từ
            for word in query_lower.split():
                if word in metadata.get("title", ""):
                    score += 2
                if word in metadata.get("author", ""):
                    score += 1
                if word in metadata.get("category", ""):
                    score += 1
                if word in doc.get("text", "").lower():
                    score += 0.5

            if score > 0:
                matched.append({
                    "document": doc,
                    "distance": 1 / (score + 1)
                })

        matched = sorted(matched, key=lambda x: x["distance"])
        return matched[:top_k]