from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.text_utils import normalize_text, tokenize


class Retriever:
    def __init__(self, store: VectorStore, documents=None):
        self.embedder = Embedder()
        self.store = store
        self.documents = documents or []

    def retrieve(self, query: str, top_k=3):
        return self.retrieve_hybrid(query, top_k=top_k)

    def retrieve_hybrid(self, query: str, top_k=3, semantic_k=8):
        semantic_k = max(top_k, semantic_k)
        query_embedding = self.embedder.encode_query(query)
        semantic_results = self.store.search(query_embedding, top_k=semantic_k)
        query_norm = normalize_text(query)
        query_tokens = tokenize(query)
        scored = {}

        for item in semantic_results:
            doc = item["document"]
            score = max(0.0, 1.0 - float(item["distance"]))
            scored[doc["id"]] = {
                "document": doc,
                "distance": float(item["distance"]),
                "semantic_score": score,
                "keyword_score": 0.0,
                "hybrid_score": score,
            }

        for doc in self.documents:
            keyword_score = self._keyword_score(query_norm, query_tokens, doc)
            if keyword_score <= 0:
                continue

            current = scored.get(doc["id"])
            if current is None:
                current = {
                    "document": doc,
                    "distance": 1.0,
                    "semantic_score": 0.0,
                    "keyword_score": 0.0,
                    "hybrid_score": 0.0,
                }
                scored[doc["id"]] = current

            current["keyword_score"] = max(current["keyword_score"], keyword_score)
            current["hybrid_score"] = self._combine_scores(
                current["semantic_score"],
                current["keyword_score"],
                doc,
            )

        for item in scored.values():
            item["hybrid_score"] = self._combine_scores(
                item["semantic_score"],
                item["keyword_score"],
                item["document"],
            )

        ranked = sorted(
            scored.values(),
            key=lambda item: (item["hybrid_score"], -item["distance"]),
            reverse=True,
        )
        return ranked[:top_k]

    def keyword_search_books(self, query: str, top_k=3):
        query_norm = normalize_text(query)
        query_tokens = tokenize(query)
        matched = []

        for doc in self.documents:
            if doc.get("source") != "books":
                continue

            score = self._keyword_score(query_norm, query_tokens, doc)
            if score <= 0:
                continue

            matched.append({
                "document": doc,
                "distance": 1 / (score + 1),
                "semantic_score": 0.0,
                "keyword_score": score,
                "hybrid_score": score,
            })

        matched.sort(key=lambda item: item["hybrid_score"], reverse=True)
        return matched[:top_k]

    def _keyword_score(self, query_norm: str, query_tokens, doc):
        metadata = doc.get("metadata", {})
        title = metadata.get("normalized_title", "")
        author = metadata.get("normalized_author", "")
        category = metadata.get("normalized_category", "")
        question = metadata.get("normalized_question", "")
        answer = metadata.get("normalized_answer", "")
        text = doc.get("normalized_text") or normalize_text(doc.get("text", ""))

        score = 0.0

        if query_norm:
            if title and query_norm in title:
                score += 3.5
            if author and query_norm in author:
                score += 3.0
            if category and query_norm in category:
                score += 2.0
            if question and query_norm in question:
                score += 3.5
            if answer and query_norm in answer:
                score += 1.5
            if query_norm in text:
                score += 1.0

        for token in query_tokens:
            if title and token in title:
                score += 1.6
            if author and token in author:
                score += 1.4
            if category and token in category:
                score += 1.1
            if question and token in question:
                score += 1.7
            if answer and token in answer:
                score += 0.8
            if token in text:
                score += 0.4

        return score

    def _combine_scores(self, semantic_score, keyword_score, doc):
        source_bonus = 0.1 if doc.get("source") == "books" else 0.0
        return (semantic_score * 0.65) + (keyword_score * 0.35) + source_bonus
