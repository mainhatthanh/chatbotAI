from rag.embedder import Embedder
from rag.query import content_tokens, normalize_text
from rag.vector_store import VectorStore

#Hybird search = Senmatic(VectorStore) + Keyword(.)
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
        query_tokens = content_tokens(query)
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
        query_tokens = content_tokens(query)
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
        description = metadata.get("normalized_description", "")
        question = metadata.get("normalized_question", "")
        answer = metadata.get("normalized_answer", "")
        text = doc.get("normalized_text") or normalize_text(doc.get("text", ""))
        title_tokens = set(title.split())
        author_tokens = set(author.split())
        category_tokens = set(category.split())
        description_tokens = set(description.split())
        question_tokens = set(question.split())
        answer_tokens = set(answer.split())
        text_tokens = set(text.split())

        score = 0.0

        if query_norm:
            if self._contains_phrase(title, query_norm):
                score += 3.5
            if self._contains_phrase(author, query_norm):
                score += 3.0
            if self._contains_phrase(category, query_norm):
                score += 2.0
            if self._contains_phrase(description, query_norm):
                score += 4.0
            if self._contains_phrase(question, query_norm):
                score += 3.5
            if self._contains_phrase(answer, query_norm):
                score += 1.5
            if self._contains_phrase(text, query_norm):
                score += 1.0

        for token in query_tokens:
            if token in title_tokens:
                score += 1.6
            if token in author_tokens:
                score += 1.4
            if token in category_tokens:
                score += 1.1
            if token in description_tokens:
                score += 1.3
            if token in question_tokens:
                score += 1.7
            if token in answer_tokens:
                score += 0.8
            if token in text_tokens:
                score += 0.4

        return score

    def _contains_phrase(self, field_value: str, phrase: str):
        if not field_value or not phrase:
            return False
        return f" {phrase} " in f" {field_value} "

    def _combine_scores(self, semantic_score, keyword_score, doc):
        source_bonus = 0.1 if doc.get("source") == "books" else 0.0
        return (semantic_score * 0.65) + (keyword_score * 0.35) + source_bonus
