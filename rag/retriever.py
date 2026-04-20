from rag.embedder import Embedder
from rag.query import content_tokens, normalize_text
from rag.query.normalize import similar_token_score
from rag.vector_store import VectorStore

DEFAULT_TOP_K = 3
DEFAULT_SEMANTIC_K = 8
FUZZY_TITLE_THRESHOLD = 0.84

PHRASE_WEIGHTS = {
    "title": 3.5,
    "author": 3.0,
    "category": 2.0,
    "description": 4.0,
    "question": 3.5,
    "answer": 1.5,
    "text": 1.0,
}

TOKEN_WEIGHTS = {
    "title": 2.8,
    "author": 1.4,
    "category": 1.1,
    "description": 1.3,
    "question": 1.7,
    "answer": 0.8,
    "text": 0.4,
}


class Retriever:
    """Hybrid retriever: semantic search + keyword search tren cung tap document."""

    def __init__(self, store: VectorStore, documents=None, embedder=None):
        self.embedder = embedder or Embedder()
        self.store = store
        self.documents = documents or []

    def retrieve(self, query: str, top_k=DEFAULT_TOP_K):
        return self.retrieve_hybrid(query, top_k=top_k)

    def retrieve_hybrid(self, query: str, top_k=DEFAULT_TOP_K, semantic_k=DEFAULT_SEMANTIC_K):
        """Lay ung vien bang vector search, sau do bo sung diem keyword."""
        semantic_k = max(top_k, semantic_k)
        query_norm = normalize_text(query)
        query_tokens = content_tokens(query)

        scored = self._semantic_candidates(query, semantic_k)
        self._merge_keyword_scores(scored, query_norm, query_tokens)
        self._refresh_hybrid_scores(scored)

        return self._rank(scored)[:top_k]

    def keyword_search_books(self, query: str, top_k=DEFAULT_TOP_K):
        """Chi keyword search sach; huu ich khi can loc theo title/category ro rang."""
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

        return sorted(matched, key=lambda item: item["hybrid_score"], reverse=True)[:top_k]

    def _semantic_candidates(self, query, semantic_k):
        """Chuyen cosine distance thanh semantic_score: diem cao hon la gan hon."""
        query_embedding = self.embedder.encode_query(query)
        scored = {}

        for item in self.store.search(query_embedding, top_k=semantic_k):
            doc = item["document"]
            distance = float(item["distance"])
            score = max(0.0, 1.0 - distance)
            scored[doc["id"]] = {
                "document": doc,
                "distance": distance,
                "semantic_score": score,
                "keyword_score": 0.0,
                "hybrid_score": score,
            }

        return scored

    def _merge_keyword_scores(self, scored, query_norm, query_tokens):
        """Them cac document match keyword, ke ca khi vector search khong lay ra."""
        for doc in self.documents:
            keyword_score = self._keyword_score(query_norm, query_tokens, doc)
            if keyword_score <= 0:
                continue

            current = scored.setdefault(doc["id"], self._empty_score(doc))
            current["keyword_score"] = max(current["keyword_score"], keyword_score)

    def _refresh_hybrid_scores(self, scored):
        for item in scored.values():
            item["hybrid_score"] = self._combine_scores(
                item["semantic_score"],
                item["keyword_score"],
                item["document"],
            )

    def _keyword_score(self, query_norm: str, query_tokens, doc):
        fields = self._normalized_fields(doc)
        field_tokens = {
            name: set(value.split())
            for name, value in fields.items()
        }

        score = self._phrase_score(query_norm, fields)
        score += self._token_score(query_tokens, field_tokens)
        return score

    def _phrase_score(self, query_norm, fields):
        """Phrase match giup bat cac query trung nguyen cum tu voi title/author."""
        if not query_norm:
            return 0.0

        return sum(
            PHRASE_WEIGHTS[name]
            for name, value in fields.items()
            if self._contains_phrase(value, query_norm)
        )

    def _token_score(self, query_tokens, field_tokens):
        """Token match giup bat query ngan nhu 'conan', 'doraemon'."""
        score = 0.0
        title_tokens = field_tokens["title"]

        for token in query_tokens:
            for field_name, tokens in field_tokens.items():
                if token in tokens:
                    score += TOKEN_WEIGHTS[field_name]

            # Fuzzy title match sua loi go nhe: dordaemon -> doraemon.
            if len(token) >= 5 and token not in title_tokens:
                best_title_similarity = max(
                    (similar_token_score(token, title_token) for title_token in title_tokens),
                    default=0.0,
                )
                if best_title_similarity >= FUZZY_TITLE_THRESHOLD:
                    score += 2.4

        return score

    def _normalized_fields(self, doc):
        metadata = doc.get("metadata", {})
        text = doc.get("normalized_text") or normalize_text(doc.get("text", ""))
        return {
            "title": metadata.get("normalized_title", ""),
            "author": metadata.get("normalized_author", ""),
            "category": metadata.get("normalized_category", ""),
            "description": metadata.get("normalized_description", ""),
            "question": metadata.get("normalized_question", ""),
            "answer": metadata.get("normalized_answer", ""),
            "text": text,
        }

    def _empty_score(self, doc):
        return {
            "document": doc,
            "distance": 1.0,
            "semantic_score": 0.0,
            "keyword_score": 0.0,
            "hybrid_score": 0.0,
        }

    def _rank(self, scored):
        return sorted(
            scored.values(),
            key=lambda item: (item["hybrid_score"], -item["distance"]),
            reverse=True,
        )

    def _contains_phrase(self, field_value: str, phrase: str):
        if not field_value or not phrase:
            return False
        return f" {phrase} " in f" {field_value} "

    def _combine_scores(self, semantic_score, keyword_score, doc):
        # Sach duoc cong nhe vi chatbot chinh la tro ly ban sach.
        source_bonus = 0.1 if doc.get("source") == "books" else 0.0
        return (semantic_score * 0.65) + (keyword_score * 0.35) + source_bonus
