from rag.data_loader import load_all_documents
from rag.embedder import Embedder
from rag.generator import Generator
from rag.retriever import Retriever
from rag.text_utils import detect_book_intent, detect_faq_intent, normalize_text, tokenize
from rag.vector_store import VectorStore


class RAGPipeline:
    def __init__(self, books_path: str, faq_path: str):
        all_docs = load_all_documents(books_path, faq_path)

        self.book_documents = all_docs["books"]
        self.faq_documents = all_docs["faq"]

        self.embedder = Embedder()

        book_texts = [doc["text"] for doc in self.book_documents]
        book_embeddings = self.embedder.encode_text(book_texts)
        self.books_store = VectorStore()
        self.books_store.add_documents(book_embeddings, self.book_documents)

        faq_texts = [doc["text"] for doc in self.faq_documents]
        faq_embeddings = self.embedder.encode_text(faq_texts)
        self.faq_store = VectorStore()
        self.faq_store.add_documents(faq_embeddings, self.faq_documents)

        self.books_retriever = Retriever(self.books_store, self.book_documents)
        self.faq_retriever = Retriever(self.faq_store, self.faq_documents)
        self.generator = Generator()

    def is_book_query(self, query: str):
        return detect_book_intent(query)

    def answer(self, query: str, top_k=3):
        candidate_k = max(top_k + 2, 5)
        book_intent = self.is_book_query(query)
        faq_intent = detect_faq_intent(query)

        faq_results = self.faq_retriever.retrieve_hybrid(query, top_k=candidate_k)

        if faq_intent and not book_intent:
            retrieved_docs = faq_results[:top_k]
            response = self.generator.generate(query, retrieved_docs)
            return response, retrieved_docs

        book_results = self.books_retriever.retrieve_hybrid(query, top_k=candidate_k)
        retrieved_docs = self._rerank_results(query, book_results, faq_results, top_k)
        response = self.generator.generate(query, retrieved_docs)
        return response, retrieved_docs

    def _rerank_results(self, query, book_results, faq_results, top_k):
        query_norm = normalize_text(query)
        query_tokens = set(tokenize(query))
        book_intent = self.is_book_query(query)
        faq_intent = detect_faq_intent(query)
        combined = []
        has_exact_book_match = False

        for item in book_results:
            score = item.get("hybrid_score", 0.0)
            if book_intent:
                score += 0.75

            metadata = item["document"].get("metadata", {})
            normalized_title = metadata.get("normalized_title", "")
            if normalized_title and normalized_title in query_norm:
                score += 2.2
                has_exact_book_match = True
            if metadata.get("normalized_author") and metadata["normalized_author"] in query_norm:
                score += 1.4
                has_exact_book_match = True
            if normalized_title:
                matched_title_tokens = [
                    token for token in normalized_title.split()
                    if len(token) > 2 and token in query_tokens and token != "tap"
                ]
                if matched_title_tokens:
                    score += 0.9 * len(matched_title_tokens)
                    has_exact_book_match = True

            ranked = dict(item)
            ranked["hybrid_score"] = score
            combined.append(ranked)

        for item in faq_results:
            score = item.get("hybrid_score", 0.0)
            if not book_intent:
                score += 0.35
            if faq_intent and not book_intent:
                score += 0.9
            if has_exact_book_match:
                score -= 0.6

            metadata = item["document"].get("metadata", {})
            if metadata.get("normalized_question") and metadata["normalized_question"] in query_norm:
                score += 0.8

            ranked = dict(item)
            ranked["hybrid_score"] = score
            combined.append(ranked)

        deduped = {}
        for item in combined:
            doc_id = item["document"]["id"]
            existing = deduped.get(doc_id)
            if existing is None or item["hybrid_score"] > existing["hybrid_score"]:
                deduped[doc_id] = item

        ranked_results = sorted(
            deduped.values(),
            key=lambda item: (item["hybrid_score"], -item.get("distance", 1.0)),
            reverse=True,
        )
        return ranked_results[:top_k]
