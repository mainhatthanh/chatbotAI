from rag.data_loader import load_all_documents
from rag.embedder import Embedder
from rag.generator import Generator
from rag.query import detect_book_intent
from rag.retriever import Retriever
from rag.reranker import candidate_pool_size, rerank_results, should_answer_with_faq
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
        candidate_k = candidate_pool_size(query, top_k)
        book_intent = self.is_book_query(query)

        faq_results = self.faq_retriever.retrieve_hybrid(query, top_k=candidate_k)

        if should_answer_with_faq(query, book_intent, faq_results):
            retrieved_docs = faq_results[:top_k]
            response = self.generator.generate(query, retrieved_docs)
            return response, retrieved_docs

        book_results = self.books_retriever.retrieve_hybrid(query, top_k=candidate_k)
        retrieved_docs = rerank_results(query, book_results, faq_results, top_k, book_intent)
        response = self.generator.generate(query, retrieved_docs)
        return response, retrieved_docs
