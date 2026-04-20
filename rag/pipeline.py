from rag.data_loader import load_all_documents
from rag.embedder import Embedder
from rag.generator import Generator
from rag.query import detect_book_intent
from rag.retriever import Retriever
from rag.reranker import candidate_pool_size, rerank_results, should_answer_with_faq
from rag.vector_store import VectorStore


DEFAULT_TOP_K = 3


class RAGPipeline:
    def __init__(self, books_path: str, faq_path: str):
        all_docs = load_all_documents(books_path, faq_path)

        self.book_documents = all_docs["books"]
        self.faq_documents = all_docs["faq"]

        # Dung chung mot Embedder de tranh load SentenceTransformer nhieu lan.
        self.embedder = Embedder()
        self.books_store = self._build_store(self.book_documents)
        self.faq_store = self._build_store(self.faq_documents)

        # Moi retriever search trong mot tap du lieu rieng, sau do reranker tron ket qua.
        self.books_retriever = Retriever(self.books_store, self.book_documents, self.embedder)
        self.faq_retriever = Retriever(self.faq_store, self.faq_documents, self.embedder)
        self.generator = Generator()

    def _build_store(self, documents):
        """Encode documents va nap vao VectorStore cosine search."""
        texts = [doc["text"] for doc in documents]
        embeddings = self.embedder.encode_text(texts)
        store = VectorStore()
        store.add_documents(embeddings, documents)
        return store

    def is_book_query(self, query: str):
        return detect_book_intent(query)

    def answer(self, query: str, top_k=DEFAULT_TOP_K):
        """Tra loi query bang FAQ neu chac chan, nguoc lai uu tien luong sach."""
        candidate_k = candidate_pool_size(query, top_k)
        book_intent = self.is_book_query(query)

        # FAQ luon duoc search truoc de bat cac cau hoi mua hang/giao hang/thanh toan.
        faq_results = self.faq_retriever.retrieve_hybrid(query, top_k=candidate_k)

        if should_answer_with_faq(query, book_intent, faq_results):
            retrieved_docs = faq_results[:top_k]
            response = self.generator.generate(query, retrieved_docs)
            return response, retrieved_docs

        # Neu khong phai FAQ ro rang, tron ket qua sach + FAQ roi rerank theo intent.
        book_results = self.books_retriever.retrieve_hybrid(query, top_k=candidate_k)
        retrieved_docs = rerank_results(query, book_results, faq_results, top_k, book_intent)
        response = self.generator.generate(query, retrieved_docs)
        return response, retrieved_docs
