from pathlib import Path

from rag.data_loader import load_all_documents
from rag.embedder import Embedder
from rag.generator import Generator
from rag.query import detect_book_intent
from rag.retriever import Retriever
from rag.reranker import candidate_pool_size, rerank_results, should_answer_with_faq
from rag.vector_store import VectorStore


DEFAULT_TOP_K = 3
VECTORSTORE_DIR = Path("vectorstore")
FAISS_INDEX_PATH = VECTORSTORE_DIR / "faiss.index"
DOCUMENTS_PATH = VECTORSTORE_DIR / "documents.pkl"


class RAGPipeline:
    def __init__(self, books_path: str, faq_path: str):
        all_docs = load_all_documents(books_path, faq_path)

        self.book_documents = all_docs["books"]
        self.faq_documents = all_docs["faq"]
        self.all_documents = self.book_documents + self.faq_documents

        # Dung chung mot Embedder de tranh load SentenceTransformer nhieu lan.
        self.embedder = Embedder()
        self.store = self._load_or_build_store(self.all_documents, [books_path, faq_path])

        # Moi retriever search trong mot tap du lieu rieng, sau do reranker tron ket qua.
        self.books_retriever = Retriever(self.store, self.book_documents, self.embedder, source_filter="books")
        self.faq_retriever = Retriever(self.store, self.faq_documents, self.embedder, source_filter="faq")
        self.generator = Generator()

    def _build_store(self, documents):
        """Encode documents va nap vao VectorStore cosine search."""
        texts = [doc["text"] for doc in documents]
        embeddings = self.embedder.encode_text(texts)
        store = VectorStore()
        store.add_documents(embeddings, documents)
        return store

    def _load_or_build_store(self, documents, source_paths):
        """Load FAISS store tu disk neu con moi; nguoc lai build va save lai."""
        VECTORSTORE_DIR.mkdir(exist_ok=True)
        if self._store_is_fresh(source_paths):
            return VectorStore.load(FAISS_INDEX_PATH, DOCUMENTS_PATH)

        store = self._build_store(documents)
        store.save(FAISS_INDEX_PATH, DOCUMENTS_PATH)
        return store

    def _store_is_fresh(self, source_paths):
        if not FAISS_INDEX_PATH.exists() or not DOCUMENTS_PATH.exists():
            return False

        store_mtime = min(FAISS_INDEX_PATH.stat().st_mtime, DOCUMENTS_PATH.stat().st_mtime)
        latest_source_mtime = max(Path(path).stat().st_mtime for path in source_paths)
        return store_mtime >= latest_source_mtime

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
