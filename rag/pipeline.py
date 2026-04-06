from rag.data_loader import load_all_documents
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.generator import Generator

class RAGPipeline:
    def __init__(self, books_path: str, faq_path: str):
        all_docs = load_all_documents(books_path, faq_path)

        self.book_documents = all_docs["books"]
        self.faq_documents = all_docs["faq"]

        self.embedder = Embedder()

        # books store
        book_texts = [doc["text"] for doc in self.book_documents]
        book_embeddings = self.embedder.encode_text(book_texts)
        self.books_store = VectorStore()
        self.books_store.add_documents(book_embeddings, self.book_documents)

        # faq store
        faq_texts = [doc["text"] for doc in self.faq_documents]
        faq_embeddings = self.embedder.encode_text(faq_texts)
        self.faq_store = VectorStore()
        self.faq_store.add_documents(faq_embeddings, self.faq_documents)

        self.books_retriever = Retriever(self.books_store, self.book_documents)
        self.faq_retriever = Retriever(self.faq_store, self.faq_documents)

        self.generator = Generator()

    def is_book_query(self, query: str):
        query_lower = query.lower()
        book_keywords = [
            "sách", "truyện", "tiểu thuyết", "tác giả", "giá", "còn hàng",
            "doraemon", "conan", "one piece", "harry potter",
            "không gia đình", "cha giàu cha nghèo", "đắc nhân tâm"
        ]
        return any(keyword in query_lower for keyword in book_keywords)

    def answer(self, query: str, top_k=3):
        if self.is_book_query(query):
            # Ưu tiên keyword search cho books
            retrieved_docs = self.books_retriever.keyword_search_books(query, top_k=top_k)

            # Nếu keyword search yếu thì fallback sang vector search
            if not retrieved_docs:
                retrieved_docs = self.books_retriever.retrieve(query, top_k=top_k)
        else:
            retrieved_docs = self.faq_retriever.retrieve(query, top_k=top_k)

        response = self.generator.generate(query, retrieved_docs)
        return response, retrieved_docs