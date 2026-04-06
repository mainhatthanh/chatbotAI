from rag.data_loader import load_all_documents
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.generator import Generator

docs = load_all_documents("data/books.csv", "data/faq.json")
texts = [doc["text"] for doc in docs]

embedder = Embedder()
embeddings = embedder.encode_text(texts)

store = VectorStore()
store.add_documents(embeddings, docs)

retriever = Retriever(store)
generator = Generator()

query = "Có thanh toán khi nhận hàng không?"
retrieved_docs = retriever.retrieve(query, top_k=2)
response = generator.generate(query, retrieved_docs)

print("Câu hỏi:", query)
print("Câu trả lời:", response)