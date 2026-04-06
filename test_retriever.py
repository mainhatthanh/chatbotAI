from rag.data_loader import load_all_documents
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.retriever import Retriever

docs = load_all_documents("data/books.csv", "data/faq.json")
texts = [doc["text"] for doc in docs]

embedder = Embedder()
embeddings = embedder.encode_text(texts)

store = VectorStore()
store.add_documents(embeddings, docs)

retriever = Retriever(store)

query = "Thanh toán như thế nào"
results = retriever.retrieve(query, top_k=2)

print("Câu hỏi:", query)
print("Các kết quả gần nhất:")
for item in results:
    print(item)