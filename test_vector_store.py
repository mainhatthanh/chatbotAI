from rag.data_loader import load_all_documents
from rag.embedder import Embedder
from rag.vector_store import VectorStore

docs = load_all_documents("data/books.csv", "data/faq.json")
texts = [doc["text"] for doc in docs]

embedder = Embedder()
embeddings = embedder.encode_text(texts)

store = VectorStore()
store.add_documents(embeddings, docs)

query = "Cửa hàng có hỗ trợ COD không?"
query_embedding = embedder.encode_query(query)

results = store.search(query_embedding, top_k=2)

print("Câu hỏi:", query)
print("Kết quả tìm được:")
for item in results:
    print(item)