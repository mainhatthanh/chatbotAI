from rag.data_loader import load_all_documents
from rag.embedder import Embedder

docs = load_all_documents("data/books.csv", "data/faq.json")
texts = [doc["text"] for doc in docs]

embedder = Embedder()
embeddings = embedder.encode_text(texts)

print("Số documents:", len(texts))
print("Shape embeddings:", embeddings.shape)
print("Vector đầu tiên có độ dài:", len(embeddings[0]))