from rag.data_loader import load_all_documents

docs = load_all_documents("data/books.csv", "data/faq.json")

print("Tổng số documents:", len(docs))
for doc in docs:
    print(doc)