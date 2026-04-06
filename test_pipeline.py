from rag.pipeline import RAGPipeline

pipeline = RAGPipeline("data/books.csv", "data/faq.json")

query = "Có sách AI nào cho người mới không?"
response, retrieved_docs = pipeline.answer(query, top_k=2)

print("Câu hỏi:", query)
print("Câu trả lời:", response)
print("\nCác document tìm được:")
for item in retrieved_docs:
    print(item)