from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = ["Tôi thích học AI", "Machine learning rất thú vị"]
embeddings = model.encode(sentences)

print(embeddings.shape)