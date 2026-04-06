from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)


    def encode_text(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)
    

    def encode_query(self, query: str):
        return self.model.encode([query], convert_to_numpy=True)[0]