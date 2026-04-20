from sentence_transformers import SentenceTransformer

from rag.model_config import resolve_local_model_path


class Embedder:
    def __init__(self, model_path=None):
        self.model_path = model_path or resolve_local_model_path()
        # local_files_only=True giup app offline khong goi Hugging Face khi runtime.
        self.model = SentenceTransformer(
            str(self.model_path),
            local_files_only=True,
        )

    def encode_text(self, texts):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def encode_query(self, query: str):
        return self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
