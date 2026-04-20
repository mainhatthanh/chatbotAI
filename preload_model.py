#!/usr/bin/env python3
"""
Tai model embedding ve thu muc local cua du an de app co the chay offline.

Chay mot lan khi co internet:
    venv\Scripts\python.exe preload_model.py
"""

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from rag.model_config import MODEL_NAME, PROJECT_MODEL_DIR, resolve_local_model_path


def preload_model():
    print(f"Dang tai model: {MODEL_NAME}")
    print(f"Thu muc local: {PROJECT_MODEL_DIR}")

    try:
        # Tai day du snapshot ve thu muc trong project de runtime khong can internet.
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=str(PROJECT_MODEL_DIR),
            local_dir_use_symlinks=False,
        )

        # Load lai bang local_files_only de kiem tra dung che do offline.
        model_path = resolve_local_model_path()
        model = SentenceTransformer(str(model_path), local_files_only=True)

        test_texts = ["Xin chao", "Hello world"]
        embeddings = model.encode(test_texts)

        print(f"OK: Model da san sang offline. Shape embeddings: {embeddings.shape}")
        return True
    except Exception as exc:
        print(f"Loi khi tai/kiem tra model: {exc}")
        return False


if __name__ == "__main__":
    if preload_model():
        print("\nHoan thanh. Ban co the chay ung dung khi khong co mang.")
    else:
        print("\nThat bai. Hay kiem tra ket noi internet va cai dat dependencies.")
