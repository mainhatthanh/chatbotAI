import json

import pandas as pd

from rag.query import normalize_text, repair_text

BOOK_SOURCE = "books"
FAQ_SOURCE = "faq"


def _book_search_text(metadata):
    """Tao text dua vao embedding: gom du thong tin can search va tra loi."""
    return (
        f"Ten sach: {metadata['title']}. "
        f"Tac gia: {metadata['author']}. "
        f"The loai: {metadata['category']}. "
        f"Mo ta: {metadata['description']}. "
        f"Gia: {metadata['price']} dong. "
        f"So luong con: {metadata['stock']}. "
        f"Tu khoa tim kiem: {metadata['title']}, {metadata['author']}, {metadata['category']}."
    )


def _book_metadata(row):
    """Sua loi encoding trong data, sau do tao metadata goc va ban normalized."""
    title = repair_text(row["title"])
    author = repair_text(row["author"])
    category = repair_text(row["category"])
    description = repair_text(row["description"])
    price = str(row["price"])
    stock = str(row["stock"])

    return {
        "title": title,
        "author": author,
        "category": category,
        "description": description,
        "price": price,
        "stock": stock,
        "normalized_title": normalize_text(title),
        "normalized_author": normalize_text(author),
        "normalized_category": normalize_text(category),
        "normalized_description": normalize_text(description),
    }


def _faq_search_text(question, answer):
    """Tao text FAQ dua vao embedding va keyword search."""
    return f"Cau hoi: {question}. Tra loi: {answer}"


def load_books(csv_path: str):
    """Doc CSV sach va chuyen moi dong thanh document cho RAG."""
    df = pd.read_csv(csv_path)
    documents = []

    for _, row in df.iterrows():
        metadata = _book_metadata(row)
        text = _book_search_text(metadata)

        documents.append({
            "id": f"book_{row['id']}",
            "text": text,
            "source": BOOK_SOURCE,
            "metadata": metadata,
            "normalized_text": normalize_text(text),
        })

    return documents


def load_faq(json_path: str):
    """Doc FAQ JSON va chuyen thanh document de search chung voi sach."""
    with open(json_path, "r", encoding="utf-8") as f:
        faq_data = json.load(f)

    documents = []
    for i, item in enumerate(faq_data, start=1):
        question = repair_text(item["question"])
        answer = repair_text(item["answer"])
        text = _faq_search_text(question, answer)

        documents.append({
            "id": f"faq_{i}",
            "text": text,
            "source": FAQ_SOURCE,
            "metadata": {
                "question": question,
                "answer": answer,
                "normalized_question": normalize_text(question),
                "normalized_answer": normalize_text(answer),
            },
            "normalized_text": normalize_text(text),
        })

    return documents


def load_all_documents(books_path: str, faq_path: str):
    """Tra ve hai tap document tach rieng de pipeline build store rieng."""
    return {
        "books": load_books(books_path),
        "faq": load_faq(faq_path),
    }
