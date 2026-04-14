import json

import pandas as pd

from rag.query import normalize_text, repair_text


def load_books(csv_path: str):
    df = pd.read_csv(csv_path)
    documents = []

    for _, row in df.iterrows():
        title = repair_text(row["title"])
        author = repair_text(row["author"])
        category = repair_text(row["category"])
        description = repair_text(row["description"])
        price = str(row["price"])
        stock = str(row["stock"])

        text = (
            f"Tên sách: {title}. "
            f"Tác giả: {author}. "
            f"Thể loại: {category}. "
            f"Mô tả: {description}. "
            f"Giá: {price} đồng. "
            f"Số lượng còn: {stock}. "
            f"Từ khóa tìm kiếm: {title}, {author}, {category}."
        )

        documents.append({
            "id": f"book_{row['id']}",
            "text": text,
            "source": "books",
            "metadata": {
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
            },
            "normalized_text": normalize_text(text),
        })

    return documents


def load_faq(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        faq_data = json.load(f)

    documents = []
    for i, item in enumerate(faq_data, start=1):
        question = repair_text(item["question"])
        answer = repair_text(item["answer"])
        text = (
            f"Câu hỏi: {question}. "
            f"Trả lời: {answer}"
        )

        documents.append({
            "id": f"faq_{i}",
            "text": text,
            "source": "faq",
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
    return {
        "books": load_books(books_path),
        "faq": load_faq(faq_path)
    }
