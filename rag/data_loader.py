import json
import pandas as pd

def load_books(csv_path: str):
    df = pd.read_csv(csv_path)
    documents = []

    for _, row in df.iterrows():
        text = (
            f"Tên sách: {row['title']}. "
            f"Tác giả: {row['author']}. "
            f"Thể loại: {row['category']}. "
            f"Mô tả: {row['description']}. "
            f"Giá: {row['price']} đồng. "
            f"Số lượng còn: {row['stock']}. "
            f"Từ khóa tìm kiếm: {row['title']}, {row['author']}, {row['category']}."
        )

        documents.append({
            "id": f"book_{row['id']}",
            "text": text,
            "source": "books",
            "metadata": {
                "title": str(row["title"]).lower(),
                "author": str(row["author"]).lower(),
                "category": str(row["category"]).lower(),
                "price": str(row["price"]),
                "stock": str(row["stock"])
            }
        })

    return documents


def load_faq(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        faq_data = json.load(f)

    documents = []
    for i, item in enumerate(faq_data, start=1):
        text = (
            f"Câu hỏi: {item['question']}. "
            f"Trả lời: {item['answer']}"
        )

        documents.append({
            "id": f"faq_{i}",
            "text": text,
            "source": "faq"
        })

    return documents


def load_all_documents(books_path: str, faq_path: str):
    return {
        "books": load_books(books_path),
        "faq": load_faq(faq_path)
    }