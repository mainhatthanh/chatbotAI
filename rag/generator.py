class Generator:
    def generate(self, query: str, retrieved_docs):
        if not retrieved_docs:
            return "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp."

        top_doc = retrieved_docs[0]["document"]
        source = top_doc["source"]
        text = top_doc["text"]

        if source == "faq":
            if "Trả lời:" in text:
                answer_part = text.split("Trả lời:", 1)[1].strip()
                return answer_part
            return text

        if source == "books":
            parts = text.split(". ")
            info = {}

            for part in parts:
                if ":" in part:
                    key, value = part.split(":", 1)
                    info[key.strip()] = value.strip().rstrip(".")

            title = info.get("Tên sách", "Không rõ")
            author = info.get("Tác giả", "Không rõ")
            category = info.get("Thể loại", "Không rõ")
            description = info.get("Mô tả", "Không rõ")
            price = info.get("Giá", "Không rõ")
            stock = info.get("Số lượng còn", "Không rõ")

            return (
                f"Tôi tìm thấy một sách phù hợp:\n"
                f"- Tên sách: {title}\n"
                f"- Tác giả: {author}\n"
                f"- Thể loại: {category}\n"
                f"- Mô tả: {description}\n"
                f"- Giá: {price}\n"
                f"- Số lượng còn: {stock}"
            )

        return text