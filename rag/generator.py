from rag.text_utils import (
    detect_author_intent,
    detect_list_intent,
    detect_price_intent,
    detect_stock_intent,
)


class Generator:
    def generate(self, query, retrieved_docs):
        if not retrieved_docs:
            return "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp."

        top_doc = retrieved_docs[0]["document"]
        top_score = retrieved_docs[0].get("hybrid_score", 0.0)
        if top_score < 0.35:
            return "Xin lỗi, tôi chưa tìm thấy thông tin đủ chắc chắn để trả lời. Bạn có thể hỏi cụ thể hơn không?"

        if top_doc["source"] == "faq":
            return self._answer_faq(retrieved_docs)

        if top_doc["source"] == "books":
            return self._answer_books(query, retrieved_docs)

        return top_doc.get("text", "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp.")

    def _answer_faq(self, retrieved_docs):
        metadata = retrieved_docs[0]["document"].get("metadata", {})
        answer = metadata.get("answer")
        if answer:
            return answer

        text = retrieved_docs[0]["document"].get("text", "")
        if "Trả lời:" in text:
            return text.split("Trả lời:", 1)[1].strip()
        return text

    def _answer_books(self, query, retrieved_docs):
        top_books = [item["document"] for item in retrieved_docs if item["document"].get("source") == "books"]
        if not top_books:
            return "Xin lỗi, tôi chưa tìm thấy sách phù hợp."

        main_book = top_books[0]
        info = main_book.get("metadata", {})

        if detect_list_intent(query):
            lines = ["Tôi gợi ý một vài sách phù hợp:"]
            for book in top_books[:3]:
                meta = book.get("metadata", {})
                lines.append(
                    f"- {meta.get('title', 'Không rõ')} | tác giả: {meta.get('author', 'Không rõ')} | "
                    f"thể loại: {meta.get('category', 'Không rõ')} | giá: {meta.get('price', 'Không rõ')} đồng | "
                    f"còn: {meta.get('stock', 'Không rõ')}"
                )
            return "\n".join(lines)

        if detect_price_intent(query) and detect_stock_intent(query):
            return (
                f"{info.get('title', 'Cuốn sách này')} hiện có giá {info.get('price', 'Không rõ')} đồng "
                f"và còn {info.get('stock', 'Không rõ')} cuốn trong kho."
            )

        if detect_price_intent(query):
            return (
                f"{info.get('title', 'Cuốn sách này')} có giá {info.get('price', 'Không rõ')} đồng."
            )

        if detect_stock_intent(query):
            return (
                f"{info.get('title', 'Cuốn sách này')} hiện còn {info.get('stock', 'Không rõ')} cuốn trong kho."
            )

        if detect_author_intent(query):
            return (
                f"{info.get('title', 'Cuốn sách này')} do {info.get('author', 'Không rõ')} viết."
            )

        return (
            f"Tôi tìm thấy sách phù hợp: {info.get('title', 'Không rõ')} của {info.get('author', 'Không rõ')}. "
            f"Thể loại: {info.get('category', 'Không rõ')}. "
            f"Giá: {info.get('price', 'Không rõ')} đồng. "
            f"Còn: {info.get('stock', 'Không rõ')}. "
            f"Mô tả ngắn: {info.get('description', 'Không rõ')}"
        )
