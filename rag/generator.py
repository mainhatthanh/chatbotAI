from rag.book_response import (
    filter_books_for_display,
    format_book_list,
    matched_books,
    no_description_match_message,
    should_list_books,
)
from rag.query import (
    detect_author_intent,
    detect_description_intent,
    detect_price_intent,
    detect_stock_intent,
    score_book_match,
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

        main_book = max(
            top_books,
            key=lambda book: score_book_match(query, book.get("metadata", {})),
        )
        info = main_book.get("metadata", {})

        if should_list_books(query) or self._should_list_multiple_books(query, top_books):
            if detect_description_intent(query) and not matched_books(query, top_books):
                return no_description_match_message()
            return format_book_list(filter_books_for_display(query, top_books))

        if detect_author_intent(query):
            return f"{info.get('title', 'Cuon sach nay')} do {info.get('author', 'Khong ro')} viết."

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

        return (
            f"Tôi tìm thấy sách phù hợp: {info.get('title', 'Không rõ')} của {info.get('author', 'Không rõ')}. "
            f"Thể loại: {info.get('category', 'Không rõ')}. "
            f"Giá: {info.get('price', 'Không rõ')} đồng. "
            f"Còn: {info.get('stock', 'Không rõ')}. "
            f"Mô tả ngắn: {info.get('description', 'Không rõ')}"
        )
    def _should_list_multiple_books(self, query, books):
        if len(books) < 2:
            return False
        if detect_author_intent(query) or detect_price_intent(query) or detect_stock_intent(query):
            return False
        return True
