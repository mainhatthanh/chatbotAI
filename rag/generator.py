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

MIN_CONFIDENT_SCORE = 0.35


class Generator:
    """Tao cau tra loi cuoi cung tu cac document da retrieve/rerank."""

    def generate(self, query, retrieved_docs):
        if not retrieved_docs:
            return "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp."

        top_doc = retrieved_docs[0]["document"]
        top_score = retrieved_docs[0].get("hybrid_score", 0.0)
        if top_score < MIN_CONFIDENT_SCORE:
            return (
                "Xin lỗi, tôi chưa tìm thấy thông tin đủ chắc chắn để trả lời. "
                "Bạn có thể hỏi cụ thể hơn không?"
            )

        if top_doc["source"] == "faq":
            return self._answer_faq(retrieved_docs)

        if top_doc["source"] == "books":
            return self._answer_books(query, retrieved_docs)

        return top_doc.get("text", "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp.")

    def _answer_faq(self, retrieved_docs):
        """FAQ uu tien metadata.answer vi day la cau tra loi sach tu file JSON."""
        metadata = retrieved_docs[0]["document"].get("metadata", {})
        answer = metadata.get("answer")
        if answer:
            return answer

        text = retrieved_docs[0]["document"].get("text", "")
        if "Trả lời:" in text:
            return text.split("Trả lời:", 1)[1].strip()
        return text

    def _answer_books(self, query, retrieved_docs):
        top_books = [
            item["document"]
            for item in retrieved_docs
            if item["document"].get("source") == "books"
        ]
        if not top_books:
            return "Xin lỗi, tôi chưa tìm thấy sách phù hợp."

        main_book = self._best_book_for_query(query, top_books)
        info = main_book.get("metadata", {})

        if self._should_return_book_list(query, top_books):
            if detect_description_intent(query) and not matched_books(query, top_books):
                return no_description_match_message()
            return format_book_list(filter_books_for_display(query, top_books))

        return self._answer_single_book(query, info)

    def _best_book_for_query(self, query, books):
        """Chon sach co title/author/description khop query nhat trong top results."""
        return max(
            books,
            key=lambda book: score_book_match(query, book.get("metadata", {})),
        )

    def _should_return_book_list(self, query, books):
        if should_list_books(query):
            return True

        if len(books) < 2:
            return False

        # Cau hoi tac gia/gia/ton kho can tra loi mot sach cu the.
        return not (
            detect_author_intent(query)
            or detect_price_intent(query)
            or detect_stock_intent(query)
        )

    def _answer_single_book(self, query, info):
        title = info.get("title", "Cuốn sách này")
        author = info.get("author", "Không rõ")
        category = info.get("category", "Không rõ")
        price = info.get("price", "Không rõ")
        stock = info.get("stock", "Không rõ")
        description = info.get("description", "Không rõ")

        if detect_author_intent(query):
            return f"{title} do {author} viết."

        if detect_price_intent(query) and detect_stock_intent(query):
            return f"{title} hiện có giá {price} đồng và còn {stock} cuốn trong kho."

        if detect_price_intent(query):
            return f"{title} có giá {price} đồng."

        if detect_stock_intent(query):
            return f"{title} hiện còn {stock} cuốn trong kho."

        return (
            f"Tôi tìm thấy sách phù hợp: {title} của {author}. "
            f"Thể loại: {category}. "
            f"Giá: {price} đồng. "
            f"Còn: {stock}. "
            f"Mô tả ngắn: {description}"
        )
