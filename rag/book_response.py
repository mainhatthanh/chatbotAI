from rag.query import (
    detect_category_intent,
    detect_description_intent,
    detect_existence_intent,
    detect_list_intent,
    detected_categories,
    score_book_match,
)


def score_books(query, books):
    return [
        (book, score_book_match(query, book.get("metadata", {})))
        for book in books
    ]


def ranked_books(query, books):
    return [
        book for book, _ in sorted(
            score_books(query, books),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def matched_books(query, books):
    requested_categories = detected_categories(query)
    if requested_categories:
        books = [
            book for book in books
            if book.get("metadata", {}).get("normalized_category") in requested_categories
        ]

    scored_books = sorted(score_books(query, books), key=lambda item: item[1], reverse=True)
    top_match_score = max((score for _, score in scored_books), default=0.0)
    min_match_score = max(0.6, top_match_score * 0.45)

    if detect_category_intent(query):
        min_match_score = 0.1

    if detect_existence_intent(query):
        min_match_score = max(0.6, top_match_score * 0.75)

    if detect_description_intent(query):
        min_match_score = max(0.6, top_match_score * 0.55)

    return [book for book, score in scored_books if score >= min_match_score]


def should_list_books(query):
    return (
        detect_list_intent(query)
        or detect_existence_intent(query)
        or detect_category_intent(query)
        or detect_description_intent(query)
    )


def filter_books_for_display(query, books):
    requested_categories = detected_categories(query)
    if requested_categories:
        books = [
            book for book in books
            if book.get("metadata", {}).get("normalized_category") in requested_categories
        ]

    ranked = ranked_books(query, books)
    matched = matched_books(query, books)

    if detect_description_intent(query):
        return matched

    if detect_category_intent(query):
        return matched

    return matched or ranked


def no_description_match_message():
    return "Xin lỗi, tôi chưa tìm thấy cuốn sách nào khớp rõ với mô tả đó."


def format_book_list(books):
    lines = ["Tôi tìm thấy những sách phù hợp:"]
    for book in books[:5]:
        meta = book.get("metadata", {})
        lines.append(
            f"- {meta.get('title', 'Không rõ')} | tác giả: {meta.get('author', 'Không rõ')} | "
            f"thể loại: {meta.get('category', 'Không rõ')} | giá: {meta.get('price', 'Không rõ')} đồng | "
            f"còn: {meta.get('stock', 'Không rõ')}"
        )
    return "\n".join(lines)
