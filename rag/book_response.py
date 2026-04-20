from rag.query import (
    detect_category_intent,
    detect_description_intent,
    detect_existence_intent,
    detect_list_intent,
    detected_categories,
    score_book_match,
)

MAX_BOOKS_IN_RESPONSE = 5


def score_books(query, books):
    """Gan diem tung sach theo muc do khop voi query goc."""
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
    """Loc sach du diem khop; nguong thay doi theo loai intent."""
    books = _filter_by_requested_categories(query, books)
    scored_books = sorted(score_books(query, books), key=lambda item: item[1], reverse=True)
    top_match_score = max((score for _, score in scored_books), default=0.0)
    min_match_score = _minimum_match_score(query, top_match_score)

    return [book for book, score in scored_books if score >= min_match_score]


def should_list_books(query):
    """Cac intent nay nen tra ve danh sach thay vi mot cuon sach."""
    return (
        detect_list_intent(query)
        or detect_existence_intent(query)
        or detect_category_intent(query)
        or detect_description_intent(query)
    )


def filter_books_for_display(query, books):
    """Chon nhung sach se hien thi trong cau tra loi danh sach."""
    books = _filter_by_requested_categories(query, books)
    ranked = ranked_books(query, books)
    matched = matched_books(query, books)

    if detect_description_intent(query) or detect_category_intent(query):
        return matched

    return matched or ranked


def no_description_match_message():
    return "Xin lỗi, tôi chưa tìm thấy cuốn sách nào khớp rõ với mô tả đó."


def format_book_list(books):
    lines = ["Tôi tìm thấy những sách phù hợp:"]
    for book in books[:MAX_BOOKS_IN_RESPONSE]:
        meta = book.get("metadata", {})
        lines.append(
            f"- {meta.get('title', 'Không rõ')} | tác giả: {meta.get('author', 'Không rõ')} | "
            f"thể loại: {meta.get('category', 'Không rõ')} | giá: {meta.get('price', 'Không rõ')} đồng | "
            f"còn: {meta.get('stock', 'Không rõ')}"
        )
    return "\n".join(lines)


def _filter_by_requested_categories(query, books):
    requested_categories = detected_categories(query)
    if not requested_categories:
        return books

    return [
        book for book in books
        if book.get("metadata", {}).get("normalized_category") in requested_categories
    ]


def _minimum_match_score(query, top_match_score):
    """Nguong cao hon cho cau hoi ton tai/mo ta de tranh liet ke qua rong."""
    min_match_score = max(0.6, top_match_score * 0.45)

    if detect_category_intent(query):
        return 0.1

    if detect_existence_intent(query):
        return max(0.6, top_match_score * 0.75)

    if detect_description_intent(query):
        return max(0.7, top_match_score * 0.75)

    return min_match_score
