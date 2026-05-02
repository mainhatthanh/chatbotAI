from rag.query import (
    detect_category_intent,
    detect_description_intent,
    detect_existence_intent,
    detect_list_intent,
    detected_categories,
    score_book_match,
)

MAX_BOOKS_IN_RESPONSE = 5
UNKNOWN_VALUE = "Kh\u00f4ng r\u00f5"


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

    if detect_existence_intent(query):
        return matched

    return matched or ranked


def no_description_match_message():
    return "Xin l\u1ed7i, t\u00f4i ch\u01b0a t\u00ecm th\u1ea5y cu\u1ed1n s\u00e1ch n\u00e0o kh\u1edbp r\u00f5 v\u1edbi m\u00f4 t\u1ea3 \u0111\u00f3."


def no_book_match_message():
    return "Xin l\u1ed7i, hi\u1ec7n t\u1ea1i c\u1eeda h\u00e0ng ch\u01b0a c\u00f3 s\u00e1ch b\u1ea1n y\u00eau c\u1ea7u."


def format_book_list(books):
    lines = ["T\u00f4i t\u00ecm th\u1ea5y nh\u1eefng s\u00e1ch ph\u00f9 h\u1ee3p:"]
    for book in books[:MAX_BOOKS_IN_RESPONSE]:
        meta = book.get("metadata", {})
        lines.append(
            f"- {meta.get('title', UNKNOWN_VALUE)} | t\u00e1c gi\u1ea3: {meta.get('author', UNKNOWN_VALUE)} | "
            f"th\u1ec3 lo\u1ea1i: {meta.get('category', UNKNOWN_VALUE)} | gi\u00e1: {meta.get('price', UNKNOWN_VALUE)} \u0111\u1ed3ng | "
            f"c\u00f2n: {meta.get('stock', UNKNOWN_VALUE)}"
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
