import re
import unicodedata


SUSPICIOUS_MOJIBAKE_MARKERS = ("Ãƒ", "Ã„", "Ã†", "Ã¡Â»", "Ã¡Âº", "Ã¡", "Ã‚")
QUERY_STOPWORDS = {
    "ai", "ay", "ban", "bao", "biet", "cho", "chu", "co", "con", "cuon",
    "cua", "duoc", "gi", "giup", "hay", "het", "khong", "la", "loai", "may",
    "minh", "mot", "nao", "nay", "nhi", "nhieu", "noi", "o", "sach", "shop",
    "so", "tap", "the", "thi", "toi", "tra", "trong", "tu", "van", "ve", "voi",
}
KNOWN_CATEGORIES = [
    "truyen tranh",
    "tieu thuyet",
    "ky nang",
    "thieu nhi",
    "khoa hoc",
]
DESCRIPTION_HINT_PHRASES = [
    "noi ve",
    "ke ve",
    "mo ta",
    "gioi thieu",
    "co noi dung",
    "noi dung ve",
    "cau chuyen ve",
    "hanh trinh",
    "phieu luu",
    "ban nang",
]


def repair_text(text):
    if not isinstance(text, str):
        return ""

    if not any(marker in text for marker in SUSPICIOUS_MOJIBAKE_MARKERS):
        return text

    for encoding in ("cp1252", "latin1"):
        try:
            repaired = text.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue

        if repaired:
            return repaired

    return text


def remove_accents(text):
    text = text.replace("đ", "d").replace("Đ", "D")
    text = text.replace("Ä‘", "d").replace("Ä", "D")
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(text):
    repaired = repair_text(str(text)).lower().strip()
    accentless = remove_accents(repaired)
    cleaned = re.sub(r"[^a-z0-9\s]", " ", accentless)
    return re.sub(r"\s+", " ", cleaned).strip()


def tokenize(text):
    return [token for token in normalize_text(text).split() if len(token) > 1]


def content_tokens(text):
    return [token for token in tokenize(text) if token not in QUERY_STOPWORDS]


def contains_any(text, phrases):
    normalized = normalize_text(text)
    padded_text = f" {normalized} "

    for phrase in phrases:
        normalized_phrase = normalize_text(phrase)
        if normalized_phrase and f" {normalized_phrase} " in padded_text:
            return True

    return False


def extract_volume_number(text):
    normalized = normalize_text(text)
    match = re.search(r"\btap\s+(\d+)\b", normalized)
    if match:
        return match.group(1)
    return None


def score_book_match(query, metadata):
    query_norm = normalize_text(query)
    query_tokens = set(content_tokens(query))
    title = metadata.get("normalized_title", "")
    author = metadata.get("normalized_author", "")
    category = metadata.get("normalized_category", "")
    description = metadata.get("normalized_description", "")
    score = 0.0

    if title and title in query_norm:
        score += 4.0
    if author and author in query_norm:
        score += 2.0
    if category and category in query_norm:
        score += 1.0
    if description and query_norm and query_norm in description:
        score += 3.2

    title_tokens = [token for token in title.split() if len(token) > 2 and token != "tap"]
    score += 0.6 * sum(1 for token in title_tokens if token in query_tokens)

    author_tokens = [token for token in author.split() if len(token) > 2]
    score += 0.4 * sum(1 for token in author_tokens if token in query_tokens)

    description_tokens = [token for token in description.split() if len(token) > 2]
    score += 0.8 * sum(1 for token in description_tokens if token in query_tokens)

    query_volume = extract_volume_number(query)
    title_volume = extract_volume_number(title)
    if query_volume and title_volume:
        if query_volume == title_volume:
            score += 2.5
        else:
            score -= 2.0

    return score


def detect_book_intent(query):
    normalized = normalize_text(query)

    strong_book_phrases = [
        "goi y sach", "tim sach", "mua sach", "cuon sach", "quyen sach",
        "truyen tranh", "tieu thuyet", "ten sach", "sach nao", "truyen nao",
        "gioi thieu sach", "tim truyen", "mua truyen",
    ]
    known_entities = [
        "nha gia kim", "doraemon", "conan", "harry potter",
        "one piece", "khong gia dinh", "dac nhan tam", "nguyen nhat anh",
    ]
    book_attribute_phrases = [
        "tac gia", "viet boi", "gia sach", "sach gia bao nhieu",
        "con hang", "ton kho", "het hang",
    ]

    return (
        any(normalize_text(phrase) in normalized for phrase in strong_book_phrases)
        or any(normalize_text(phrase) in normalized for phrase in known_entities)
        or any(normalize_text(phrase) in normalized for phrase in book_attribute_phrases)
        or detect_description_intent(query)
    )


def detect_faq_intent(query):
    return contains_any(query, [
        "xin chao", "chao", "hello", "hi", "shop oi", "ad oi",
        "ban la ai", "ban giup duoc gi", "can ho tro",
        "doi tra", "tra hang", "hoan tien", "chinh sach", "giao hang",
        "ship", "van chuyen", "thanh toan", "cod", "chuyen khoan",
        "phuong thuc thanh toan", "bao lau", "thoi gian giao hang",
        "ho tro", "tu van", "tu van sach", "cua hang",
    ])


def detect_list_intent(query):
    return contains_any(query, [
        "goi y", "de xuat", "danh sach", "nhung sach", "co sach nao",
        "sach nao", "truyen nao", "nhung cuon nao", "cuon nao",
    ])


def detect_existence_intent(query):
    normalized = normalize_text(query)
    return normalized.startswith("co ") and normalized.endswith(" khong")


def detect_category_intent(query):
    return any(category in normalize_text(query) for category in KNOWN_CATEGORIES)


def detect_description_intent(query):
    normalized = normalize_text(query)
    return any(phrase in normalized for phrase in DESCRIPTION_HINT_PHRASES)


def detect_price_intent(query):
    explicit_price = contains_any(query, [
        "gia bao nhieu",
        "bao nhieu tien",
        "gia tien",
        "muc gia",
        "co gia",
        "ban voi gia",
        "gia cua",
    ])
    generic_price = (
        contains_any(query, ["bao nhieu"])
        and not detect_author_intent(query)
        and not detect_stock_intent(query)
    )
    return explicit_price or generic_price


def detect_stock_intent(query):
    return contains_any(query, [
        "con hang",
        "ton kho",
        "so luong",
        "het hang",
        "con bao nhieu",
        "bao nhieu cuon",
        "con may cuon",
        "may cuon",
        "so cuon con lai",
    ])


def detect_author_intent(query):
    return contains_any(query, ["tac gia", "viet boi", "ai viet", "cua ai"])


# Backward-compatible re-exports from the modular query package.
from rag.query import (  # noqa: E402
    KNOWN_CATEGORIES,
    QUERY_STOPWORDS,
    SUSPICIOUS_MOJIBAKE_MARKERS,
    contains_any,
    content_tokens,
    detect_author_intent,
    detect_book_intent,
    detect_category_intent,
    detect_description_intent,
    detect_existence_intent,
    detect_faq_intent,
    detect_list_intent,
    detect_price_intent,
    detect_stock_intent,
    extract_volume_number,
    normalize_text,
    remove_accents,
    repair_text,
    score_book_match,
    tokenize,
)
