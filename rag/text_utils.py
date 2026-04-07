import re
import unicodedata


SUSPICIOUS_MOJIBAKE_MARKERS = ("Ã", "Ä", "Æ", "á»", "áº", "á", "Â")


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
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(text):
    repaired = repair_text(str(text)).lower().strip()
    accentless = remove_accents(repaired)
    cleaned = re.sub(r"[^a-z0-9\s]", " ", accentless)
    return re.sub(r"\s+", " ", cleaned).strip()


def tokenize(text):
    return [token for token in normalize_text(text).split() if len(token) > 1]


def contains_any(text, phrases):
    normalized = normalize_text(text)
    return any(normalize_text(phrase) in normalized for phrase in phrases)


def detect_book_intent(query):
    normalized = normalize_text(query)

    explicit_book_phrases = [
        "goi y sach", "tim sach", "mua sach", "cuon sach", "quyen sach",
        "truyen tranh", "tieu thuyet", "ten sach", "sach nao", "truyen nao",
        "sach", "truyen",
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
        any(normalize_text(phrase) in normalized for phrase in explicit_book_phrases)
        or any(normalize_text(phrase) in normalized for phrase in known_entities)
        or any(normalize_text(phrase) in normalized for phrase in book_attribute_phrases)
    )


def detect_faq_intent(query):
    return contains_any(query, [
        "doi tra", "tra hang", "hoan tien", "chinh sach", "giao hang",
        "ship", "van chuyen", "thanh toan", "cod", "chuyen khoan",
        "phuong thuc thanh toan", "bao lau", "thoi gian giao hang",
        "ho tro", "tu van", "cua hang",
    ])


def detect_list_intent(query):
    return contains_any(query, [
        "goi y", "de xuat", "danh sach", "nhung sach", "co sach nao",
        "sach nao", "truyen nao",
    ])


def detect_price_intent(query):
    return contains_any(query, ["gia", "bao nhieu tien", "bao nhieu"])


def detect_stock_intent(query):
    return contains_any(query, ["con hang", "ton kho", "so luong", "het hang"])


def detect_author_intent(query):
    return contains_any(query, ["tac gia", "viet boi", "cua ai"])
