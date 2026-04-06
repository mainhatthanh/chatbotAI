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
    return contains_any(query, [
        "sach", "truyen", "tieu thuyet", "tac gia", "gia", "con hang",
        "ton kho", "nha gia kim", "doraemon", "conan", "harry potter",
        "one piece", "khong gia dinh", "dac nhan tam", "nguyen nhat anh",
        "goi y sach", "tim sach", "mua sach",
    ])


def detect_list_intent(query):
    return contains_any(query, [
        "nao", "goi y", "de xuat", "danh sach", "nhung sach", "co sach",
    ])


def detect_price_intent(query):
    return contains_any(query, ["gia", "bao nhieu tien", "bao nhieu"])


def detect_stock_intent(query):
    return contains_any(query, ["con hang", "ton kho", "so luong", "het hang"])


def detect_author_intent(query):
    return contains_any(query, ["tac gia", "viet boi", "cua ai"])
