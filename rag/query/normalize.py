import re
import unicodedata


SUSPICIOUS_MOJIBAKE_MARKERS = ("Ãƒ", "Ã„", "Ã†", "Ã¡Â»", "Ã¡Âº", "Ã¡", "Ã‚")
QUERY_STOPWORDS = {
    "ai", "ay", "ban", "bao", "biet", "cho", "chu", "co", "con", "cuon",
    "cua", "duoc", "gi", "giup", "goi", "hay", "het", "khong", "la", "loai",
    "may", "minh", "mot", "nao", "nay", "nhi", "nhieu", "noi", "o", "sach",
    "shop", "so", "tap", "the", "thi", "toi", "tra", "trong", "truyen", "tu",
    "van", "ve", "voi",
}


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
    text = text.replace("Ä‘", "d").replace("Ä", "D")
    text = text.replace("Ã„â€˜", "d").replace("Ã„Â", "D")
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
