from rag.query.normalize import contains_any, normalize_text


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
    return bool(detected_categories(query))


def detected_categories(query):
    normalized = normalize_text(query)
    padded_query = f" {normalized} "
    return [
        category
        for category in KNOWN_CATEGORIES
        if f" {category} " in padded_query
    ]


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
