from rag.query import (
    detect_category_intent,
    detect_description_intent,
    detect_existence_intent,
    detect_faq_intent,
    detected_categories,
    normalize_text,
    score_book_match,
    tokenize,
)
from rag.query.normalize import contains_similar_token

MIN_CANDIDATE_K = 5
WIDE_CANDIDATE_K = 8


def candidate_pool_size(query, top_k):
    """Mo rong pool khi query can nhieu ung vien de loc lai chinh xac."""
    candidate_k = max(top_k + 2, MIN_CANDIDATE_K)
    needs_wide_pool = (
        detect_existence_intent(query)
        or detect_category_intent(query)
        or detect_description_intent(query)
    )
    if needs_wide_pool:
        candidate_k = max(candidate_k, WIDE_CANDIDATE_K)
    return candidate_k


def should_answer_with_faq(query, book_intent, faq_results):
    """Chi uu tien FAQ khi query that su la FAQ hoac FAQ co diem rat cao."""
    faq_intent = detect_faq_intent(query)
    top_faq_score = faq_results[0]["hybrid_score"] if faq_results else 0.0
    return faq_intent and (not book_intent or top_faq_score >= 0.85)


def rerank_results(query, book_results, faq_results, top_k, book_intent):
    """Tron ket qua sach/FAQ va cong diem theo intent de chon document tot nhat."""
    context = _rerank_context(query)
    combined = []
    has_book_match = False

    for item in book_results:
        ranked = _score_book_result(item, query, context, book_intent)
        if ranked is None:
            continue

        has_book_match = has_book_match or ranked.pop("_has_book_match")
        combined.append(ranked)

    for item in faq_results:
        ranked = _score_faq_result(item, context, book_intent, has_book_match)
        if ranked is not None:
            combined.append(ranked)

    return _rank(_dedupe_by_best_score(combined))[:top_k]


def _rerank_context(query):
    return {
        "query_norm": normalize_text(query),
        "query_tokens": set(tokenize(query)),
        "faq_intent": detect_faq_intent(query),
        "requested_categories": detected_categories(query),
    }


def _score_book_result(item, query, context, book_intent):
    metadata = item["document"].get("metadata", {})
    if not _matches_requested_category(metadata, context["requested_categories"]):
        return None

    score = item.get("hybrid_score", 0.0)
    if book_intent:
        score += 0.75

    score += score_book_match(query, metadata)
    title_score, has_book_match = _title_author_bonus(metadata, context)
    score += title_score

    ranked = dict(item)
    ranked["hybrid_score"] = score
    ranked["_has_book_match"] = has_book_match
    return ranked


def _score_faq_result(item, context, book_intent, has_book_match):
    if context["requested_categories"]:
        return None

    score = item.get("hybrid_score", 0.0)
    if not book_intent:
        score += 0.35
    if context["faq_intent"] and not book_intent:
        score += 0.9

    # Khi title sach da match ro, giam FAQ de tranh tra loi mua hang sai ngu canh.
    if has_book_match:
        score -= 0.6

    metadata = item["document"].get("metadata", {})
    question = metadata.get("normalized_question")
    if question and question in context["query_norm"]:
        score += 0.8

    ranked = dict(item)
    ranked["hybrid_score"] = score
    return ranked


def _title_author_bonus(metadata, context):
    query_norm = context["query_norm"]
    query_tokens = context["query_tokens"]
    normalized_title = metadata.get("normalized_title", "")
    normalized_author = metadata.get("normalized_author", "")
    score = 0.0
    has_book_match = False

    if normalized_title and normalized_title in query_norm:
        score += 2.2
        has_book_match = True

    if normalized_author and normalized_author in query_norm:
        score += 1.4
        has_book_match = True

    title_tokens = _title_tokens(normalized_title)
    matched_title_tokens = [token for token in title_tokens if token in query_tokens]
    if matched_title_tokens:
        score += 2.5 * len(matched_title_tokens)
        has_book_match = True
    elif contains_similar_token(query_tokens, title_tokens):
        # Fuzzy match giup cac loi go nhe van bam dung title.
        score += 2.0
        has_book_match = True

    return score, has_book_match


def _title_tokens(normalized_title):
    return [
        token
        for token in normalized_title.split()
        if len(token) > 2 and token != "tap"
    ]


def _matches_requested_category(metadata, requested_categories):
    if not requested_categories:
        return True
    return metadata.get("normalized_category") in requested_categories


def _dedupe_by_best_score(items):
    deduped = {}
    for item in items:
        doc_id = item["document"]["id"]
        existing = deduped.get(doc_id)
        if existing is None or item["hybrid_score"] > existing["hybrid_score"]:
            deduped[doc_id] = item
    return deduped.values()


def _rank(items):
    return sorted(
        items,
        key=lambda item: (item["hybrid_score"], -item.get("distance", 1.0)),
        reverse=True,
    )
