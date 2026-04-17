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


def candidate_pool_size(query, top_k):
    candidate_k = max(top_k + 2, 5)
    if detect_existence_intent(query) or detect_category_intent(query) or detect_description_intent(query):
        candidate_k = max(candidate_k, 8)
    return candidate_k


def should_answer_with_faq(query, book_intent, faq_results):
    faq_intent = detect_faq_intent(query)
    top_faq_score = faq_results[0]["hybrid_score"] if faq_results else 0.0
    return faq_intent and (not book_intent or top_faq_score >= 0.85)


def rerank_results(query, book_results, faq_results, top_k, book_intent):
    query_norm = normalize_text(query)
    query_tokens = set(tokenize(query))
    faq_intent = detect_faq_intent(query)
    requested_categories = detected_categories(query)
    combined = []
    has_exact_book_match = False

    for item in book_results:
        score = item.get("hybrid_score", 0.0)
        if book_intent:
            score += 0.75

        metadata = item["document"].get("metadata", {})
        if requested_categories and metadata.get("normalized_category") not in requested_categories:
            continue

        score += score_book_match(query, metadata)
        normalized_title = metadata.get("normalized_title", "")
        if normalized_title and normalized_title in query_norm:
            score += 2.2
            has_exact_book_match = True
        if metadata.get("normalized_author") and metadata["normalized_author"] in query_norm:
            score += 1.4
            has_exact_book_match = True
        if normalized_title:
            matched_title_tokens = [
                token for token in normalized_title.split()
                if len(token) > 2 and token in query_tokens and token != "tap"
            ]
            if matched_title_tokens:
                score += 0.9 * len(matched_title_tokens)
                has_exact_book_match = True

        ranked = dict(item)
        ranked["hybrid_score"] = score
        combined.append(ranked)

    for item in faq_results:
        if requested_categories and combined:
            continue

        score = item.get("hybrid_score", 0.0)
        if not book_intent:
            score += 0.35
        if faq_intent and not book_intent:
            score += 0.9
        if has_exact_book_match:
            score -= 0.6

        metadata = item["document"].get("metadata", {})
        if metadata.get("normalized_question") and metadata["normalized_question"] in query_norm:
            score += 0.8

        ranked = dict(item)
        ranked["hybrid_score"] = score
        combined.append(ranked)

    deduped = {}
    for item in combined:
        doc_id = item["document"]["id"]
        existing = deduped.get(doc_id)
        if existing is None or item["hybrid_score"] > existing["hybrid_score"]:
            deduped[doc_id] = item

    ranked_results = sorted(
        deduped.values(),
        key=lambda item: (item["hybrid_score"], -item.get("distance", 1.0)),
        reverse=True,
    )
    return ranked_results[:top_k]
