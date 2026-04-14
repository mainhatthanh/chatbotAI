from rag.query.normalize import content_tokens, extract_volume_number, normalize_text


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
