import math
import re
from typing import Optional, Tuple
try:
    from rouge_score import rouge_scorer
    ROUGE_SCORER = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
except ImportError:
    ROUGE_SCORER = None


def normalize_text(text: str) -> str:
    """Normalize text by removing control characters and standardizing Unicode."""
    # Remove control characters except newline, tab, and carriage return
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    # Replace Unicode spaces with regular spaces
    text = re.sub(r'[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000\uFEFF]', ' ', text)
    # Replace Unicode quotes with standard quotes
    text = re.sub(r'[\u201C\u201D\u2018\u2019\u201E\u201F\u2039\u203A\u00AB\u00BB]', '"', text)
    # Replace Unicode dashes with standard dash
    text = re.sub(r'[\u2013\u2014\u2015]', '-', text)
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text


def normalize_for_matching(text: str) -> str:
    """Normalize text for matching - quotes, whitespace, punctuation."""
    # Step 1: Separate numbers and words
    text = re.sub(r'(?<=[^\W\d_])(?<![MmXx])(?=\d)|(?<=\d)(?=[^\W\d_])', ' ', text)
    # Step 2: Remove quotes and punctuation (but preserve decimals)
    text = re.sub(r'[\'"`‛„:;()\[\]\-\u2013\u2014]|[.,](?!(?<=\d[.,])\d)', ' ', text)
    # Step 3: Normalize whitespace and lowercase
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def trim_match_edges(query: str, match: str, normalize_text: bool = False) -> str:
    """Trim unnecessary characters from the beginning and end of a match."""
    if not match or match == query:
        return match
    if ROUGE_SCORER is None:
        raise ImportError(
            "rouge-score is required for fuzzy string matching. Install it via: pip install rouge-score"
        )

    # Get initial recall score
    if normalize_text:
        initial_recall = ROUGE_SCORER.score(
            normalize_for_matching(query), normalize_for_matching(match)
        )['rougeL'].recall
    else:
        initial_recall = ROUGE_SCORER.score(query, match)['rougeL'].recall

    best_match = match
    best_recall = initial_recall

    # Try trimming from the beginning
    for i in range(1, min(len(match) // 2, 20)):  # Don't trim more than half or 20 chars
        trimmed = match[i:]
        if normalize_text:
            scores = ROUGE_SCORER.score(normalize_for_matching(query), normalize_for_matching(trimmed))
        else:
            scores = ROUGE_SCORER.score(query, trimmed)

        if scores['rougeL'].recall >= best_recall:
            best_match = trimmed
            best_recall = scores['rougeL'].recall
        else:
            break  # Stop if recall drops

    # Try trimming from the end
    match = best_match  # Start with best so far
    for i in range(1, min(len(match) // 2, 20)):
        trimmed = match[:-i]
        if normalize_text:
            scores = ROUGE_SCORER.score(normalize_for_matching(query), normalize_for_matching(trimmed))
        else:
            scores = ROUGE_SCORER.score(query, trimmed)

        if scores['rougeL'].recall >= best_recall:
            best_match = trimmed
            best_recall = scores['rougeL'].recall
        else:
            break  # Stop if recall drops

    return best_match


def find_closest_match(
    query: str,
    text: str,
    window_margin: int = 10,
    min_similarity: float = 0.9,
    normalize_text: bool = False
) -> Optional[str]:
    """Finds the substring of `text` that best matches `query` using ROUGE-L similarity."""
    if query in text:
        return query
    if ROUGE_SCORER is None:
        raise ImportError(
            "rouge-score is required for fuzzy string matching. Install it via: pip install rouge-score"
        )

    # We'll attempt to match substrings in `text` that are around len(query).
    query_len = len(query)
    best_substring = ""
    best_score = -math.inf

    target_query = normalize_for_matching(query) if normalize_text else query

    # Because we want to allow some variation (extra words, punctuation, etc.),
    # we use a window size = query_len +/- window_margin
    min_len = max(1, query_len - window_margin)
    max_len = min(len(text), query_len + window_margin)

    # Slide over the text and compare substrings
    for start_idx in range(len(text)):
        for length in range(min_len, max_len + 1):
            end_idx = start_idx + length
            if end_idx > len(text):
                break  # no need to go further if we exceed text length

            candidate_substring = text[start_idx:end_idx].strip()
            if not candidate_substring:
                continue

            if normalize_text:
                normalized_candidate = normalize_for_matching(candidate_substring)
                scores = ROUGE_SCORER.score(target_query, normalized_candidate)
            else:
                scores = ROUGE_SCORER.score(target_query, candidate_substring)

            rouge_l_score = scores['rougeL'].fmeasure

            if rouge_l_score > best_score:
                best_score = rouge_l_score
                best_substring = candidate_substring

            if best_score > 0.95:
                trimmed = trim_match_edges(query, best_substring, normalize_text)
                if trimmed in text:
                    return trimmed

    if best_score >= min_similarity:
        trimmed = trim_match_edges(query, best_substring, normalize_text)
        if trimmed in text:
            return trimmed

    if not normalize_text:
        return find_closest_match(query, text, window_margin, min_similarity, normalize_text=True)

    return None


def try_matching_span_in_text(
    span: str,
    text: str,
    cur_idx: int = 0,
    min_similarity: float = 0.8
) -> Tuple[Optional[str], Optional[int]]:
    """Try to match a span to a substring in the text."""
    closest_match: Optional[str] = find_closest_match(span, text[cur_idx:], min_similarity=min_similarity)

    if closest_match is not None:
        try:
            span_idx = text[cur_idx:].index(closest_match)
            return closest_match, cur_idx + span_idx
        except ValueError:
            pass

    # try searching for the span in the previous part of the text
    if cur_idx > 0:
        return try_matching_span_in_text(span, text, cur_idx=0, min_similarity=min_similarity)

    return None, None
