import re
import math
from typing import Any, Dict, Type, Union, Optional, Tuple
from pydantic import BaseModel, parse_obj_as
from pydantic_core import from_json
from rouge_score import rouge_scorer

ROUGE_SCORER = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
def normalize_text(text: str) -> str:
    # Normalize text by removing control characters and standardizing Unicode.

    # Remove control characters except newline, tab, and carriage return
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    # Replace Unicode spaces with regular spaces
    text = re.sub(r'[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000\uFEFF]', ' ', text)

    # Replace Unicode quotes with standard quotes
    text = re.sub(r'[\u201C\u201D\u2018\u2019\u201E\u201F\u2039\u203A\u00AB\u00BB]', '"', text)

    # Replace Unicode quotes with standard quotes
    text = re.sub(r'[\u201C\u201D\u2018\u2019\u201E\u201F\u2039\u203A\u00AB\u00BB]', '"', text)

    # Replace Unicode dashes with standard dash
    text = re.sub(r'[\u2013\u2014\u2015]', '-', text)

    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    return text
def normalize_for_matching(text: str) -> str:
    """Normalize text for matching - quotes, whitespace, punctuation."""
    # One-liner version:
    # return re.sub(r'\s+', ' ', re.sub(r'[\'"`''""‛„:;()\[\]\-–—]|[.,](?!(?<=\d[.,])\d)', ' ', re.sub(r'(?<=[^\W\d_])(?<![MmXx])(?=\d)|(?<=\d)(?=[^\W\d_])', ' ', text))).strip().lower()
    text = re.sub(
        r'(?<=[^\W\d_])(?<![MmXx])(?=\d)|(?<=\d)(?=[^\W\d_])', ' ', text)

    
        # Step 2: Remove quotes and punctuation (but preserve decimals)
    text = re.sub(r'[\'"`''""‛„:;()\[\]\-–—]|[.,](?!(?<=\d[.,])\d)', ' ', text)

         # Step 3: Normalize whitespace and lowercase
    text = re.sub(r'\s+', ' ', text).strip().lower()
    
    return text

def trim_match_edges(query: str, match: str, normalize_text: bool = False) -> str:
    #Trim unnecessary characters from the beginning and end of a match
    if not match or match == query:
       return match
    # Get initial recall score
    if normalize_text:
        initial_recall = ROUGE_SCORER.score(normalize_for_matching(query), normalize_for_matching(match))['rougeL'].recall
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
        break  # Stop if recall drop

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
def find_closest_match(query: str, text: str, window_margin: int = 10, min_similarity: float = 0.9, normalize_text: bool = False) -> Optional[str]:
    '''Finds the substring of `text` that best matches `query` using ROUGE-L similarity.'''
    if query in text:
        return query

    # We'll attempt to match substrings in `text` that are around len(query).
    query_len = len(query)
     # Precompute the best match info
    best_substring = ""
    best_score = -math.inf

    if normalize_text:
        query = normalize_for_matching(query)

     # Because we want to allow some variation (extra words, punctuation, etc.),
    # we use a window size = query_len +/- window_margin
    min_len = max(1, query_len - window_margin)
    max_len = min(len(text), query_len + window_margin)

    # Slide over the text and compare substrings
    for start_idx in range(len(text)):
         # We'll try multiple substring lengths in [min_len, max_len]
        for length in range(min_len, max_len + 1):
            end_idx = start_idx + length

             if end_idx > len(text):
                break  # no need to go further if we exceed text length
            candidate_substring = text[start_idx:end_idx].strip()
            if normalize_text:
                normalized_candidate_substring = normalize_for_matching(candidate_substring)
                scores = ROUGE_SCORER.score(query, normalized_candidate_substring)\
            else:
                scores = ROUGE_SCORER.score(query, candidate_substring)
            rouge_l_score = scores['rougeL'].fmeasure  # f-measure of ROUGE-L

            if rouge_l_score > best_score:
                best_score = rouge_l_score
                best_substring = candidate_substring

            if best_score > 0.95:
                # Trim edges if not perfect
                best_substring = trim_match_edges(query, best_substring, normalize_text)
                assert best_substring in text, f"Best substring {best_substring} not in text {text}"
                return best_substring
        if best_score >= min_similarity:
             # Trim edges before returning  
              best_substring = trim_match_edges(query, best_substring, normalize_text)
              assert best_substring in text
              return best_substring




