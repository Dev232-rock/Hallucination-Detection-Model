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