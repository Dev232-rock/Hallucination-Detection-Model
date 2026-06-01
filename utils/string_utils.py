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