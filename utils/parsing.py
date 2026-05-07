# Utilities for parsing and validating JSON from LLM responses.

import re
from typing import Any, List, Type, TypeVar, Union

from pydantic import BaseModel, parse_obj_as
from pydantic_core import from_json

from .string_utils import normalize_text

T = TypeVar('T', bound=BaseModel)
def parse_and_validate_json(
    llm_response: str, 
    schema: Union[Type[BaseModel], Any], 
    allow_partial: bool = False
) -> Any:
    ''' Parse and validate JSON from an LLM response.
    
    This function:
    1. Normalizes the text to handle encoding issues
    2. Removes markdown code fences
    3. Extracts the JSON structure
    4. Parses and validates against the provided schema
    """
    # Normalize text to handle control characters
    cleaned_response = normalize_text(llm_response)'''
    
    # Remove markdown code fences (```json or ```)
    cleaned_response = re.sub(
        r"```(?:json)?", "", 
        cleaned_response, 
        flags=re.IGNORECASE
    ).replace("```", "").strip()

     # Find JSON structure (object or array)
    json_match = re.search(r'(\{.*\}|\[.*\])', cleaned_response, flags=re.DOTALL)
    if not json_match:
        raise ValueError(
            f"No valid JSON object or array found in response: {llm_response[:200]}..."
        )