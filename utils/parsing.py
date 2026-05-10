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
    # Parse and validate JSON from an LLM response.
    
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
    json_str = json_match.group(0).strip()

    try:
        # Parse JSON
        parsed = from_json(json_str, allow_partial=allow_partial)
        
        # Validate against schema
        validated_data = parse_obj_as(schema, parsed)
        
        return validated_data
    except Exception as e:
        raise ValueError(
            f"Error parsing/validating JSON: {e}\n"
            f"JSON string: {json_str[:200]}..."
        ) from e
def validate_dicts_to_pydantic(
    dicts: List[dict],
    model: Type[T],
    skip_invalid: bool = False
) -> List[T]:
    # Validate a list of dictionaries against a Pydantic model. 
     validated = []

     for i, item_dict in enumerate(dicts):
        try:
            validated_item = model.model_validate(item_dict)
            validated.append(validated_item)
        except Exception as e:
            if skip_invalid:
                # Silently skip invalid items
                continueexcept Exception as e:
            if skip_invalid:
                # Silently skip invalid items
                continue