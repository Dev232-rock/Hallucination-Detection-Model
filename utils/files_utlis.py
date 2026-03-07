"""File I/O utilities."""

import json
import yaml
import numpy as np
import torch
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import os

# Convert non-serializable objects to JSON-serializable types to  Handles common ML types like numpy arrays, torch tensors, dtypes, etc 
def default_serializer(obj: Any) -> Any:
    if isinstance(obj, torch.dtype):
        return str(obj)
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.cpu().tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, "__dataclass_fields__"):
        return {k: default_serializer(v) for k, v in obj.__dict__.items()}
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return obj
#  Converts a dataclass to a dictionary. Will recurse through lists, dicts, and nested dataclasses.
def dataclass_to_dict(obj) -> dict:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [dataclass_to_dict(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    else:
        return obj
# Converts a Pydantic model to a dictionary using model_dump()
def pydantic_to_dict(obj) -> dict:
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif isinstance(obj, list):
        return [pydantic_to_dict(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: pydantic_to_dict(v) for k, v in obj.items()}
    else:
        return obj

def make_directory_wrapped(filepath: str, **kwargs) -> None:
    if isinstance(filepath, Path):
        parent_dir = filepath.parent
    else:
        parent_dir = "/".join(filepath.split("/")[:-1])
    os.makedirs(parent_dir, exist_ok=True, **kwargs)

def save_jsonl(
    dict_list: List[Dict[Any, Any]],
    filepath: str,
    append: bool = False,
    serialize_dataclasses: bool = False,
    serialize_pydantic: bool = False,
) -> None:
# Write a list of dictionaries to a jsonlines file.
    make_directory_wrapped(filepath)

    if isinstance(dict_list, dict):
        dict_list = [dict_list]

    if append and (not os.path.exists(filepath)):
        print(f"File {filepath} does not exist, setting append to False")
        append = False

    mode = "a" if append else "w"
    with open(filepath, mode=mode) as file:
        for item in dict_list:
            if serialize_dataclasses:
                item = dataclass_to_dict(item)
            elif serialize_pydantic:
                item = pydantic_to_dict(item)
            file.write(json.dumps(item) + "\n")
def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:

    #Load data from a JSON file
    path = Path(path)
    data = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data