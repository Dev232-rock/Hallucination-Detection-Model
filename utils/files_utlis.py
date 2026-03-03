"""File I/O utilities."""

import json
import yaml
import numpy as np
import torch
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import os

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
