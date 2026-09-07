# Tokenized dataset classes with token-level labels for probe training.
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import torch
import datasets
from jaxtyping import Float, Int
from termcolor import colored
from torch import Tensor
from torch.utils.data import Dataset

rom tqdm import tqdm
from transformers import AutoTokenizer

from utils.tokenization import find_assistant_tokens_slice, find_string_in_tokens, slice_to_list
from .types import AnnotatedSpan, ProbingItem
from .dataset_converters import get_prepare_function
@dataclass
class TokenizedProbingDatasetConfig:
    # Configuration for tokenizing and labeling a probing dataset at token level\
    dataset_id: str             
    hf_repo: str
    subset: Optional[str] = None
    split: str = "train"
    max_length: int = 2048
    ignore_buffer: int = 0  # Buffer around spans to ignore
                                                                                                                                       