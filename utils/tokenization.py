from typing import List, Optional

import torch
from jaxtyping import Int
from torch import Tensor
from transformers import AutoTokenizer
def find_string_in_tokens(target: str, tokens: Tensor, tokenizer: AutoTokenizer, max_iters: int = 100) -> slice:
    # Performs a binary search to look for a target string inside some tokens.
    assert target in tokenizer.decode(tokens), "The target isn't contained in the whole array of tokens"
    # Binary search over the end index of the slice
    n_iters = max_iters
    end_idx_left, end_idx_right = 0, len(tokens) 
    while end_idx_left != end_idx_right and n_iters > 0:
        mid = (end_idx_left + end_idx_right) // 2
        if target in tokenizer.decode(tokens[:mid]):
            end_idx_right = mid
        else:
            end_idx_left = mid + 1
        n_iters -= 1
    end_idx = end_idx_left

    # Binary search over the start index of the slice
    n_iters = max_iters
    start_idx_left, start_idx_right = 0, end_idx-1 
    while start_idx_left != start_idx_right and n_iters > 0:
         mid = (start_idx_left + start_idx_right + 1) // 2
        if target in tokenizer.decode(tokens[mid:end_idx]):
            start_idx_left = mid
        else:
            start_idx_right = mid-1
        n_iters -= 1
    start_idx = start_idx_left