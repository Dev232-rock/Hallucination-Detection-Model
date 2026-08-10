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

     target_slice = slice(start_idx, end_idx)

     if target not in tokenizer.decode(tokens[target_slice]):
        raise ValueError(f"Failed to find {target} in tokens: {[tokenizer.decode([tok]) for tok in tokens]}")
    return target_slice
def find_assistant_tokens_slice(
    input_ids: Int[Tensor, "seq_len"], 
    input_str: str, 
    tokenizer: AutoTokenizer
) -> slice:
     # Find the slice of tokens that marks the start of the assistant's response.
     eot_tokens = [
        '<|eot_id|><|start_header_id|>assistant<|end_header_id|>',  # llama 3.1 end-of-turn tokens
        '<|im_start|>assistant',  # qwen end-of-turn tokens
        '<start_of_turn>model',  # gemma end-of-turn tokens
        "[/INST]",  # mistral end-of-turn tokens
    ] 
    for eot_token in eot_tokens:
     if eot_token in input_str:
          try:
              return find_string_in_tokens(eot_token, input_ids, tokenizer)
          except (AssertionError, ValueError):
                continue
    print(f"Could not find assistant tokens in the input_ids: {input_str[:100]}...")
    return slice(0, 0)
def slice_to_list(slice_obj: slice, length: Optional[int] = None) -> List[int]:
    # Convert a slice object to a list of indices.
    start, stop, step = slice_obj.start, slice_obj.stop, slice_obj.step

    # If length is not provided, use stop if it's not None, else raise an error
    if length is None:
        if stop is not None:
            length = stop
         else:
            raise ValueError("Length must be provided if stop is None")