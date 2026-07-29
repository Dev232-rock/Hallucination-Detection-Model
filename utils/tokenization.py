from typing import List, Optional

import torch
from jaxtyping import Int
from torch import Tensor
from transformers import AutoTokenizer
def find_string_in_tokens(target: str, tokens: Tensor, tokenizer: AutoTokenizer, max_iters: int = 100) -> slice:
    # Performs a binary search to look for a target string inside some tokens.
    assert target in tokenizer.decode(tokens), "The target isn't contained in the whole array of tokens"