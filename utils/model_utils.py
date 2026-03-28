"""Model loading and setup utilities."""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel


def get_device() -> torch.device:
    'Get the best available device (CUDA, MPS, or CPU).'
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
def load_model_and_tokenizer(
    model_name: str,
    device_map: Optional[Union[str, dict]] = "auto",
    torch_dtype: Optional[torch.dtype] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    # Load a model and tokenizer from HuggingFace.
     # Set default dtype
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16