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

         # Set default dtype
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='right'
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer
def setup_model_with_lora(
    model: AutoModelForCausalLM,
    lora_config: dict,
    lora_weights_path: Optional[str] = None,
) -> PeftModel:
    # Setup a model with LoRA adapters and Create LoRA configuration
    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
    )
  # Apply LoRA to model
    if lora_weights_path:
        # Load pre-trained LoRA weights
        model = PeftModel.from_pretrained(model, lora_weights_path)
    else:
        # Initialize new LoRA adapters
        model = get_peft_model(model, peft_config)
    
    return model

    def get_model_layers(model: PreTrainedModel) -> List[nn.Module]:
    # Get the list of transformer layers from a model.
    # Handle PeftModel by getting the base model
    if isinstance(model, PeftModel):
        base_model = model.get_base_model()
    else:
        base_model = model


     # Common patterns for accessing layers in different model architectures
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
        # LLaMA, Mistral, etc.
        return list(base_model.model.layers)
    elif hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'h'):
        # GPT-2, GPT-J, etc.
        return list(base_model.transformer.h)
    elif hasattr(base_model, 'encoder') and hasattr(base_model.encoder, 'layer'):
        # BERT, RoBERTa, etc.
        return list(base_model.encoder.layer)
    elif hasattr(base_model, 'gpt_neox') and hasattr(base_model.gpt_neox, 'layers'):
        # GPT-NeoX
        return list(base_model.gpt_neox.layers)
    else:
        raise ValueError(f"Unknown model architecture: {type(base_model)}")