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
    
    def get_num_layers(model_or_name: Union[str, PreTrainedModel]) -> int:
    # Get the number of transformer layers in a model\.
    # If it's a string (model name), use the predefined mapping
    if isinstance(model_or_name, str):
        model_layers_map = {
            "meta-llama/Meta-Llama-3.1-8B-Instruct": 32,
            "meta-llama/Meta-Llama-3.1-70B-Instruct": 80,
            "meta-llama/Meta-Llama-3.1-405B-Instruct": 126,
            "google/gemma-2-2b-it": 26,
            "google/gemma-2-9b-it": 42,
            "google/gemma-2-27b-it": 46,
            "Qwen/Qwen2.5-0.5B-Instruct": 24,
            "Qwen/Qwen2.5-1.5B-Instruct": 28,
            "Qwen/Qwen2.5-3B-Instruct": 36,
            "Qwen/Qwen2.5-7B-Instruct": 28,
            "Qwen/Qwen2.5-14B-Instruct": 48,
            "Qwen/Qwen2.5-32B-Instruct": 64,
            "meta-llama/Llama-3.3-70B-Instruct": 80,
            "mistralai/Mistral-Small-24B-Instruct-2501": 40,
        }
        if model_or_name in model_layers_map:
            return model_layers_map[model_or_name]
        else:
            raise ValueError(f"Model {model_or_name} not supported. Please add it to the model_layers_map.")
    
    # If it's a model instance, count the layers
    return len(get_model_layers(model_or_name))


    def get_model_layers_prefix(model: PreTrainedModel) -> str:
        #Get the prefix path to the model layers.
    # Handle PeftModel
        if isinstance(model, PeftModel):
            base_model = model.get_base_model()
        else:
            base_model = model

    if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
        return "model.layers"
    elif hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'h'):
        return "transformer.h"
    elif hasattr(base_model, 'encoder') and hasattr(base_model.encoder, 'layer'):
        return "encoder.layer"
    elif hasattr(base_model, 'gpt_neox') and hasattr(base_model.gpt_neox, 'layers'):
        return "gpt_neox.layers"
    else:
        raise ValueError(f"Unknown model architecture: {type(base_model)}")

def get_model_hidden_size(model: PreTrainedModel) -> int:
    # Get the hidden size of a transformer model.
    # Handle PeftModel
    if isinstance(model, PeftModel):
        base_model = model.get_base_model()
    else:
        base_model = model
    if hasattr(base_model, 'config'):
        config = base_model.config
        # Try common attribute names
        for attr in ['hidden_size', 'd_model', 'n_embd', 'embed_dim']:
            if hasattr(config, attr):
                return getattr(config, attr)
     # If we can't find it in config, try to infer from the model structure
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'embed_tokens'):
        return base_model.model.embed_tokens.weight.shape[1]
    
    raise ValueError(f"Could not determine hidden size for model type {type(base_model)}")     

def setup_lora_for_layers(
    model: PreTrainedModel,
    layer_indices: List[int],
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    bias: str = "none",
) -> Union[PeftModel, PreTrainedModel]:       
    #  Setup LoRA adapters for specific layers in a model.