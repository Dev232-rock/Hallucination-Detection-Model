"""Model loading and setup utilities."""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

try:
    from peft import LoraConfig, PeftModel, get_peft_model
    HAS_PEFT = True
except ImportError:
    PeftModel = type("PeftModel", (), {})
    LoraConfig = None
    get_peft_model = None
    HAS_PEFT = False


def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU)."""
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
    """Load a model and tokenizer from HuggingFace."""
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
        padding_side="right",
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
    """Setup a model with LoRA adapters."""
    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
    )
    if lora_weights_path:
        model = PeftModel.from_pretrained(model, lora_weights_path)
    else:
        model = get_peft_model(model, peft_config)

    return model


def get_model_layers(model: PreTrainedModel) -> List[nn.Module]:
    """Get the list of transformer layers from a model."""
    if isinstance(model, PeftModel):
        base_model = model.get_base_model()
    else:
        base_model = model

    if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        return list(base_model.model.layers)
    elif hasattr(base_model, "transformer") and hasattr(base_model.transformer, "h"):
        return list(base_model.transformer.h)
    elif hasattr(base_model, "encoder") and hasattr(base_model.encoder, "layer"):
        return list(base_model.encoder.layer)
    elif hasattr(base_model, "gpt_neox") and hasattr(base_model.gpt_neox, "layers"):
        return list(base_model.gpt_neox.layers)
    else:
        raise ValueError(f"Unknown model architecture: {type(base_model)}")


def get_num_layers(model_or_name: Union[str, PreTrainedModel]) -> int:
    """Get the number of transformer layers in a model."""
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
            raise ValueError(
                f"Model {model_or_name} not supported. Please add it to the model_layers_map."
            )

    return len(get_model_layers(model_or_name))


def get_model_layers_prefix(model: PreTrainedModel) -> str:
    """Get the prefix path to the model layers."""
    if isinstance(model, PeftModel):
        base_model = model.get_base_model()
    else:
        base_model = model

    if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        return "model.layers"
    elif hasattr(base_model, "transformer") and hasattr(base_model.transformer, "h"):
        return "transformer.h"
    elif hasattr(base_model, "encoder") and hasattr(base_model.encoder, "layer"):
        return "encoder.layer"
    elif hasattr(base_model, "gpt_neox") and hasattr(base_model.gpt_neox, "layers"):
        return "gpt_neox.layers"
    else:
        raise ValueError(f"Unknown model architecture: {type(base_model)}")


def get_model_hidden_size(model: PreTrainedModel) -> int:
    """Get the hidden size of a transformer model."""
    if isinstance(model, PeftModel):
        base_model = model.get_base_model()
    else:
        base_model = model
    if hasattr(base_model, "config"):
        config = base_model.config
        for attr in ["hidden_size", "d_model", "n_embd", "embed_dim"]:
            if hasattr(config, attr):
                return getattr(config, attr)
    if hasattr(base_model, "model") and hasattr(base_model.model, "embed_tokens"):
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
    """Setup LoRA adapters for specific layers in a model."""
    if not layer_indices:
        print("No LoRA layers specified, returning base model")
        return model

    layer_prefix = get_model_layers_prefix(model)
    target_modules = []
    module_suffixes = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]

    for layer_idx in layer_indices:
        for module_suffix in module_suffixes:
            target_modules.append(f"{layer_prefix}.{layer_idx}.{module_suffix}")

    print(f"Creating LoRA adapters for layers {layer_indices}...")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    return get_peft_model(model, lora_config)


def print_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    """Print information about trainable parameters in a model."""
    trainable_params = 0
    total_params = 0

    print("Parameters that will be trained:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"  - {name}: shape {param.shape}, device {param.device}")
        total_params += param.numel()

    trainable_params_percentage = 100 * trainable_params / total_params
    print(f"\nTotal trainable parameters: {trainable_params:,} ({trainable_params_percentage:.2f}%)")
    print(f"Total parameters: {total_params:,}")
    return trainable_params, total_params