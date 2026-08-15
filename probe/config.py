# Configuration classes for probe training.

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union, Literal

from utils.probe_loader import LOCAL_PROBES_DIR
from utils.model_utils import get_num_layers
from .dataset import TokenizedProbingDatasetConfig

@dataclass
class ProbeConfig:
    # Configuration for a probe model.
    probe_id: str = "llama3_1_8b_lora_lambda_kl=0.5"

    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    layer: Optional[int] = None  # Which layer to attach the probe to
     # LoRA configuration
    lora_layers: Optional[Union[List[int], str]] = "all"  # Which layers to apply LoRA to
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha scaling
    lora_dropout: float = 0.05  # LoRA dropout

    # Loading configuration
    load_from: Optional[Literal['disk', 'hf']] = None  # "disk", "hf", or None
    probe_path: Optional[Path] = None  # Local path for disk loading
    hf_repo_id: Optional[str] = "andyrdt/hallucination-probes"  # HuggingFace repository ID
    threshold: float = 0.5  # Classification threshold