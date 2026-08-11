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
