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

    def __post_init__(self):
        # Validate configuration.
        self.probe_path = LOCAL_PROBES_DIR / self.probe_id

        if self.load_from == "hf" and not self.hf_repo_id:
            raise ValueError("hf_repo_id must be specified when load_from='hf'")
        
        if self.load_from in ['disk'] and not self.probe_path.exists():
            raise ValueError(f"Probe with ID {self.probe_id} not found in disk at path {self.probe_path}")

        if self.layer is None:
            # default to hooking the value head at the last layer of the underlying LM
            self.layer = get_num_layers(self.model_name) - 1

        if isinstance(self.lora_layers, str):
            if self.lora_layers == 'all':
                # default to training LoRA adaptors on all layers
                # (up to the layer where we hook the value head)
                self.lora_layers = list(range(0, self.layer + 1))
            elif self.lora_layers.lower() == 'none':
                self.lora_layers = []
            else:
                self.lora_layers = [int(layer) for layer in self.lora_layers.strip('[').strip(']').split(",")]
        elif self.lora_layers is None:
            self.lora_layers = []
        assert all(isinstance(l, int) for l in self.lora_layers)
@dataclass
class TrainingConfig:
    #Configuration for probe training.  
    wandb_project: str = "hallucination-probes"
    wandb_name: Optional[str] = None
    probe_config: ProbeConfig = field(default_factory=ProbeConfig)

    upload_to_hf: bool = False
    save_evaluation_metrics: bool = True
    save_roc_curves: bool = False
    dump_raw_eval_results: bool = False


       # Training hyperparameters
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    high_loss_threshold: Optional[float] = None  # Threshold for masking high-loss tokens
    lambda_lm: float = 0.0  # Weight for language modeling loss regularization
    lambda_kl: float = 0.0  # Weight for KL divergence regularization
    anneal_max_aggr: bool = True  # Whether to anneal span-level max aggregation loss
    anneal_warmup: float = 1.0  # Fraction of training for span loss warmup
    learning_rate: float = 5e-5  # Overall learning rate (deprecated)
    probe_head_lr: Optional[float] = 5e-3 # Separate LR for probe head
    lora_lr: Optional[float] = 5e-5  # Separate LR for LoRA parameters
    sparsity_penalty_weight: Optional[float] = None
    num_train_samples: Optional[int] = None  # Limit training samples
    max_steps: int = -1  # Override num_epochs if set
    num_train_epochs: int = 1
    enable_gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    eval_steps: Optional[int] = -1  # Only manually evaluate at the end
    evaluation_strategy: str = "no"  # "steps", "epoch", or "no"
    logging_steps: int = 10
    seed: int = 42

     # Dataset configuration
    train_datasets: List[dict] = field(default_factory=list)
    eval_datasets: List[dict] = field(default_factory=list)

    # These will be populated in __post_init__
    train_dataset_configs: List[TokenizedProbingDatasetConfig] = field(default_factory=list, init=False)
    eval_dataset_configs: List[TokenizedProbingDatasetConfig] = field(default_factory=list, init=False)
    