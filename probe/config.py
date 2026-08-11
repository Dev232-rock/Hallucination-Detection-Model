# Configuration classes for probe training.

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union, Literal

from utils.probe_loader import LOCAL_PROBES_DIR
from utils.model_utils import get_num_layers
from .dataset import TokenizedProbingDatasetConfig
