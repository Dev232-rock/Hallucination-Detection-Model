# Utilities for downloading and uploading probe models to/from HuggingFace Hub

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from huggingface_hub import HfApi, hf_hub_download, login
from huggingface_hub.utils import validate_repo_id

# Default directory for saving probes locally
LOCAL_PROBES_DIR = Path(__file__).parent.parent / "value_head_probes"

def download_probe_from_hf(
    repo_id: str,
    probe_id: Optional[str] = None,
    local_folder: Optional[Union[str, Path]] = None,
    hf_repo_subfolder_prefix: str = "",
    token: Optional[str] = None
) -> None:
    # Simplified probe download function for Modal.
    api = HfApi()

    if local_folder is None:
        local_folder = LOCAL_PROBES_DIR / probe_id