# Utilities for downloading and uploading probe models to/from HuggingFace Hub

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from huggingface_hub import HfApi, hf_hub_download, login
from huggingface_hub.utils import validate_repo_id