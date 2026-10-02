"""Hallucination Detection Probe package."""

from .config import ProbeConfig, TrainingConfig, EvaluationConfig
from .dataset import TokenizedProbingDataset, TokenizedProbingDatasetConfig
from .dataset_converters import get_prepare_function, DATASET_CONVERTERS
from .model import ProbeHead, MultiLayerProbeHead, ProbedModel
from .types import AnnotatedSpan, ProbingItem
from .inference import HallucinationDetector, DetectionResult, SpanResult

__all__ = [
    "ProbeConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "TokenizedProbingDataset",
    "TokenizedProbingDatasetConfig",
    "get_prepare_function",
    "DATASET_CONVERTERS",
    "ProbeHead",
    "MultiLayerProbeHead",
    "ProbedModel",
    "AnnotatedSpan",
    "ProbingItem",
    "HallucinationDetector",
    "DetectionResult",
    "SpanResult",
]
