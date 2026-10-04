"""Hallucination Detection Probe package."""

from .config import ProbeConfig, TrainingConfig, EvaluationConfig
from .dataset import TokenizedProbingDataset, TokenizedProbingDatasetConfig, StreamingProbingDataset
from .dataset_converters import (
    get_prepare_function,
    DATASET_CONVERTERS,
    prepare_truthfulqa,
    prepare_truthfulqa_all,
    prepare_factscore,
    prepare_shroom,
)
from .model import ProbeHead, MultiLayerProbeHead, ProbedModel
from .types import AnnotatedSpan, ProbingItem
from .inference import HallucinationDetector, DetectionResult, SpanResult, SentenceResult

__all__ = [
    # Config
    "ProbeConfig",
    "TrainingConfig",
    "EvaluationConfig",
    # Dataset
    "TokenizedProbingDataset",
    "TokenizedProbingDatasetConfig",
    "StreamingProbingDataset",
    "get_prepare_function",
    "DATASET_CONVERTERS",
    "prepare_truthfulqa",
    "prepare_truthfulqa_all",
    "prepare_factscore",
    "prepare_shroom",
    # Model
    "ProbeHead",
    "MultiLayerProbeHead",
    "ProbedModel",
    # Types
    "AnnotatedSpan",
    "ProbingItem",
    # Inference
    "HallucinationDetector",
    "DetectionResult",
    "SpanResult",
    "SentenceResult",
]
