"""Core data types for the hallucination detection probe system."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AnnotatedSpan:
    """A text span with a hallucination label.

    Attributes:
        span:   The text content of the span.
        label:  1.0 = hallucinated, 0.0 = factual, -100 = ignore/N/A.
        index:  Character index of the span within the completion text (used
                for ordering spans during tokenization).
    """
    span: str
    label: float          # 1.0 = hallucinated, 0.0 = factual, -100 = ignore
    index: int = 0        # character offset in the completion


@dataclass
class ProbingItem:
    """A single training/evaluation example for the hallucination probe.

    Attributes:
        prompt:     The user's prompt / question.
        completion: The model's response (may contain hallucinations).
        spans:      Annotated text spans within the completion.
        metadata:   Optional dict for storing extra dataset-specific info.
    """
    prompt: str
    completion: str
    spans: List[AnnotatedSpan] = field(default_factory=list)
    metadata: Optional[dict] = None
