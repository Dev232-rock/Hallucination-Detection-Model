"""Dataset converter functions that transform raw HuggingFace dataset rows
into ProbingItem instances ready for tokenization and training.

Each converter is keyed by ``dataset_id`` (the string used in
``TokenizedProbingDatasetConfig.dataset_id``).  New datasets should be added
by:
1. Writing a ``prepare_<name>(row: dict) -> ProbingItem`` function.
2. Registering it in ``DATASET_CONVERTERS`` below.
3. Registering it in ``SUPPORTED_DATASETS`` if it needs special HF loading.
"""

from typing import Callable, Dict, List, Optional

from .types import AnnotatedSpan, ProbingItem

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_spans(
    annotated_list: List[dict],
    completion: str,
) -> List[AnnotatedSpan]:
    """Convert a list of annotation dicts into AnnotatedSpan objects.

    Each dict is expected to contain at minimum:
        - ``text``  (str)  – the span text
        - ``label`` (float | int | str) – 1.0/0.0/−100 or 'hallucinated'/'factual'

    The ``index`` field is derived by searching for the span in the completion.
    """
    label_map = {
        "hallucinated": 1.0,
        "non_factual": 1.0,
        "factual": 0.0,
        "supported": 0.0,
        "na": -100.0,
        "n/a": -100.0,
        "ignore": -100.0,
    }

    spans: List[AnnotatedSpan] = []
    for ann in annotated_list:
        span_text = ann.get("text") or ann.get("span") or ""
        raw_label = ann.get("label", -100)

        if isinstance(raw_label, str):
            label = label_map.get(raw_label.lower().strip(), -100.0)
        else:
            label = float(raw_label)

        # Locate this span within the completion
        idx = completion.find(span_text)
        spans.append(AnnotatedSpan(span=span_text, label=label, index=max(idx, 0)))

    return spans


# ---------------------------------------------------------------------------
# Per-dataset prepare functions
# ---------------------------------------------------------------------------

def prepare_ragtruth(row: dict) -> Optional[ProbingItem]:
    """RAGTruth dataset (https://huggingface.co/datasets/wandb/RAGTruth).

    Expected columns: ``source_info``, ``response``, ``labels``
    """
    prompt = row.get("source_info", "") or row.get("prompt", "")
    completion = row.get("response", "") or row.get("completion", "")
    raw_labels = row.get("labels", []) or []

    if not prompt or not completion:
        return None

    spans = _build_spans(raw_labels, completion)
    return ProbingItem(prompt=prompt, completion=completion, spans=spans)


def prepare_felm(row: dict) -> Optional[ProbingItem]:
    """FELM factual error localisation dataset.

    Expected columns: ``prompt``, ``completion``, ``annotations``
    """
    prompt = row.get("prompt", "")
    completion = row.get("completion", "") or row.get("response", "")
    annotations = row.get("annotations", []) or []

    if not prompt or not completion:
        return None

    spans = _build_spans(annotations, completion)
    return ProbingItem(prompt=prompt, completion=completion, spans=spans)


def prepare_halueval(row: dict) -> Optional[ProbingItem]:
    """HaluEval dataset with sentence-level hallucination flags.

    Expected columns: ``question``, ``answer``, ``hallucination``
    """
    prompt = row.get("question", "") or row.get("prompt", "")
    completion = row.get("answer", "") or row.get("completion", "")
    is_hallucinated = row.get("hallucination", "no")

    if not prompt or not completion:
        return None

    label = 1.0 if str(is_hallucinated).lower() in ("yes", "true", "1") else 0.0
    span = AnnotatedSpan(span=completion, label=label, index=0)
    return ProbingItem(prompt=prompt, completion=completion, spans=[span])


def prepare_generic(row: dict) -> Optional[ProbingItem]:
    """Fallback converter for datasets that already conform to the ProbingItem
    schema (i.e., have ``prompt``, ``completion``, and ``spans`` columns).
    """
    prompt = row.get("prompt", "")
    completion = row.get("completion", "") or row.get("response", "")
    raw_spans = row.get("spans", []) or []

    if not prompt or not completion:
        return None

    spans = _build_spans(raw_spans, completion)
    return ProbingItem(prompt=prompt, completion=completion, spans=spans)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASET_CONVERTERS: Dict[str, Callable[[dict], Optional[ProbingItem]]] = {
    "ragtruth": prepare_ragtruth,
    "felm": prepare_felm,
    "halueval": prepare_halueval,
    "generic": prepare_generic,
}


def get_prepare_function(dataset_id: str) -> Callable[[dict], Optional[ProbingItem]]:
    """Return the prepare function for the given dataset_id.

    Falls back to ``prepare_generic`` if the dataset is not explicitly
    registered, so new datasets can be tested without modifying this file.
    """
    key = dataset_id.lower().strip()
    if key not in DATASET_CONVERTERS:
        import warnings
        warnings.warn(
            f"No dedicated converter found for dataset '{dataset_id}'. "
            "Falling back to 'generic' converter. Make sure your dataset "
            "has 'prompt', 'completion', and 'spans' columns.",
            UserWarning,
            stacklevel=2,
        )
        return DATASET_CONVERTERS["generic"]
    return DATASET_CONVERTERS[key]
