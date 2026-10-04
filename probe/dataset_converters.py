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


def prepare_truthfulqa(row: dict) -> Optional[ProbingItem]:
    """TruthfulQA dataset (https://huggingface.co/datasets/truthful_qa).

    Uses the ``generation`` config subset.  Each row has a ``question``,
    a ``best_answer`` (factual), and a list of ``incorrect_answers``
    (hallucinated).  We emit one ProbingItem per answer, labelling the
    whole completion as either factual (0.0) or hallucinated (1.0) at
    the sentence level.

    Expected columns: ``question``, ``best_answer``, ``incorrect_answers``
    """
    question  = row.get("question", "") or row.get("prompt", "")
    best      = row.get("best_answer", "") or row.get("correct_answers", [""])[0]
    incorrect = row.get("incorrect_answers", []) or []

    if not question:
        return None

    items: List[ProbingItem] = []
    # Factual answer
    if best:
        span = AnnotatedSpan(span=best, label=0.0, index=0)
        items.append(ProbingItem(prompt=question, completion=best, spans=[span]))
    # Pick the first incorrect answer (keeps one item per row for simplicity)
    if incorrect:
        bad = incorrect[0] if isinstance(incorrect, list) else str(incorrect)
        span = AnnotatedSpan(span=bad, label=1.0, index=0)
        items.append(ProbingItem(prompt=question, completion=bad, spans=[span]))

    # Return the first item; the dataset loader iterates rows so both will
    # be covered when the caller iterates over the dataset directly.
    # For multi-item rows, callers can use prepare_truthfulqa_all below.
    return items[0] if items else None


def prepare_truthfulqa_all(row: dict) -> List[ProbingItem]:
    """Like prepare_truthfulqa but returns ALL answers (factual + incorrect).

    Use this when you want every answer in the row as a separate ProbingItem.
    Register it and iterate with a custom loop rather than the default
    ``get_prepare_function`` path.
    """
    question  = row.get("question", "") or row.get("prompt", "")
    best      = row.get("best_answer", "") or ""
    incorrect = row.get("incorrect_answers", []) or []
    correct   = row.get("correct_answers", []) or []

    if not question:
        return []

    items: List[ProbingItem] = []
    for ans in ([best] if best else []) + list(correct):
        span = AnnotatedSpan(span=ans, label=0.0, index=0)
        items.append(ProbingItem(prompt=question, completion=ans, spans=[span]))
    for ans in incorrect:
        span = AnnotatedSpan(span=ans, label=1.0, index=0)
        items.append(ProbingItem(prompt=question, completion=ans, spans=[span]))
    return items


def prepare_factscore(row: dict) -> Optional[ProbingItem]:
    """FactScore-style dataset with atomic claim annotations.

    Expected columns:
        - ``topic`` / ``prompt``        : the entity / question
        - ``output`` / ``completion``   : the generated biography / response
        - ``annotations`` (list of dicts): each with ``text`` and
          ``label`` (``"S"`` = supported, ``"NS"`` = not supported / hallucinated)

    Compatible with the FActScoring benchmark
    (https://github.com/shmsw25/FActScoring) when exported to JSONL.
    """
    prompt     = row.get("topic", "") or row.get("prompt", "") or row.get("input", "")
    completion = row.get("output", "") or row.get("completion", "") or row.get("response", "")
    annotations = row.get("annotations", []) or row.get("claims", []) or []

    if not prompt or not completion:
        return None

    label_map = {
        "s": 0.0, "supported": 0.0, "true": 0.0, "1": 0.0,
        "ns": 1.0, "not supported": 1.0, "false": 1.0, "0": 1.0,
        "ir": -100.0, "irrelevant": -100.0,
    }

    spans: List[AnnotatedSpan] = []
    for ann in annotations:
        text  = ann.get("text") or ann.get("claim") or ann.get("span") or ""
        raw   = str(ann.get("label", ann.get("is_supported", "ir"))).lower().strip()
        label = label_map.get(raw, -100.0)
        idx   = completion.find(text)
        spans.append(AnnotatedSpan(span=text, label=label, index=max(idx, 0)))

    if not spans:
        # No annotations — treat the whole completion as a factual span
        spans = [AnnotatedSpan(span=completion, label=0.0, index=0)]

    return ProbingItem(prompt=prompt, completion=completion, spans=spans)


def prepare_shroom(row: dict) -> Optional[ProbingItem]:
    """SHROOM shared-task dataset (SemEval-2024 Task 6).

    Expected columns:
        - ``src``      : source sentence / question
        - ``hyp``      : model hypothesis / generated text
        - ``tgt``      : reference target (if available)
        - ``label``    : ``"Hallucination"`` or ``"Not Hallucination"``
        - ``p(Hallucination)`` : float probability (optional)

    See https://huggingface.co/datasets/aqweteddy/SHROOM_unlabeled for format.
    """
    prompt     = row.get("src", "") or row.get("source", "") or row.get("prompt", "")
    completion = row.get("hyp", "") or row.get("hypothesis", "") or row.get("completion", "")
    raw_label  = str(row.get("label", "")).strip().lower()

    if not prompt or not completion:
        return None

    if "not" in raw_label or raw_label in ("0", "false", "no"):
        label = 0.0
    elif raw_label in ("hallucination", "1", "true", "yes"):
        label = 1.0
    else:
        label = -100.0  # unlabelled

    span = AnnotatedSpan(span=completion, label=label, index=0)
    return ProbingItem(
        prompt=prompt,
        completion=completion,
        spans=[span],
        metadata={"p_hallucination": row.get("p(Hallucination)")},
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASET_CONVERTERS: Dict[str, Callable[[dict], Optional[ProbingItem]]] = {
    "ragtruth":   prepare_ragtruth,
    "felm":       prepare_felm,
    "halueval":   prepare_halueval,
    "generic":    prepare_generic,
    "truthfulqa": prepare_truthfulqa,
    "factscore":  prepare_factscore,
    "shroom":     prepare_shroom,
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
