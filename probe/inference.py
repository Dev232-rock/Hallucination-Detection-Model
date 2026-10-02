"""High-level inference API for hallucination detection.

Usage (Python):
    from probe.inference import HallucinationDetector
    detector = HallucinationDetector.from_pretrained("llama3_1_8b_lora_lambda_kl=0.5")
    result = detector.detect(
        prompt="What is the capital of France?",
        completion="Paris is the capital of France and was founded in 200 BC by the Romans."
    )
    print(result)

Usage (CLI):
    python -m probe.inference \\
        --probe_id llama3_1_8b_lora_lambda_kl=0.5 \\
        --prompt "What is the capital of France?" \\
        --completion "Paris is the capital of France and was founded in 200 BC."
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer

from utils.model_utils import get_device, load_model_and_tokenizer
from utils.tokenization import find_assistant_tokens_slice
from utils.probe_loader import download_probe_from_hf

from .config import ProbeConfig
from .model import ProbedModel


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SpanResult:
    """A detected hallucinated span with its score."""
    text: str
    start_token: int
    end_token: int
    score: float          # mean hallucination probability over the span


@dataclass
class DetectionResult:
    """Full detection output for one (prompt, completion) pair."""
    prompt: str
    completion: str
    token_scores: List[float]       # per-token hallucination probability
    token_texts: List[str]          # decoded text for each token
    hallucinated_spans: List[SpanResult] = field(default_factory=list)
    is_hallucinated: bool = False   # True if any span exceeds threshold
    max_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "is_hallucinated": self.is_hallucinated,
            "max_score": self.max_score,
            "hallucinated_spans": [
                {
                    "text": s.text,
                    "start_token": s.start_token,
                    "end_token": s.end_token,
                    "score": s.score,
                }
                for s in self.hallucinated_spans
            ],
            "token_scores": self.token_scores,
        }

    def __str__(self) -> str:
        lines = [
            f"Hallucinated: {self.is_hallucinated}  (max score: {self.max_score:.3f})",
        ]
        if self.hallucinated_spans:
            lines.append("Detected spans:")
            for span in self.hallucinated_spans:
                lines.append(f"  [{span.score:.3f}] \"{span.text}\"")
        else:
            lines.append("No hallucinations detected.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class HallucinationDetector:
    """Friendly inference wrapper around a trained ProbedModel.

    Args:
        probed_model:  Trained ProbedModel.
        tokenizer:     Matching tokenizer.
        threshold:     Per-token probability above which a token is flagged.
        min_span_tokens: Minimum consecutive flagged tokens to form a span.
        device:        Torch device to run on.
    """

    def __init__(
        self,
        probed_model: ProbedModel,
        tokenizer: AutoTokenizer,
        threshold: float = 0.5,
        min_span_tokens: int = 1,
        device: Optional[torch.device] = None,
    ):
        self.model = probed_model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.min_span_tokens = min_span_tokens
        self.device = device or get_device()
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        probe_id: str,
        model_name: Optional[str] = None,
        load_from: str = "disk",
        hf_repo_id: str = "andyrdt/hallucination-probes",
        threshold: float = 0.5,
    ) -> "HallucinationDetector":
        """Load a detector from a saved probe (disk or HuggingFace Hub).

        Args:
            probe_id:   Identifier of the probe (directory name under value_head_probes/).
            model_name: Override the base model name (defaults to auto-detected).
            load_from:  ``"disk"`` or ``"hf"``.
            hf_repo_id: HuggingFace repo for ``load_from="hf"``.
            threshold:  Hallucination score threshold.
        """
        config = ProbeConfig(
            probe_id=probe_id,
            load_from=load_from,
            hf_repo_id=hf_repo_id,
        )
        if model_name:
            config.model_name = model_name

        if load_from == "hf":
            download_probe_from_hf(
                repo_id=hf_repo_id,
                probe_id=probe_id,
            )

        device = get_device()
        _, tokenizer = load_model_and_tokenizer(config.model_name)
        probed_model = ProbedModel.load(config=config, path=config.probe_path)

        return cls(
            probed_model=probed_model,
            tokenizer=tokenizer,
            threshold=threshold,
            device=device,
        )

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def detect(
        self,
        prompt: str,
        completion: str,
        return_token_scores: bool = True,
    ) -> DetectionResult:
        """Detect hallucinations in a (prompt, completion) pair.

        Args:
            prompt:             The user's question / context.
            completion:         The model's response to analyse.
            return_token_scores: If False, token_scores list is omitted for speed.

        Returns:
            A :class:`DetectionResult` with hallucinated spans and scores.
        """
        # Tokenise as a conversation
        conversation = [
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": completion},
        ]
        full_text = self.tokenizer.apply_chat_template(conversation, tokenize=False)
        if self.tokenizer.bos_token and self.tokenizer.bos_token in full_text:
            full_text = full_text.replace(self.tokenizer.bos_token, "")

        encoding = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Run model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probs = outputs["probe_probs"][0].cpu().float()  # (seq_len,)

        # Decode tokens
        seq_len = input_ids.shape[1]
        token_texts = [self.tokenizer.decode(input_ids[0, i]) for i in range(seq_len)]

        # Find where the assistant response starts
        input_str = self.tokenizer.decode(input_ids[0])
        assistant_slice = find_assistant_tokens_slice(input_ids[0], input_str, self.tokenizer)
        completion_start = assistant_slice.stop

        # Only score tokens in the completion
        completion_probs = probs[completion_start:].numpy()
        completion_tokens = token_texts[completion_start:]

        # Extract contiguous hallucinated spans
        hallucinated_spans = self._extract_spans(
            probs=completion_probs,
            tokens=completion_tokens,
            offset=completion_start,
        )

        max_score = float(completion_probs.max()) if len(completion_probs) > 0 else 0.0
        is_hallucinated = max_score >= self.threshold

        return DetectionResult(
            prompt=prompt,
            completion=completion,
            token_scores=completion_probs.tolist() if return_token_scores else [],
            token_texts=completion_tokens,
            hallucinated_spans=hallucinated_spans,
            is_hallucinated=is_hallucinated,
            max_score=max_score,
        )

    def detect_batch(
        self,
        pairs: List[Tuple[str, str]],
        **kwargs,
    ) -> List[DetectionResult]:
        """Run detect on a list of (prompt, completion) pairs."""
        return [self.detect(p, c, **kwargs) for p, c in pairs]

    # ------------------------------------------------------------------
    # Span extraction
    # ------------------------------------------------------------------

    def _extract_spans(
        self,
        probs: "np.ndarray",
        tokens: List[str],
        offset: int,
    ) -> List[SpanResult]:
        """Merge consecutive above-threshold tokens into spans."""
        import numpy as np

        flagged = probs >= self.threshold
        spans: List[SpanResult] = []
        i = 0
        while i < len(flagged):
            if flagged[i]:
                j = i
                while j < len(flagged) and flagged[j]:
                    j += 1
                if (j - i) >= self.min_span_tokens:
                    span_text = "".join(tokens[i:j])
                    span_score = float(np.mean(probs[i:j]))
                    spans.append(SpanResult(
                        text=span_text,
                        start_token=offset + i,
                        end_token=offset + j - 1,
                        score=span_score,
                    ))
                i = j
            else:
                i += 1
        return spans

    # ------------------------------------------------------------------
    # Pretty display
    # ------------------------------------------------------------------

    def highlight(self, result: DetectionResult) -> str:
        """Return the completion with hallucinated spans wrapped in [[ ]]."""
        text = result.completion
        for span in sorted(result.hallucinated_spans, key=lambda s: -s.start_token):
            text = text.replace(span.text.strip(), f"[[{span.text.strip()}]]", 1)
        return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect hallucinations in text")
    parser.add_argument("--probe_id",   type=str, required=True)
    parser.add_argument("--prompt",     type=str, required=True)
    parser.add_argument("--completion", type=str, required=True)
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--load_from",  type=str, default="disk", choices=["disk", "hf"])
    parser.add_argument("--hf_repo_id", type=str, default="andyrdt/hallucination-probes")
    parser.add_argument("--json",       action="store_true", help="Output JSON instead of human-readable")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    detector = HallucinationDetector.from_pretrained(
        probe_id=args.probe_id,
        load_from=args.load_from,
        hf_repo_id=args.hf_repo_id,
        threshold=args.threshold,
    )

    result = detector.detect(prompt=args.prompt, completion=args.completion)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("\n" + "=" * 60)
        print(result)
        print("\nHighlighted completion:")
        print(detector.highlight(result))
        print("=" * 60)
