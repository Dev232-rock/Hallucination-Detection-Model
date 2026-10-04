"""High-level inference API for hallucination detection with multi-signal ensemble scoring.

Usage (Python):
    from probe.inference import HallucinationDetector
    detector = HallucinationDetector.from_pretrained("llama3_1_8b_lora_lambda_kl=0.5")
    result = detector.detect(
        prompt="What is the capital of France?",
        completion="Paris is the capital of France and was founded in 200 BC by the Romans."
    )
    print(result)
    print(detector.explain_prediction(result))

Usage (CLI):
    python -m probe.inference \\
        --probe_id llama3_1_8b_lora_lambda_kl=0.5 \\
        --prompt "What is the capital of France?" \\
        --completion "Paris is the capital of France and was founded in 200 BC." \\
        --scoring_mode ensemble --explain
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from utils.model_utils import get_device, load_model_and_tokenizer
from utils.tokenization import find_assistant_tokens_slice
from utils.probe_loader import download_probe_from_hf

from .config import ProbeConfig
from .model import ProbedModel, MultiSignalScorer


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SpanResult:
    """A detected hallucinated span with multi-signal diagnostic scores."""
    text: str
    start_token: int
    end_token: int
    score: float                      # active decision score (e.g., ensemble probability)
    probe_score: float = 0.0          # internal latent representation probability
    entropy_score: float = 0.0        # next-token predictive uncertainty
    attention_score: float = 0.0      # attention dispersion / context decoupling
    dominant_signal: str = "ensemble" # which signal contributed most to the detection
    category: str = "general"         # taxonomy category (e.g., entity_fabrication, context_drift)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "score": round(float(self.score), 4),
            "probe_score": round(float(self.probe_score), 4),
            "entropy_score": round(float(self.entropy_score), 4),
            "attention_score": round(float(self.attention_score), 4),
            "dominant_signal": self.dominant_signal,
            "category": self.category,
        }


@dataclass
class SentenceResult:
    """Aggregated multi-signal hallucination scores for one sentence in the completion."""
    text: str                          # sentence text
    score: float                       # max active score within the sentence
    mean_score: float                  # mean active score within the sentence
    is_hallucinated: bool              # True if max score >= detector threshold
    probe_score: float = 0.0           # mean probe probability in this sentence
    entropy_score: float = 0.0         # mean predictive entropy in this sentence
    attention_score: float = 0.0       # mean attention dispersion in this sentence
    dominant_signal: str = "ensemble"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "score": round(float(self.score), 4),
            "mean_score": round(float(self.mean_score), 4),
            "is_hallucinated": bool(self.is_hallucinated),
            "probe_score": round(float(self.probe_score), 4),
            "entropy_score": round(float(self.entropy_score), 4),
            "attention_score": round(float(self.attention_score), 4),
            "dominant_signal": self.dominant_signal,
        }


@dataclass
class DetectionResult:
    """Full multi-signal detection output for one (prompt, completion) pair."""
    prompt: str
    completion: str
    token_scores: List[float]                       # active decision scores per token
    token_texts: List[str]                          # decoded token strings
    hallucinated_spans: List[SpanResult] = field(default_factory=list)
    sentence_scores: List[SentenceResult] = field(default_factory=list)
    is_hallucinated: bool = False
    max_score: float = 0.0

    # Multi-signal breakdowns
    token_probe_scores: List[float] = field(default_factory=list)
    token_entropy_scores: List[float] = field(default_factory=list)
    token_attention_scores: List[float] = field(default_factory=list)
    token_ensemble_scores: List[float] = field(default_factory=list)
    layer_attribution: Optional[Dict[str, List[float]]] = None

    scoring_mode: str = "ensemble"
    signal_weights: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def hallucination_score(self) -> float:
        """Alias for max_score for backwards compatibility."""
        return self.max_score

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "is_hallucinated": self.is_hallucinated,
            "max_score": round(float(self.max_score), 4),
            "scoring_mode": self.scoring_mode,
            "signal_weights": self.signal_weights,
            "hallucinated_spans": [s.to_dict() for s in self.hallucinated_spans],
            "sentence_scores": [s.to_dict() for s in self.sentence_scores],
            "token_scores": [round(float(s), 4) for s in self.token_scores],
            "token_probe_scores": [round(float(s), 4) for s in self.token_probe_scores],
            "token_entropy_scores": [round(float(s), 4) for s in self.token_entropy_scores],
            "token_attention_scores": [round(float(s), 4) for s in self.token_attention_scores],
            "token_ensemble_scores": [round(float(s), 4) for s in self.token_ensemble_scores],
            "layer_attribution": self.layer_attribution,
            "diagnostics": self.diagnostics,
        }

    def __str__(self) -> str:
        lines = [
            f"Hallucinated: {self.is_hallucinated}  (max score: {self.max_score:.3f}, mode: {self.scoring_mode})",
        ]
        if self.hallucinated_spans:
            lines.append("Detected Spans (Multi-Signal Breakdown):")
            for span in self.hallucinated_spans:
                lines.append(
                    f"  [{span.score:.3f}] \"{span.text}\" "
                    f"[Probe: {span.probe_score:.2f} | Entropy: {span.entropy_score:.2f} | Attn: {span.attention_score:.2f}] "
                    f"→ {span.category.upper()} (Dominant: {span.dominant_signal})"
                )
        else:
            lines.append("No hallucinations detected.")

        if self.sentence_scores:
            lines.append("Sentence Scores:")
            for s in self.sentence_scores:
                flag = " ⚠ [HALLUCINATED]" if s.is_hallucinated else " ✓ [FACTUAL]"
                lines.append(f"  [{s.score:.3f}]{flag} {s.text[:80].strip()!r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class HallucinationDetector:
    """Friendly inference wrapper around a trained ProbedModel with multi-signal fusion.

    Signals combined:
    1. Probe Head Probabilities (internal hidden activation classification).
    2. Predictive Token Entropy (generation confidence & logit margin).
    3. Attention Dispersion (context decoupling from prompt grounding).

    Args:
        probed_model:       Trained ProbedModel.
        tokenizer:          Matching tokenizer.
        threshold:          Hallucination decision threshold.
        min_span_tokens:    Minimum consecutive tokens required to form a span.
        device:             Torch device to run inference on.
        scoring_mode:       Scoring strategy ('ensemble', 'probe_only', 'entropy_only', 'attention_only').
        signal_weights:     Weights for ensembling {'probe': 0.55, 'entropy': 0.25, 'attention': 0.20}.
        output_attentions:  Whether to extract attention matrices for dispersion analysis.
    """

    def __init__(
        self,
        probed_model: ProbedModel,
        tokenizer: AutoTokenizer,
        threshold: float = 0.5,
        min_span_tokens: int = 1,
        device: Optional[torch.device] = None,
        scoring_mode: str = "ensemble",
        signal_weights: Optional[Dict[str, float]] = None,
        output_attentions: bool = True,
    ):
        self.model = probed_model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.min_span_tokens = min_span_tokens
        self.scoring_mode = scoring_mode
        self.signal_weights = signal_weights or dict(MultiSignalScorer.DEFAULT_WEIGHTS)
        self.output_attentions = output_attentions
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
        scoring_mode: str = "ensemble",
        signal_weights: Optional[Dict[str, float]] = None,
        output_attentions: bool = True,
        device: Optional[str] = None,
    ) -> "HallucinationDetector":
        """Load a detector from a saved probe (disk or HuggingFace Hub).

        Args:
            probe_id:           Identifier of the probe (directory under value_head_probes/).
            model_name:         Override base model name.
            load_from:          ``"disk"`` or ``"hf"``.
            hf_repo_id:         HuggingFace repo ID if downloading.
            threshold:          Decision threshold.
            scoring_mode:       ``"ensemble"``, ``"probe_only"``, ``"entropy_only"``, or ``"attention_only"``.
            signal_weights:     Custom weight dictionary for signal fusion.
            output_attentions:  Whether to compute attention dispersion.
            device:             Device override ('cuda', 'cpu', 'auto').
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

        actual_device = get_device() if (device is None or device == "auto") else torch.device(device)
        _, tokenizer = load_model_and_tokenizer(config.model_name)
        probed_model = ProbedModel.load(config=config, path=config.probe_path, map_location=str(actual_device))

        return cls(
            probed_model=probed_model,
            tokenizer=tokenizer,
            threshold=threshold,
            device=actual_device,
            scoring_mode=scoring_mode,
            signal_weights=signal_weights,
            output_attentions=output_attentions,
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
        """Detect hallucinations in a (prompt, completion) pair using multi-signal scoring.

        Args:
            prompt:             The user query or context.
            completion:         The assistant response to analyze.
            return_token_scores: If True, returns full token score arrays.

        Returns:
            A :class:`DetectionResult` with spans, multi-signal scores, and layer diagnostics.
        """
        # Tokenize conversation
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

        seq_len = input_ids.shape[1]
        token_texts = [self.tokenizer.decode(input_ids[0, i]) for i in range(seq_len)]

        # Find where completion starts
        input_str = self.tokenizer.decode(input_ids[0])
        assistant_slice = find_assistant_tokens_slice(input_ids[0], input_str, self.tokenizer)
        completion_start = assistant_slice.stop

        # Run model with multi-signal computation
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=self.output_attentions,
            compute_multi_signal=True,
            prompt_length=completion_start,
            signal_weights=self.signal_weights,
            return_layer_breakdown=True,
        )

        # Extract per-signal probabilities
        probe_probs = outputs["probe_probs"][0].cpu().float().numpy()
        entropy_scores = (
            outputs["entropy_scores"][0].cpu().float().numpy()
            if "entropy_scores" in outputs and outputs["entropy_scores"] is not None
            else np.zeros_like(probe_probs)
        )
        attention_scores = (
            outputs["attention_dispersion_scores"][0].cpu().float().numpy()
            if "attention_dispersion_scores" in outputs and outputs["attention_dispersion_scores"] is not None
            else np.zeros_like(probe_probs)
        )
        ensemble_scores = (
            outputs["ensemble_scores"][0].cpu().float().numpy()
            if "ensemble_scores" in outputs and outputs["ensemble_scores"] is not None
            else probe_probs
        )

        # Slice for completion tokens
        c_tokens = token_texts[completion_start:]
        c_probe = probe_probs[completion_start:]
        c_entropy = entropy_scores[completion_start:]
        c_attn = attention_scores[completion_start:]
        c_ensemble = ensemble_scores[completion_start:]

        # Choose active scores according to scoring_mode
        if self.scoring_mode == "probe_only":
            active_scores = c_probe
        elif self.scoring_mode == "entropy_only":
            active_scores = c_entropy
        elif self.scoring_mode == "attention_only":
            active_scores = c_attn
        else:
            active_scores = c_ensemble

        # Extract multi-signal spans
        hallucinated_spans = self._extract_spans_multisignal(
            active_scores=active_scores,
            probe_scores=c_probe,
            entropy_scores=c_entropy,
            attn_scores=c_attn,
            tokens=c_tokens,
            offset=completion_start,
        )

        max_score = float(active_scores.max()) if len(active_scores) > 0 else 0.0
        is_hallucinated = max_score >= self.threshold

        # Extract layer attributions if available
        layer_attribution_dict: Optional[Dict[str, List[float]]] = None
        if "layer_attributions" in outputs and outputs["layer_attributions"] is not None:
            l_data = outputs["layer_attributions"]
            layer_attribution_dict = {}
            for lyr, tensor_prob in l_data.get("layer_probs", {}).items():
                layer_attribution_dict[lyr] = tensor_prob[0, completion_start:].cpu().float().tolist()

        result = DetectionResult(
            prompt=prompt,
            completion=completion,
            token_scores=active_scores.tolist() if return_token_scores else [],
            token_texts=c_tokens,
            hallucinated_spans=hallucinated_spans,
            is_hallucinated=is_hallucinated,
            max_score=max_score,
            token_probe_scores=c_probe.tolist() if return_token_scores else [],
            token_entropy_scores=c_entropy.tolist() if return_token_scores else [],
            token_attention_scores=c_attn.tolist() if return_token_scores else [],
            token_ensemble_scores=c_ensemble.tolist() if return_token_scores else [],
            layer_attribution=layer_attribution_dict,
            scoring_mode=self.scoring_mode,
            signal_weights=dict(self.signal_weights),
            diagnostics={
                "mean_probe_score": float(np.mean(c_probe)) if len(c_probe) > 0 else 0.0,
                "mean_entropy_score": float(np.mean(c_entropy)) if len(c_entropy) > 0 else 0.0,
                "mean_attention_score": float(np.mean(c_attn)) if len(c_attn) > 0 else 0.0,
                "num_flagged_spans": len(hallucinated_spans),
            },
        )

        # Also populate sentence scores
        self.sentence_level_scores(result)
        return result

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def detect_batch(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 8,
        **kwargs,
    ) -> List[DetectionResult]:
        """Run batched multi-signal inference over a list of (prompt, completion) pairs."""
        return_token_scores = kwargs.get("return_token_scores", True)
        results: List[DetectionResult] = []

        for batch_start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[batch_start : batch_start + batch_size]
            conversations = []
            for prompt, completion in batch_pairs:
                conversation = [
                    {"role": "user",      "content": prompt},
                    {"role": "assistant", "content": completion},
                ]
                full_text = self.tokenizer.apply_chat_template(conversation, tokenize=False)
                if self.tokenizer.bos_token and self.tokenizer.bos_token in full_text:
                    full_text = full_text.replace(self.tokenizer.bos_token, "")
                conversations.append((prompt, completion, full_text))

            orig_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"
            batch_enc = self.tokenizer(
                [c[2] for c in conversations],
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True,
            )
            self.tokenizer.padding_side = orig_padding_side

            input_ids = batch_enc["input_ids"].to(self.device)
            attention_mask = batch_enc["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=self.output_attentions,
                    compute_multi_signal=True,
                    signal_weights=self.signal_weights,
                )

            all_probe = outputs["probe_probs"].cpu().float()
            all_entropy = (
                outputs["entropy_scores"].cpu().float()
                if "entropy_scores" in outputs and outputs["entropy_scores"] is not None
                else torch.zeros_like(all_probe)
            )
            all_attn = (
                outputs["attention_dispersion_scores"].cpu().float()
                if "attention_dispersion_scores" in outputs and outputs["attention_dispersion_scores"] is not None
                else torch.zeros_like(all_probe)
            )
            all_ensemble = (
                outputs["ensemble_scores"].cpu().float()
                if "ensemble_scores" in outputs and outputs["ensemble_scores"] is not None
                else all_probe
            )

            for i, (prompt, completion, _full_text) in enumerate(conversations):
                attn_mask_i = attention_mask[i]
                seq_len = int(attn_mask_i.sum())

                probe_i = all_probe[i][-seq_len:].numpy()
                entropy_i = all_entropy[i][-seq_len:].numpy()
                attn_i = all_attn[i][-seq_len:].numpy()
                ensemble_i = all_ensemble[i][-seq_len:].numpy()
                input_ids_i = input_ids[i][-seq_len:]

                input_str = self.tokenizer.decode(input_ids_i)
                assistant_slice = find_assistant_tokens_slice(input_ids_i, input_str, self.tokenizer)
                completion_start = assistant_slice.stop

                c_tokens = [
                    self.tokenizer.decode(input_ids_i[j])
                    for j in range(completion_start, len(input_ids_i))
                ]
                c_probe = probe_i[completion_start:]
                c_entropy = entropy_i[completion_start:]
                c_attn = attn_i[completion_start:]
                c_ensemble = ensemble_i[completion_start:]

                if self.scoring_mode == "probe_only":
                    active_scores = c_probe
                elif self.scoring_mode == "entropy_only":
                    active_scores = c_entropy
                elif self.scoring_mode == "attention_only":
                    active_scores = c_attn
                else:
                    active_scores = c_ensemble

                hallucinated_spans = self._extract_spans_multisignal(
                    active_scores=active_scores,
                    probe_scores=c_probe,
                    entropy_scores=c_entropy,
                    attn_scores=c_attn,
                    tokens=c_tokens,
                    offset=completion_start,
                )

                max_score = float(active_scores.max()) if len(active_scores) > 0 else 0.0
                is_hallucinated = max_score >= self.threshold

                res = DetectionResult(
                    prompt=prompt,
                    completion=completion,
                    token_scores=active_scores.tolist() if return_token_scores else [],
                    token_texts=c_tokens,
                    hallucinated_spans=hallucinated_spans,
                    is_hallucinated=is_hallucinated,
                    max_score=max_score,
                    token_probe_scores=c_probe.tolist() if return_token_scores else [],
                    token_entropy_scores=c_entropy.tolist() if return_token_scores else [],
                    token_attention_scores=c_attn.tolist() if return_token_scores else [],
                    token_ensemble_scores=c_ensemble.tolist() if return_token_scores else [],
                    scoring_mode=self.scoring_mode,
                    signal_weights=dict(self.signal_weights),
                )
                self.sentence_level_scores(res)
                results.append(res)

        return results

    # ------------------------------------------------------------------
    # Multi-signal span extraction & taxonomy categorization
    # ------------------------------------------------------------------

    def _extract_spans_multisignal(
        self,
        active_scores: np.ndarray,
        probe_scores: np.ndarray,
        entropy_scores: np.ndarray,
        attn_scores: np.ndarray,
        tokens: List[str],
        offset: int,
    ) -> List[SpanResult]:
        """Merge consecutive above-threshold tokens into spans with multi-signal taxonomy."""
        flagged = active_scores >= self.threshold
        spans: List[SpanResult] = []
        i = 0

        while i < len(flagged):
            if flagged[i]:
                j = i
                while j < len(flagged) and flagged[j]:
                    j += 1
                if (j - i) >= self.min_span_tokens:
                    span_text = "".join(tokens[i:j])
                    span_score = float(np.mean(active_scores[i:j]))
                    s_probe = float(np.mean(probe_scores[i:j])) if len(probe_scores) > 0 else 0.0
                    s_entropy = float(np.mean(entropy_scores[i:j])) if len(entropy_scores) > 0 else 0.0
                    s_attn = float(np.mean(attn_scores[i:j])) if len(attn_scores) > 0 else 0.0

                    # Determine dominant signal
                    signal_contributions = {
                        "probe": s_probe * self.signal_weights.get("probe", 0.55),
                        "entropy": s_entropy * self.signal_weights.get("entropy", 0.25),
                        "attention": s_attn * self.signal_weights.get("attention", 0.20),
                    }
                    dominant_signal = max(signal_contributions, key=signal_contributions.get)

                    # Determine taxonomy category
                    if s_probe >= self.threshold and s_entropy >= 0.40:
                        category = "entity_fabrication"
                    elif s_attn >= 0.55:
                        category = "context_drift"
                    elif s_entropy >= 0.60:
                        category = "uncertainty_spike"
                    else:
                        category = "calibrated_hallucination"

                    spans.append(SpanResult(
                        text=span_text,
                        start_token=offset + i,
                        end_token=offset + j - 1,
                        score=span_score,
                        probe_score=s_probe,
                        entropy_score=s_entropy,
                        attention_score=s_attn,
                        dominant_signal=dominant_signal,
                        category=category,
                    ))
                i = j
            else:
                i += 1
        return spans

    # ------------------------------------------------------------------
    # Sentence-level aggregation
    # ------------------------------------------------------------------

    def sentence_level_scores(
        self,
        result: DetectionResult,
    ) -> List[SentenceResult]:
        """Split the completion into sentences and score each one using multi-signal aggregation."""
        if not result.token_scores:
            return []

        probs = np.array(result.token_scores)
        probe_p = np.array(result.token_probe_scores) if result.token_probe_scores else probs
        entropy_p = np.array(result.token_entropy_scores) if result.token_entropy_scores else np.zeros_like(probs)
        attn_p = np.array(result.token_attention_scores) if result.token_attention_scores else np.zeros_like(probs)
        tokens = result.token_texts

        completion_text = "".join(tokens)

        try:
            import nltk
            try:
                sentences = nltk.sent_tokenize(completion_text)
            except LookupError:
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)
                sentences = nltk.sent_tokenize(completion_text)
        except ImportError:
            sentences = re.split(r'(?<=[.!?])\s+', completion_text.strip())
            sentences = [s for s in sentences if s.strip()]

        if not sentences:
            return []

        sentence_results: List[SentenceResult] = []
        char_cursor = 0
        token_cursor = 0

        for sent_text in sentences:
            sent_start_char = completion_text.find(sent_text, char_cursor)
            if sent_start_char == -1:
                char_cursor += len(sent_text)
                continue
            sent_end_char = sent_start_char + len(sent_text)

            sent_token_probs: List[float] = []
            sent_probe_probs: List[float] = []
            sent_entropy_probs: List[float] = []
            sent_attn_probs: List[float] = []

            for t_idx in range(token_cursor, len(tokens)):
                tok_char_start = len("".join(tokens[token_cursor:t_idx]))
                tok_char_end = tok_char_start + len(tokens[t_idx])

                abs_tok_start = char_cursor + tok_char_start
                abs_tok_end = char_cursor + tok_char_end

                if abs_tok_start >= sent_end_char:
                    break
                if abs_tok_end <= sent_start_char:
                    token_cursor = t_idx + 1
                    continue

                sent_token_probs.append(float(probs[t_idx]) if t_idx < len(probs) else 0.0)
                sent_probe_probs.append(float(probe_p[t_idx]) if t_idx < len(probe_p) else 0.0)
                sent_entropy_probs.append(float(entropy_p[t_idx]) if t_idx < len(entropy_p) else 0.0)
                sent_attn_probs.append(float(attn_p[t_idx]) if t_idx < len(attn_p) else 0.0)

            char_cursor = sent_end_char

            if not sent_token_probs:
                sent_token_probs = [0.0]
                sent_probe_probs = [0.0]
                sent_entropy_probs = [0.0]
                sent_attn_probs = [0.0]

            max_score = float(np.max(sent_token_probs))
            mean_score = float(np.mean(sent_token_probs))
            mean_probe = float(np.mean(sent_probe_probs))
            mean_entropy = float(np.mean(sent_entropy_probs))
            mean_attn = float(np.mean(sent_attn_probs))

            contribs = {
                "probe": mean_probe * self.signal_weights.get("probe", 0.55),
                "entropy": mean_entropy * self.signal_weights.get("entropy", 0.25),
                "attention": mean_attn * self.signal_weights.get("attention", 0.20),
            }
            dominant_sig = max(contribs, key=contribs.get)

            sentence_results.append(SentenceResult(
                text=sent_text,
                score=max_score,
                mean_score=mean_score,
                is_hallucinated=max_score >= self.threshold,
                probe_score=mean_probe,
                entropy_score=mean_entropy,
                attention_score=mean_attn,
                dominant_signal=dominant_sig,
            ))

        result.sentence_scores = sentence_results
        return sentence_results

    # ------------------------------------------------------------------
    # Explainability & Diagnostics
    # ------------------------------------------------------------------

    def explain_prediction(self, result: DetectionResult) -> str:
        """Generate a structured diagnostic explanation of the multi-signal detection."""
        lines = [
            "=" * 70,
            "🔍 MULTI-SIGNAL HALLUCINATION DIAGNOSTICS",
            "=" * 70,
            f"Overall Status   : {'🚨 HALLUCINATION DETECTED' if result.is_hallucinated else '✅ FACTUALLY GROUNDED'}",
            f"Max Confidence   : {result.max_score:.4f} (Threshold = {self.threshold:.2f})",
            f"Active Strategy  : {result.scoring_mode.upper()} "
            f"[Weights: Probe={result.signal_weights.get('probe', 0):.2f}, "
            f"Entropy={result.signal_weights.get('entropy', 0):.2f}, "
            f"Attn={result.signal_weights.get('attention', 0):.2f}]",
            "-" * 70,
        ]

        if not result.hallucinated_spans:
            lines.append("No suspicious spans detected in the generated completion.")
        else:
            lines.append(f"Flagged Spans ({len(result.hallucinated_spans)} detected):")
            for idx, s in enumerate(result.hallucinated_spans, 1):
                lines.append(f"\n[{idx}] \"{s.text.strip()}\"")
                lines.append(f"    • Category        : {s.category.upper()}")
                lines.append(f"    • Dominant Signal : {s.dominant_signal.upper()}")
                lines.append(f"    • Fused Score     : {s.score:.4f}")
                lines.append(f"    • Signal Breakdown: Probe={s.probe_score:.3f} | Entropy={s.entropy_score:.3f} | Attention-Drift={s.attention_score:.3f}")

                # Explanation rule
                if s.category == "entity_fabrication":
                    explanation = "High internal activation divergence combined with elevated predictive entropy indicates model is fabricating entities/dates."
                elif s.category == "context_drift":
                    explanation = "Attention decoupled from prompt grounding context, drifting into ungrounded free-form generation."
                elif s.category == "uncertainty_spike":
                    explanation = "Spike in logit distribution entropy; model lacked high confidence during next-token selection."
                else:
                    explanation = "Calibrated multi-signal agreement triggered decision threshold."
                lines.append(f"    • Root Cause      : {explanation}")

        if result.layer_attribution:
            lines.append("\n" + "-" * 70)
            lines.append("Layer Attribution Overview (Top Detection Layers):")
            for lyr, probs in list(result.layer_attribution.items())[:5]:
                max_lyr = max(probs) if probs else 0.0
                lines.append(f"    • Layer {lyr:>2} : peak score = {max_lyr:.3f}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def highlight(self, result: DetectionResult) -> str:
        """Return the completion with hallucinated spans wrapped in [[ ]]."""
        text = result.completion
        for span in sorted(result.hallucinated_spans, key=lambda s: -s.start_token):
            text = text.replace(span.text.strip(), f"[[{span.text.strip()}]]", 1)
        return text

    def detect_file(
        self,
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        batch_size: int = 16,
        include_sentences: bool = True,
    ) -> int:
        """Run batch hallucination detection over a JSONL file."""
        from tqdm import tqdm

        input_path = Path(input_file)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records: List[dict] = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "prompt" not in data or "completion" not in data:
                        raise ValueError(f"Line {line_no} missing 'prompt' or 'completion'")
                    records.append(data)
                except Exception as e:
                    print(f"Warning: skipping line {line_no}: {e}")

        total_samples = len(records)
        print(f"Loaded {total_samples} samples from {input_path}")
        if total_samples == 0:
            return 0

        start_time = time.time()
        processed_count = 0

        with open(output_path, "w", encoding="utf-8") as out_f:
            for i in tqdm(range(0, total_samples, batch_size), desc="Scoring JSONL batches"):
                chunk = records[i : i + batch_size]
                pairs = [(r["prompt"], r["completion"]) for r in chunk]

                results = self.detect_batch(pairs)

                for orig_record, res in zip(chunk, results):
                    out_record = dict(orig_record)
                    out_record["is_hallucinated"] = res.is_hallucinated
                    out_record["hallucination_score"] = float(res.hallucination_score)
                    out_record["scoring_mode"] = res.scoring_mode
                    out_record["hallucinated_spans"] = [s.to_dict() for s in res.hallucinated_spans]
                    out_record["highlighted_completion"] = self.highlight(res)

                    if include_sentences:
                        sent_results = self.sentence_level_scores(res)
                        out_record["sentences"] = [s.to_dict() for s in sent_results]

                    out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                    processed_count += 1

        elapsed = time.time() - start_time
        throughput = processed_count / max(elapsed, 1e-4)
        print(f"Finished {processed_count} samples in {elapsed:.2f}s ({throughput:.1f} samples/s)")
        print(f"Saved results to: {output_path}")
        return processed_count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Signal Hallucination Detector")
    parser.add_argument("--probe_id",          type=str, required=True, help="Probe directory name or HF identifier")
    parser.add_argument("--prompt",            type=str, default=None, help="Input prompt text")
    parser.add_argument("--completion",        type=str, default=None, help="Input completion text")
    parser.add_argument("--input_file",        type=str, default=None, help="Path to input .jsonl file for batch scoring")
    parser.add_argument("--output_file",       type=str, default=None, help="Path to output .jsonl file")
    parser.add_argument("--batch_size",        type=int, default=16, help="Batch size for batch processing")
    parser.add_argument("--threshold",         type=float, default=0.5, help="Hallucination decision threshold")
    parser.add_argument("--scoring_mode",      type=str, default="ensemble",
                        choices=["ensemble", "probe_only", "entropy_only", "attention_only"],
                        help="Signal scoring mode")
    parser.add_argument("--weight_probe",      type=float, default=0.55, help="Weight for hidden-state probe")
    parser.add_argument("--weight_entropy",    type=float, default=0.25, help="Weight for predictive logit entropy")
    parser.add_argument("--weight_attention",  type=float, default=0.20, help="Weight for attention dispersion")
    parser.add_argument("--device",            type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--load_from",         type=str, default="disk", choices=["disk", "hf"])
    parser.add_argument("--hf_repo_id",        type=str, default="andyrdt/hallucination-probes")
    parser.add_argument("--json",              action="store_true", help="Output JSON format")
    parser.add_argument("--include_sentences", action="store_true", help="Include sentence-level scores")
    parser.add_argument("--explain",           action="store_true", help="Print multi-signal diagnostic explanation")
    return parser.parse_args()


def main():
    args = parse_args()

    weights = {
        "probe": args.weight_probe,
        "entropy": args.weight_entropy,
        "attention": args.weight_attention,
    }

    detector = HallucinationDetector.from_pretrained(
        probe_id=args.probe_id,
        load_from=args.load_from,
        hf_repo_id=args.hf_repo_id,
        device=args.device,
        threshold=args.threshold,
        scoring_mode=args.scoring_mode,
        signal_weights=weights,
    )

    if args.input_file:
        if not args.output_file:
            in_p = Path(args.input_file)
            args.output_file = str(in_p.with_name(f"{in_p.stem}_detected{in_p.suffix}"))
        detector.detect_file(
            input_file=args.input_file,
            output_file=args.output_file,
            batch_size=args.batch_size,
            include_sentences=args.include_sentences,
        )
    elif args.prompt is not None and args.completion is not None:
        result = detector.detect(prompt=args.prompt, completion=args.completion)

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print("\n" + "=" * 60)
            print(result)
            if args.include_sentences:
                print("\nSentence-level Breakdown:")
                sent_results = detector.sentence_level_scores(result)
                for i, s in enumerate(sent_results, 1):
                    tag = "🚨 [HALLUCINATION]" if s.is_hallucinated else "✅ [FACTUAL]"
                    print(f"  {i}. {tag} (max={s.score:.3f}, mean={s.mean_score:.3f}, dominant={s.dominant_signal}): {s.text}")
            print("\nHighlighted completion:")
            print(detector.highlight(result))

            if args.explain:
                print("\n" + detector.explain_prediction(result))
            print("=" * 60)
    else:
        print("Error: Must provide either (--prompt AND --completion) or --input_file")
        exit(1)


if __name__ == "__main__":
    main()
