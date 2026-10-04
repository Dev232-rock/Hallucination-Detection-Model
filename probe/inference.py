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

import numpy as np
import torch
import torch.nn.functional as F
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
class SentenceResult:
    """Aggregated hallucination score for one sentence in the completion."""
    text: str             # sentence text
    score: float          # max token hallucination probability within the sentence
    mean_score: float     # mean token hallucination probability within the sentence
    is_hallucinated: bool # True if score >= detector threshold


@dataclass
class DetectionResult:
    """Full detection output for one (prompt, completion) pair."""
    prompt: str
    completion: str
    token_scores: List[float]       # per-token hallucination probability
    token_texts: List[str]          # decoded text for each token
    hallucinated_spans: List[SpanResult] = field(default_factory=list)
    sentence_scores: List[SentenceResult] = field(default_factory=list)
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
            "sentence_scores": [
                {
                    "text": s.text,
                    "score": s.score,
                    "mean_score": s.mean_score,
                    "is_hallucinated": s.is_hallucinated,
                }
                for s in self.sentence_scores
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
        if self.sentence_scores:
            lines.append("Sentence scores:")
            for s in self.sentence_scores:
                flag = " ⚠" if s.is_hallucinated else ""
                lines.append(f"  [{s.score:.3f}]{flag} {s.text[:80].strip()!r}")
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
        batch_size: int = 8,
        **kwargs,
    ) -> List[DetectionResult]:
        """Run batched inference over a list of ``(prompt, completion)`` pairs.

        Instead of calling :meth:`detect` sequentially (one forward pass per
        sample), this method pads all pairs in each mini-batch to the same
        length and performs **a single forward pass per batch**, giving
        substantially better GPU utilisation.

        Args:
            pairs:      List of ``(prompt, completion)`` tuples.
            batch_size: Number of pairs to process per forward pass.
            **kwargs:   Extra keyword arguments forwarded to :meth:`detect`
                        (e.g. ``return_token_scores``).

        Returns:
            List of :class:`DetectionResult` in the same order as *pairs*.
        """
        return_token_scores = kwargs.get("return_token_scores", True)
        results: List[DetectionResult] = []

        for batch_start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[batch_start : batch_start + batch_size]

            # ── 1. Tokenise each pair independently so we know the lengths ──
            encodings = []
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

            # Batch-tokenise with left-padding so the *last* token positions
            # are aligned (right-padding would misalign completions).
            orig_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"
            batch_enc = self.tokenizer(
                [c[2] for c in conversations],
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True,
            )
            self.tokenizer.padding_side = orig_padding_side  # restore

            input_ids      = batch_enc["input_ids"].to(self.device)       # (B, L)
            attention_mask = batch_enc["attention_mask"].to(self.device)  # (B, L)

            # ── 2. Single forward pass ──
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            all_probs = outputs["probe_probs"].cpu().float()  # (B, L)

            # ── 3. Slice per-sample results ──
            for i, (prompt, completion, full_text) in enumerate(conversations):
                probs_i      = all_probs[i]          # (L,)
                input_ids_i  = input_ids[i]          # (L,)
                attn_mask_i  = attention_mask[i]     # (L,)
                seq_len      = int(attn_mask_i.sum()) # unpadded length

                # Trim left-padding
                probs_i     = probs_i[-seq_len:]     # (seq_len,)
                input_ids_i = input_ids_i[-seq_len:] # (seq_len,)

                # Find where the assistant response starts
                input_str = self.tokenizer.decode(input_ids_i)
                from utils.tokenization import find_assistant_tokens_slice
                assistant_slice   = find_assistant_tokens_slice(input_ids_i, input_str, self.tokenizer)
                completion_start  = assistant_slice.stop

                completion_probs  = probs_i[completion_start:].numpy()
                completion_tokens = [
                    self.tokenizer.decode(input_ids_i[j])
                    for j in range(completion_start, len(input_ids_i))
                ]

                hallucinated_spans = self._extract_spans(
                    probs=completion_probs,
                    tokens=completion_tokens,
                    offset=completion_start,
                )

                max_score      = float(completion_probs.max()) if len(completion_probs) > 0 else 0.0
                is_hallucinated = max_score >= self.threshold

                results.append(DetectionResult(
                    prompt=prompt,
                    completion=completion,
                    token_scores=completion_probs.tolist() if return_token_scores else [],
                    token_texts=completion_tokens,
                    hallucinated_spans=hallucinated_spans,
                    is_hallucinated=is_hallucinated,
                    max_score=max_score,
                ))

        return results

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
    # Sentence-level aggregation
    # ------------------------------------------------------------------

    def sentence_level_scores(
        self,
        result: DetectionResult,
    ) -> List[SentenceResult]:
        """Split the completion into sentences and score each one.

        The score for a sentence is the **max** per-token hallucination
        probability among all tokens that fall inside that sentence.  The mean
        is also stored for softer ranking.

        Sentence splitting uses ``nltk.sent_tokenize`` when available,
        falling back to a simple regex split on punctuation boundaries.

        Args:
            result: A :class:`DetectionResult` returned by :meth:`detect`.

        Returns:
            List of :class:`SentenceResult`, one per sentence.  Also stores
            the list in ``result.sentence_scores`` for convenience.
        """
        if not result.token_scores:
            return []

        probs  = np.array(result.token_scores)   # (n_completion_tokens,)
        tokens = result.token_texts               # same length

        # Reconstruct the completion text from decoded tokens
        completion_text = "".join(tokens)

        # Split into sentences
        try:
            import nltk
            try:
                sentences = nltk.sent_tokenize(completion_text)
            except LookupError:
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)
                sentences = nltk.sent_tokenize(completion_text)
        except ImportError:
            # Fallback: split on sentence-ending punctuation
            import re
            sentences = re.split(r'(?<=[.!?])\s+', completion_text.strip())
            sentences = [s for s in sentences if s.strip()]

        if not sentences:
            return []

        sentence_results: List[SentenceResult] = []
        char_cursor = 0
        token_cursor = 0

        for sent_text in sentences:
            # Find the start of this sentence in the completion text
            sent_start_char = completion_text.find(sent_text, char_cursor)
            if sent_start_char == -1:
                # If not found (unlikely), skip
                char_cursor += len(sent_text)
                continue
            sent_end_char = sent_start_char + len(sent_text)

            # Map character range → token range by re-assembling tokens
            sent_token_probs: List[float] = []
            rebuilt = ""
            for t_idx in range(token_cursor, len(tokens)):
                rebuilt_next = rebuilt + tokens[t_idx]
                tok_start_char = sum(len(tokens[k]) for k in range(token_cursor, t_idx))

                # Check if this token overlaps with the sentence character range
                tok_char_start = len("".join(tokens[token_cursor:t_idx]))
                tok_char_end   = tok_char_start + len(tokens[t_idx])

                # Map relative to char_cursor
                abs_tok_start = char_cursor + tok_char_start
                abs_tok_end   = char_cursor + tok_char_end

                if abs_tok_start >= sent_end_char:
                    break  # past the sentence
                if abs_tok_end <= sent_start_char:
                    token_cursor = t_idx + 1
                    continue  # before the sentence

                sent_token_probs.append(float(probs[t_idx]) if t_idx < len(probs) else 0.0)

            char_cursor = sent_end_char

            if not sent_token_probs:
                sent_token_probs = [0.0]

            max_score  = float(np.max(sent_token_probs))
            mean_score = float(np.mean(sent_token_probs))

            sentence_results.append(SentenceResult(
                text=sent_text,
                score=max_score,
                mean_score=mean_score,
                is_hallucinated=max_score >= self.threshold,
            ))

        result.sentence_scores = sentence_results
        return sentence_results

    def calibrate(
        self,
        validation_pairs: List[Tuple[str, str, float]],
        metric: str = "f1",
        batch_size: int = 8,
        save_path: Optional[Union[str, Path]] = None,
    ) -> float:
        """Find the optimal decision threshold from a labelled validation set.

        The method sweeps 100 candidate thresholds between 0 and 1 and picks
        the one that maximises *metric* at the **span level** (max-aggregation).
        The best threshold is stored in ``self.threshold`` and, optionally,
        persisted to disk.

        Args:
            validation_pairs: List of ``(prompt, completion, label)`` triples
                where *label* is ``1.0`` (hallucinated) or ``0.0`` (factual)
                at the **response level** (sentence/response granularity).
            metric:    One of ``"f1"``, ``"accuracy"``, ``"auc"``.
            batch_size: Batch size for inference.
            save_path: If provided, writes ``{"threshold": <value>}`` to this
                JSON file so it can be reloaded later.

        Returns:
            The best threshold found.
        """
        if not validation_pairs:
            raise ValueError("validation_pairs must not be empty")

        pairs      = [(p, c) for p, c, _ in validation_pairs]
        true_labels = np.array([float(l) for _, _, l in validation_pairs])

        print(f"Calibrating threshold on {len(pairs)} validation pairs ...")
        results = self.detect_batch(pairs, batch_size=batch_size, return_token_scores=True)

        # Per-response score: max token probability in the completion
        pred_scores = np.array([
            max(r.token_scores) if r.token_scores else 0.0
            for r in results
        ])

        if len(np.unique(true_labels)) < 2:
            print("Warning: only one class present in validation set — cannot calibrate.")
            return self.threshold

        best_threshold = 0.5
        best_value     = -1.0
        candidates     = np.linspace(0.0, 1.0, 101)

        from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

        for t in candidates:
            preds = (pred_scores >= t).astype(float)
            if metric == "f1":
                value = f1_score(true_labels, preds, zero_division=0)
            elif metric == "accuracy":
                value = accuracy_score(true_labels, preds)
            elif metric == "auc":
                # AUC is threshold-independent; just compute it once
                value = roc_auc_score(true_labels, pred_scores)
            else:
                raise ValueError(f"Unknown metric '{metric}'. Choose from: f1, accuracy, auc.")

            if value > best_value:
                best_value     = value
                best_threshold = float(t)

            if metric == "auc":
                break  # AUC doesn't depend on threshold

        self.threshold = best_threshold
        print(f"  Best threshold = {best_threshold:.4f}  ({metric} = {best_value:.4f})")

        if save_path is not None:
            import json
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump({"threshold": best_threshold, metric: best_value}, f, indent=2)
            print(f"  Threshold saved → {save_path}")

        return best_threshold

    @classmethod
    def load_threshold(cls, path: Union[str, Path]) -> float:
        """Load a previously calibrated threshold from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return float(data["threshold"])

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
        """Run batch hallucination detection over a JSONL file.

        Each input JSONL line must contain at least "prompt" and "completion".
        All other fields from the input are preserved, and detection fields are added:
            - is_hallucinated (bool)
            - hallucination_score (float)
            - hallucinated_spans (list of dicts)
            - highlighted_completion (str)
            - sentences (list of dicts, if include_sentences is True)

        Args:
            input_file: Path to input .jsonl file.
            output_file: Path to output .jsonl file.
            batch_size: Number of pairs per forward pass.
            include_sentences: Whether to compute sentence-level scores.

        Returns:
            Total number of samples processed.
        """
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
                    out_record["hallucinated_spans"] = [
                        {
                            "text": s.text,
                            "start_token": s.start_token,
                            "end_token": s.end_token,
                            "score": float(s.score),
                        }
                        for s in res.hallucinated_spans
                    ]
                    out_record["highlighted_completion"] = self.highlight(res)

                    if include_sentences:
                        sent_results = self.sentence_level_scores(res)
                        out_record["sentences"] = [
                            {
                                "text": s.text,
                                "score": float(s.score),
                                "mean_score": float(s.mean_score),
                                "is_hallucinated": s.is_hallucinated,
                            }
                            for s in sent_results
                        ]

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
    parser = argparse.ArgumentParser(description="Detect hallucinations in text or JSONL datasets")
    parser.add_argument("--probe_id",          type=str, required=True, help="Probe directory name or HF identifier")
    parser.add_argument("--prompt",            type=str, default=None, help="Input prompt text")
    parser.add_argument("--completion",        type=str, default=None, help="Input completion text")
    parser.add_argument("--input_file",        type=str, default=None, help="Path to input .jsonl file for batch scoring")
    parser.add_argument("--output_file",       type=str, default=None, help="Path to output .jsonl file")
    parser.add_argument("--batch_size",        type=int, default=16, help="Batch size for batch processing")
    parser.add_argument("--threshold",         type=float, default=0.5, help="Hallucination decision threshold")
    parser.add_argument("--device",            type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--load_from",         type=str, default="disk", choices=["disk", "hf"])
    parser.add_argument("--hf_repo_id",        type=str, default="andyrdt/hallucination-probes")
    parser.add_argument("--json",              action="store_true", help="Output JSON instead of human-readable (single mode)")
    parser.add_argument("--include_sentences", action="store_true", help="Include sentence-level scores in output")
    return parser.parse_args()


def main():
    args = parse_args()

    detector = HallucinationDetector.from_pretrained(
        probe_id=args.probe_id,
        load_from=args.load_from,
        hf_repo_id=args.hf_repo_id,
        device=args.device,
        threshold=args.threshold,
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
            out_data = result.to_dict()
            if args.include_sentences:
                sent_results = detector.sentence_level_scores(result)
                out_data["sentences"] = [
                    {
                        "text": s.text,
                        "score": float(s.score),
                        "mean_score": float(s.mean_score),
                        "is_hallucinated": s.is_hallucinated,
                    }
                    for s in sent_results
                ]
            print(json.dumps(out_data, indent=2))
        else:
            print("\n" + "=" * 60)
            print(result)
            if args.include_sentences:
                print("\nSentence-level Breakdown:")
                sent_results = detector.sentence_level_scores(result)
                for i, s in enumerate(sent_results, 1):
                    tag = "🚨 [HALLUCINATION]" if s.is_hallucinated else "✅ [FACTUAL]"
                    print(f"  {i}. {tag} (max={s.score:.3f}, mean={s.mean_score:.3f}): {s.text}")
            print("\nHighlighted completion:")
            print(detector.highlight(result))
            print("=" * 60)
    else:
        print("Error: Must provide either (--prompt AND --completion) or --input_file")
        exit(1)


if __name__ == "__main__":
    main()
