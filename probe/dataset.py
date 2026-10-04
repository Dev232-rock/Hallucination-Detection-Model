# Tokenized dataset classes with token-level labels for probe training.
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, Generator, Iterator, List, Optional, Tuple
import torch
import datasets
from jaxtyping import Float, Int
from termcolor import colored
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset

from tqdm import tqdm
from transformers import AutoTokenizer

from utils.tokenization import find_assistant_tokens_slice, find_string_in_tokens, slice_to_list
from .types import AnnotatedSpan, ProbingItem
from .dataset_converters import get_prepare_function
@dataclass
class TokenizedProbingDatasetConfig:
    # Configuration for tokenizing and labeling a probing dataset at token level\
    dataset_id: str             
    hf_repo: str
    subset: Optional[str] = None
    split: str = "train"
    max_length: int = 2048
    ignore_buffer: int = 0  # Buffer around spans to ignore
    default_ignore: bool = False  # If true, ignore tokens not in any span
    last_span_token: bool = False  # If true, only label the last token of each span
    pos_weight: float = 1.0  # Weight for positive (hallucination) tokens
    neg_weight: float = 1.0  # Weight for negative (supported) tokens
    shuffle: bool = True
    seed: int = 42
    process_on_the_fly: bool = False
    max_num_samples: Optional[int] = None
class TokenizedProbingDataset(Dataset):
    #Dataset for probing model activations with annotated spans.    
    def __init__(
        self,
        items: List[ProbingItem],
        config: TokenizedProbingDatasetConfig,
        tokenizer: AutoTokenizer,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.items = deepcopy(items)
        self.processed_items = [None] * len(items)
        self.debug_mode = False
        self.print_first_example = False

        self._num_skipped_spans: int = 0
        self._num_added_spans: int = 0

        if self.config.shuffle:
            self._shuffle_items()

        # Limit samples if specified (do this after shuffling)
        if self.config.max_num_samples:
            self.items = self.items[:self.config.max_num_samples]
            self.processed_items = self.processed_items[:self.config.max_num_samples]

        if not self.config.process_on_the_fly:
            self._process_items()

    def _process_items(self):
        """Pre-process all items in the dataset."""
        for i, item in tqdm(enumerate(self.items), desc=f"Processing items ({self.config.dataset_id})", total=len(self.items)):
            if i == 0 and self.print_first_example:
                self.debug_mode = True
            else:
                self.debug_mode = False
            processed_item = self._process_item(item)
            if processed_item:
                self.processed_items[i] = processed_item

        print(f"Dataset {self.config.dataset_id} stats:")
        print(f"\t- Number of added spans: {self._num_added_spans}")
        print(f"\t- Number of skipped spans: {self._num_skipped_spans} / {self._num_added_spans + self._num_skipped_spans}")
        print(f"\t- Total number of items: {len(self.items)}")

    def _process_item(self, item: ProbingItem) -> Dict:
        #Process a single example into tokenized format with labels.
        conversation = [
            {'role': 'user', 'content': item.prompt},
            {'role': 'assistant', 'content': item.completion}
        ]
        full_text = self.tokenizer.apply_chat_template(conversation, tokenize=False)

        if self.tokenizer.bos_token and self.tokenizer.bos_token in full_text:
            full_text = full_text.replace(self.tokenizer.bos_token, '')
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.config.max_length,
            padding='max_length',
            return_tensors='pt',
            padding_side='right'
        )
        input_ids: Int[Tensor, "seq_len"] = encoding["input_ids"][0]
        attention_mask: Int[Tensor, "seq_len"] = encoding["attention_mask"][0]

        labels, weights, pos_spans, neg_spans = self._compute_positional_labels(
            input_ids=input_ids,
            item=item
        )

        input_str: str = self.tokenizer.decode(input_ids)
        assistant_tokens_slice = find_assistant_tokens_slice(input_ids, input_str, self.tokenizer)
        completion_start_idx = assistant_tokens_slice.stop

        lm_labels = input_ids.clone()
        lm_labels[:completion_start_idx] = -100  # ignore all tokens in the prompt
        lm_labels[attention_mask == 0] = -100  # ignore padding tokens

        return {
            "input_ids": input_ids,  # Int[Tensor, "seq_len"]
            "attention_mask": attention_mask,  # Int[Tensor, "seq_len"]
            "classification_labels": labels,  # Float[Tensor, "seq_len"]
            "classification_weights": weights,  # Float[Tensor, "seq_len"]
            "pos_spans": pos_spans,  # List[List[int]]
            "neg_spans": neg_spans,  # List[List[int]]
            "lm_labels": lm_labels,  # Int[Tensor, "seq_len"]
        }

    def print_token_labels(
        self,
        input_ids: torch.Tensor,
        positive_indices: List[int],
        negative_indices: List[int],
        ignore_indices: List[int],
        spans: List[AnnotatedSpan]
    ):
        """Debug method to print how tokens have been labeled."""

        tokens = [self.tokenizer.decode(tok) for tok in input_ids] 
        print(f"================================================")
        print(f"Number of spans: {len(spans)}")
        print(f"Number of non-factual (hallucinated) spans: {len([f for f in spans if f.label == 1.0])}")
        print(f"Number of N/A spans: {len([f for f in spans if f.label == -100])}")
        print(f"Number of factual spans: {len([f for f in spans if f.label == 0.0])}")
        print(f"Legend: red - positive, green - negative, blue - ignored")     

        for i, token in enumerate(tokens):
            if token == self.tokenizer.eos_token:
                continue
            if i in positive_indices:
                print(colored(token, 'red'), end='')
            elif i in negative_indices:
                print(colored(token, 'green'), end='')
            elif i in ignore_indices:
                print(colored(token, 'blue'), end='')
            else:
                print(token, end='')
        print(f"================================================")

    def _compute_positional_labels(
        self,
        input_ids: torch.Tensor,
        item: ProbingItem
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]], List[List[int]]]:
        """Computes positional labels for a sequence of tokens based on annotated spans."""
        input_str: str = self.tokenizer.decode(input_ids)
        completion: str = item.completion
        
        positive_indices: List[int] = []    # indices of hallucinated spans
        negative_indices: List[int] = []    # indices of supported spans
        ignore_indices: List[int] = []      # indices to ignore in training

        positive_spans: List[List[int]] = []
        negative_spans: List[List[int]] = []

        def get_nearby_indices(span_indices: List[int]) -> List[int]:
            left_window = list(range(max(0, span_indices[0] - self.config.ignore_buffer), span_indices[0]))
            right_window = list(range(span_indices[-1] + 1, min(len(input_ids), span_indices[-1] + 1 + self.config.ignore_buffer)))
            return left_window + right_window

        # Find assistant tokens slice to know where to start looking for spans
        assistant_tokens_slice = find_assistant_tokens_slice(
            input_ids,
            input_str,
            self.tokenizer
        )
        completion_start_idx = assistant_tokens_slice.stop
        cur_idx = assistant_tokens_slice.stop

        # Sort spans by their index in the text
        spans = sorted(item.spans, key=lambda x: x.index)

        for span in spans:
            if span.span not in input_str:
                self._num_skipped_spans += 1
                continue
            try:
                # First try to find the span after the assistant tokens
                positions_slice = find_string_in_tokens(span.span, input_ids[cur_idx:], self.tokenizer)
                positions_slice = slice(positions_slice.start + cur_idx, positions_slice.stop + cur_idx)
            except (AssertionError, ValueError):
                try:
                    # If not found, try the whole input_ids
                    print(f"Repeating position_slice search on all tokens after failing to find span {repr(span.span)} in input_ids[cur_idx:]: {repr(self.tokenizer.decode(input_ids[cur_idx:]))[:50]}...")
                    positions_slice = find_string_in_tokens(span.span, input_ids, self.tokenizer)
                except (AssertionError, ValueError) as e:
                    print(f"Span {repr(span.span)} not found in input_ids, skipping entity")
                    self._num_skipped_spans += 1
                    continue

            if positions_slice is None:
                continue
            span_indices = slice_to_list(positions_slice, len(input_ids))
            if not span_indices:
                continue
            
            cur_idx = positions_slice.start
            ignore_indices_span = get_nearby_indices(span_indices)
            
            if span.label == 1.0:  # hallucinated
                # Check if span is in assistant response
                if all(idx >= completion_start_idx for idx in span_indices):
                    if self.config.last_span_token:
                        # Label only the final token of this span
                        labeled = [span_indices[-1]]
                    else:
                        labeled = span_indices
                    positive_indices.extend(labeled)
                    positive_spans.append(labeled)
                    ignore_indices.extend(ignore_indices_span)
                    self._num_added_spans += 1
                else:
                    print(f"Skipping hallucinated span not in assistant response: {repr(span.span)}")
                    self._num_skipped_spans += 1
                continue
            
            if span.label == 0.0:  # factual
                # Check if span is in prompt
                if all(idx < completion_start_idx for idx in span_indices):
                    if self.config.last_span_token:
                        labeled = [span_indices[-1]]
                    else:
                        labeled = span_indices
                    negative_indices.extend(labeled)
                    negative_spans.append(labeled)
                    continue
            
            ignore_indices.extend(span_indices)
            ignore_indices.extend(ignore_indices_span)
            self._num_added_spans += 1
        
        # Debug printing for first example
        if self.debug_mode:
            self.print_token_labels(
                input_ids=input_ids,
                positive_indices=positive_indices,
                negative_indices=negative_indices,
                ignore_indices=ignore_indices,
                spans=spans
            )
        
        # Ensure labels and weights have same length as input_ids
        labels = torch.zeros(len(input_ids))
        weights = torch.zeros(len(input_ids))
        
        for idx in positive_indices:
            labels[idx] = 1.0
            weights[idx] = self.config.pos_weight

        for idx in negative_indices:
            labels[idx] = 0.0
            weights[idx] = self.config.neg_weight
            
        for idx in ignore_indices:
            labels[idx] = -100.0
            weights[idx] = 0.0
        
        return labels, weights, positive_spans, negative_spans
    def _shuffle_items(self):
        """Shuffle items and their pre-processed counterparts in lock-step."""
        combined = list(zip(self.items, self.processed_items))
        random.seed(self.config.seed)
        random.shuffle(combined)
        self.items, self.processed_items = zip(*combined) if combined else ([], [])
        self.items = list(self.items)
        self.processed_items = list(self.processed_items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if self.config.process_on_the_fly and self.processed_items[idx] is None:
            self.processed_items[idx] = self._process_item(self.items[idx])
        return self.processed_items[idx]

    def __add__(self, other: "TokenizedProbingDataset") -> "TokenizedProbingDataset":
        """Concatenate two TokenizedProbingDataset instances."""
        if not isinstance(other, TokenizedProbingDataset):
            raise TypeError(f"Cannot concatenate TokenizedProbingDataset with {type(other)}")
        combined = TokenizedProbingDataset.__new__(TokenizedProbingDataset)
        combined.config = self.config
        combined.tokenizer = self.tokenizer
        combined.items = self.items + other.items
        combined.processed_items = self.processed_items + other.processed_items
        combined.debug_mode = False
        combined.print_first_example = False
        combined._num_skipped_spans = self._num_skipped_spans + other._num_skipped_spans
        combined._num_added_spans = self._num_added_spans + other._num_added_spans
        return combined


# ---------------------------------------------------------------------------
# Streaming / on-the-fly IterableDataset for large datasets
# ---------------------------------------------------------------------------

class StreamingProbingDataset(IterableDataset):
    """Memory-efficient streaming dataset for hallucination probe training.

    Unlike :class:`TokenizedProbingDataset`, which loads the full dataset
    into memory and pre-processes every item, this class wraps a HuggingFace
    **streaming** ``IterableDataset`` and tokenizes each item on-the-fly.
    This makes it possible to train on datasets that are too large to fit in
    RAM (e.g. multi-million-row corpora).

    Args:
        config:    Dataset configuration (same as :class:`TokenizedProbingDatasetConfig`).
        tokenizer: The model tokenizer.
        hf_dataset: An already-loaded HuggingFace ``IterableDataset``.  If
                    ``None``, the dataset will be loaded from ``config.hf_repo``
                    using the streaming API.
        shuffle_buffer: Number of examples to buffer for approximate shuffling.
                        Set to 0 to disable buffering (deterministic order).

    Example::

        import datasets as hf_datasets
        from probe.dataset import StreamingProbingDataset, TokenizedProbingDatasetConfig

        cfg = TokenizedProbingDatasetConfig(
            dataset_id=\"ragtruth\",
            hf_repo=\"wandb/RAGTruth\",
            split=\"train\",
            max_length=1024,
        )
        ds = StreamingProbingDataset(cfg, tokenizer=tokenizer, shuffle_buffer=1000)

        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    """

    def __init__(
        self,
        config: TokenizedProbingDatasetConfig,
        tokenizer: AutoTokenizer,
        hf_dataset: Optional[datasets.IterableDataset] = None,
        shuffle_buffer: int = 1000,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.shuffle_buffer = shuffle_buffer
        self.debug_mode = False

        self._prepare_fn = get_prepare_function(config.dataset_id)

        if hf_dataset is not None:
            self._hf_dataset = hf_dataset
        else:
            self._hf_dataset = datasets.load_dataset(
                config.hf_repo,
                config.subset,
                split=config.split,
                streaming=True,
            )

        if config.shuffle and shuffle_buffer > 0:
            self._hf_dataset = self._hf_dataset.shuffle(
                seed=config.seed,
                buffer_size=shuffle_buffer,
            )

    # ------------------------------------------------------------------
    # Internal helpers (mirrors TokenizedProbingDataset)
    # ------------------------------------------------------------------

    def _process_item(self, item: ProbingItem) -> Optional[Dict]:
        """Tokenize and label a single ProbingItem (on the fly)."""
        conversation = [
            {"role": "user",      "content": item.prompt},
            {"role": "assistant", "content": item.completion},
        ]
        full_text = self.tokenizer.apply_chat_template(conversation, tokenize=False)
        if self.tokenizer.bos_token and self.tokenizer.bos_token in full_text:
            full_text = full_text.replace(self.tokenizer.bos_token, "")

        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
            return_tensors="pt",
            padding_side="right",
        )
        input_ids      = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]

        # Reuse the parent class's labelling logic via a thin adapter
        _adapter = _StreamingAdapter(config=self.config, tokenizer=self.tokenizer)
        labels, weights, pos_spans, neg_spans = _adapter._compute_positional_labels(
            input_ids=input_ids, item=item
        )

        input_str = self.tokenizer.decode(input_ids)
        assistant_slice    = find_assistant_tokens_slice(input_ids, input_str, self.tokenizer)
        completion_start   = assistant_slice.stop

        lm_labels = input_ids.clone()
        lm_labels[:completion_start]       = -100
        lm_labels[attention_mask == 0]     = -100

        return {
            "input_ids":               input_ids,
            "attention_mask":          attention_mask,
            "classification_labels":   labels,
            "classification_weights":  weights,
            "pos_spans":               pos_spans,
            "neg_spans":               neg_spans,
            "lm_labels":               lm_labels,
        }

    # ------------------------------------------------------------------
    # IterableDataset protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Dict]:
        """Yield processed items one by one from the streaming source."""
        count = 0
        for row in self._hf_dataset:
            item = self._prepare_fn(row)
            if item is None:
                continue
            processed = self._process_item(item)
            if processed is None:
                continue
            yield processed
            count += 1
            if self.config.max_num_samples and count >= self.config.max_num_samples:
                break


class _StreamingAdapter:
    """Lightweight helper that exposes only the label-computation logic of
    :class:`TokenizedProbingDataset` without its full initialisation cost.
    Used internally by :class:`StreamingProbingDataset`.
    """

    def __init__(self, config: TokenizedProbingDatasetConfig, tokenizer: AutoTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self._num_skipped_spans = 0
        self._num_added_spans = 0
        self.debug_mode = False
        self.debug_mode = True 

    # Delegate to the same method body as TokenizedProbingDataset
    _compute_positional_labels = TokenizedProbingDataset._compute_positional_labels