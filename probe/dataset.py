# Tokenized dataset classes with token-level labels for probe training.
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import torch
import datasets
from jaxtyping import Float, Int
from termcolor import colored
from torch import Tensor
from torch.utils.data import Dataset

rom tqdm import tqdm
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
        #Pre-process all items in the dataset.
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
    #Debug method to print how tokens have been labeled.

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
     # Computes positional labels for a sequence of tokens based on annotated spans.
     input_str: str = self.tokenizer.decode(input_ids)
        completion: str = item.completion
        
        positive_indices: List[int] = []    # indices of hallucinated spans
        negative_indices: List[int] = []    # indices of supported spans
        ignore_indices: List[int] = []      # indices to ignore in training
