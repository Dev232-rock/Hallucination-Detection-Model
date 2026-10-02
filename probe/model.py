"""Probe model architecture for hallucination detection.

Architecture overview:
    ProbedModel
    ├── base LLM  (frozen or LoRA-tuned via PEFT)
    ├── LoRA adapters  (optional, applied to selected layers)
    └── ProbeHead  (linear classifier on top of hidden states)

The ProbeHead is attached to a specified transformer layer via a forward hook,
intercepts the hidden states, and produces per-token hallucination scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import PreTrainedModel

from utils.model_utils import (
    get_model_hidden_size,
    get_model_layers,
    load_model_and_tokenizer,
    setup_lora_for_layers,
)
from utils.hooks import add_hooks
from .config import ProbeConfig


# ---------------------------------------------------------------------------
# Probe head
# ---------------------------------------------------------------------------

class ProbeHead(nn.Module):
    """Linear probe that maps hidden states → hallucination probability.

    Args:
        hidden_size: Dimensionality of the transformer hidden state.
        dropout:     Dropout probability applied before the linear layer.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)

        # Initialise weights near zero so the probe starts neutral
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute per-token hallucination logits.

        Args:
            hidden_states: Float tensor of shape ``(batch, seq_len, hidden_size)``.

        Returns:
            Logits of shape ``(batch, seq_len)`` (squeezed last dim).
        """
        x = self.dropout(hidden_states)
        return self.linear(x).squeeze(-1)  # (batch, seq_len)

    def save(self, path: Union[str, Path]) -> None:
        """Save probe head weights and config to *path*."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "probe_head.pt")
        with open(path / "probe_head_config.json", "w") as f:
            json.dump({"hidden_size": self.hidden_size}, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path], map_location: str = "cpu") -> "ProbeHead":
        """Load a saved ProbeHead from *path*."""
        path = Path(path)
        with open(path / "probe_head_config.json") as f:
            cfg = json.load(f)
        probe = cls(hidden_size=cfg["hidden_size"])
        probe.load_state_dict(
            torch.load(path / "probe_head.pt", map_location=map_location)
        )
        return probe


# ---------------------------------------------------------------------------
# Multi-layer probe head
# ---------------------------------------------------------------------------

class MultiLayerProbeHead(nn.Module):
    """Ensemble probe that aggregates activations from multiple transformer layers.

    Each layer contributes an independent linear projection; their outputs are
    combined with a learned weighted sum before the final sigmoid.

    Args:
        hidden_size:  Dimensionality of each layer's hidden state.
        num_layers:   Number of layers to aggregate.
        dropout:      Dropout probability.
    """

    def __init__(self, hidden_size: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        # One linear per layer
        self.layer_probes = nn.ModuleList(
            [nn.Linear(hidden_size, 1) for _ in range(num_layers)]
        )
        # Learned scalar weights for aggregation
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        for linear in self.layer_probes:
            nn.init.normal_(linear.weight, mean=0.0, std=0.02)
            nn.init.zeros_(linear.bias)

    def forward(self, hidden_states_list: List[torch.Tensor]) -> torch.Tensor:
        """Aggregate per-layer logits into a single per-token score.

        Args:
            hidden_states_list: List of tensors each shaped
                ``(batch, seq_len, hidden_size)``, one per monitored layer.

        Returns:
            Float tensor of shape ``(batch, seq_len)``.
        """
        assert len(hidden_states_list) == self.num_layers, (
            f"Expected {self.num_layers} layer activations, got {len(hidden_states_list)}"
        )
        weights = torch.softmax(self.layer_weights, dim=0)  # (num_layers,)
        logits = torch.stack(
            [probe(self.dropout(hs)).squeeze(-1) for probe, hs in
             zip(self.layer_probes, hidden_states_list)],
            dim=0,
        )  # (num_layers, batch, seq_len)
        return (weights[:, None, None] * logits).sum(dim=0)  # (batch, seq_len)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "multi_probe_head.pt")
        with open(path / "multi_probe_head_config.json", "w") as f:
            json.dump(
                {"hidden_size": self.hidden_size, "num_layers": self.num_layers},
                f, indent=2,
            )

    @classmethod
    def load(cls, path: Union[str, Path], map_location: str = "cpu") -> "MultiLayerProbeHead":
        path = Path(path)
        with open(path / "multi_probe_head_config.json") as f:
            cfg = json.load(f)
        probe = cls(hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"])
        probe.load_state_dict(
            torch.load(path / "multi_probe_head.pt", map_location=map_location)
        )
        return probe


# ---------------------------------------------------------------------------
# Full probed model
# ---------------------------------------------------------------------------

class ProbedModel(nn.Module):
    """Wrapper combining a (LoRA-adapted) LLM with a hallucination probe head.

    During a forward pass the base LLM runs normally; the hidden states from
    the configured layer(s) are captured via PyTorch hooks and fed through the
    probe head to produce per-token hallucination scores.

    Args:
        model:       The underlying HuggingFace causal LM (possibly PeftModel).
        probe_head:  Either a ``ProbeHead`` (single layer) or
                     ``MultiLayerProbeHead`` (multi-layer ensemble).
        layer_idx:   Which transformer layer to hook (for single-layer probe).
        layer_indices: Which transformer layers to hook (for multi-layer probe).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        probe_head: Union[ProbeHead, MultiLayerProbeHead],
        layer_idx: Optional[int] = None,
        layer_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.model = model
        self.probe_head = probe_head
        self.layer_idx = layer_idx
        self.layer_indices = layer_indices or ([layer_idx] if layer_idx is not None else [])

        self._captured_hidden_states: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: ProbeConfig) -> "ProbedModel":
        """Build a ProbedModel from a ProbeConfig, loading weights if needed."""
        from utils.model_utils import load_model_and_tokenizer, setup_lora_for_layers

        model, _tokenizer = load_model_and_tokenizer(config.model_name)

        # Apply LoRA if requested
        if config.lora_layers:
            model = setup_lora_for_layers(
                model,
                layer_indices=config.lora_layers,
                lora_r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
            )

        hidden_size = get_model_hidden_size(model)
        probe_head = ProbeHead(hidden_size=hidden_size)

        return cls(model=model, probe_head=probe_head, layer_idx=config.layer)

    # ------------------------------------------------------------------
    # Hidden-state capture hooks
    # ------------------------------------------------------------------

    def _make_hook(self, layer_capture_list: List[Optional[torch.Tensor]], slot: int):
        """Return a forward hook that stores hidden states into a specific slot."""
        def hook(module, input, output):
            # output is typically a tuple; first element is the hidden state
            hidden = output[0] if isinstance(output, tuple) else output
            layer_capture_list[slot] = hidden.detach()
        return hook

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        classification_labels: Optional[torch.Tensor] = None,
        classification_weights: Optional[torch.Tensor] = None,
        lm_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Run the LLM + probe and compute losses.

        Returns a dict with keys:
            - ``logits``          – LM logits, shape ``(batch, seq_len, vocab)``
            - ``probe_logits``    – per-token hallucination logits, ``(batch, seq_len)``
            - ``probe_probs``     – sigmoid of probe_logits
            - ``probe_loss``      – weighted binary cross-entropy (if labels provided)
            - ``lm_loss``         – language modelling cross-entropy (if lm_labels provided)
            - ``loss``            – total loss (sum of probe_loss + lm_loss)
        """
        layers = get_model_layers(self.model)
        num_layers_to_hook = len(self.layer_indices)
        captured: List[Optional[torch.Tensor]] = [None] * num_layers_to_hook

        # Build hook list
        forward_hooks = [
            (layers[idx], self._make_hook(captured, slot))
            for slot, idx in enumerate(self.layer_indices)
        ]

        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=forward_hooks):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                **kwargs,
            )

        lm_logits = outputs.logits  # (batch, seq_len, vocab)

        # ---- Probe forward ----
        if isinstance(self.probe_head, MultiLayerProbeHead):
            assert all(c is not None for c in captured), "Some layer hooks did not fire"
            probe_logits = self.probe_head(captured)
        else:
            # Single-layer probe — use the first (and only) captured state
            assert captured[0] is not None, "Layer hook did not fire"
            probe_logits = self.probe_head(captured[0])

        probe_probs = torch.sigmoid(probe_logits)

        result: Dict[str, torch.Tensor] = {
            "logits": lm_logits,
            "probe_logits": probe_logits,
            "probe_probs": probe_probs,
        }

        # ---- Probe loss (weighted BCE) ----
        probe_loss = torch.tensor(0.0, device=input_ids.device)
        if classification_labels is not None:
            valid_mask = classification_labels != -100.0
            if valid_mask.any():
                valid_logits = probe_logits[valid_mask]
                valid_labels = classification_labels[valid_mask].float()
                valid_weights = (
                    classification_weights[valid_mask].float()
                    if classification_weights is not None
                    else torch.ones_like(valid_labels)
                )
                bce = nn.functional.binary_cross_entropy_with_logits(
                    valid_logits, valid_labels, weight=valid_weights, reduction="mean"
                )
                probe_loss = bce
        result["probe_loss"] = probe_loss

        # ---- LM loss ----
        lm_loss = torch.tensor(0.0, device=input_ids.device)
        if lm_labels is not None:
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = lm_labels[:, 1:].contiguous()
            lm_loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        result["lm_loss"] = lm_loss

        result["loss"] = probe_loss + lm_loss
        return result

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save the probe head (and LoRA adapters if present) to *path*."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.probe_head.save(path)
        # Save LoRA adapters if the model is a PeftModel
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(str(path / "lora_adapters"))
        # Save layer indices config
        with open(path / "probed_model_config.json", "w") as f:
            json.dump(
                {
                    "layer_idx": self.layer_idx,
                    "layer_indices": self.layer_indices,
                },
                f, indent=2,
            )

    @classmethod
    def load(
        cls,
        config: ProbeConfig,
        path: Union[str, Path],
        map_location: str = "cpu",
    ) -> "ProbedModel":
        """Load a saved ProbedModel from *path*."""
        path = Path(path)
        probed = cls.from_config(config)

        # Load probe head weights
        if (path / "probe_head.pt").exists():
            probed.probe_head = ProbeHead.load(path, map_location=map_location)
        elif (path / "multi_probe_head.pt").exists():
            probed.probe_head = MultiLayerProbeHead.load(path, map_location=map_location)

        # Load LoRA adapters if saved
        lora_path = path / "lora_adapters"
        if lora_path.exists() and isinstance(probed.model, PeftModel):
            probed.model.load_adapter(str(lora_path), adapter_name="default")

        return probed
