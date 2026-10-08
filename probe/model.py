"""Probe model architecture for hallucination detection with multi-signal ensemble scoring.

Architecture overview:
    ProbedModel
    ├── base LLM  (frozen or LoRA-tuned via PEFT)
    ├── LoRA adapters  (optional, applied to selected layers)
    ├── MultiLayerProbeHead / ProbeHead  (internal hidden state classifier)
    └── MultiSignalScorer  (ensembles probe logits, attention dispersion, and logit entropy)

The system supports:
1. Hidden state linear probes on single or multiple layers.
2. Dynamic learned layer aggregation with per-layer attribution tracking.
3. Multi-signal scoring combining:
   - Internal representation probe probabilities (latent factuality awareness)
   - Next-token predictive entropy & confidence margins (generation uncertainty)
   - Cross-layer attention dispersion & context decoupling (attention sink / drift)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    PeftModel = type("PeftModel", (), {})
    HAS_PEFT = False
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
# Multi-signal scoring engine
# ---------------------------------------------------------------------------

class MultiSignalScorer:
    """Helper module for computing and ensembling multi-modal hallucination signals.

    Combines:
    1. Probe representation probability (internal activation state).
    2. Logit predictive entropy & confidence margin (next-token uncertainty).
    3. Attention dispersion & context decoupling (divergence from prompt grounding).
    """

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "probe": 0.55,
        "entropy": 0.25,
        "attention": 0.20,
    }

    @staticmethod
    def compute_token_entropy(
        logits: torch.Tensor,
        temperature: float = 1.0,
        normalize: bool = True,
        eps: float = 1e-9,
    ) -> torch.Tensor:
        """Compute normalized Shannon predictive entropy per token.

        Higher values indicate high model uncertainty, which strongly correlates
        with hallucinated entities, guessing, or lack of parametric knowledge.

        Args:
            logits: Float tensor of shape ``(batch, seq_len, vocab_size)``.
            temperature: Softmax scaling temperature.
            normalize: If True, normalize by log(vocab_size) to constrain to [0, 1].
            eps: Numerical stability constant.

        Returns:
            Float tensor of shape ``(batch, seq_len)`` with values in [0.0, 1.0].
        """
        scaled_logits = logits / max(temperature, 1e-4)
        probs = F.softmax(scaled_logits, dim=-1)
        log_probs = F.log_softmax(scaled_logits, dim=-1)

        # Shannon entropy H = - sum(p * log(p))
        entropy = -torch.sum(probs * log_probs, dim=-1)  # (batch, seq_len)

        if normalize:
            vocab_size = logits.shape[-1]
            max_entropy = torch.log(torch.tensor(float(vocab_size), device=logits.device))
            norm_entropy = entropy / (max_entropy + eps)
            return torch.clamp(norm_entropy, 0.0, 1.0)

        return entropy

    @staticmethod
    def compute_attention_dispersion(
        attentions: Optional[Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]],
        prompt_length: int = 0,
        target_layers: Optional[List[int]] = None,
        eps: float = 1e-9,
    ) -> Optional[torch.Tensor]:
        """Compute attention dispersion and context decoupling score.

        When models generate grounded facts, attention focuses on context tokens.
        When hallucinating, attention often decouples from the prompt and disperses
        into uniform background noise or concentrates unnaturally on delimiters.

        Args:
            attentions: Tuple of layer attention tensors, each ``(batch, heads, seq_len, seq_len)``.
            prompt_length: Index dividing prompt context from assistant completion.
            target_layers: Optional subset of layer indices to inspect.
            eps: Numerical stability constant.

        Returns:
            Float tensor of shape ``(batch, seq_len)`` in [0.0, 1.0], or None if attentions unavailable.
        """
        if attentions is None or len(attentions) == 0:
            return None

        # Determine which layers to pool (default to top third of layers where context routing happens)
        num_layers = len(attentions)
        if target_layers is None:
            start_l = max(0, num_layers // 2)
            target_layers = list(range(start_l, num_layers))

        selected = [attentions[idx] for idx in target_layers if idx < num_layers and attentions[idx] is not None]
        if not selected:
            return None

        # Mean across selected layers and attention heads -> (batch, seq_len, seq_len)
        stacked = torch.stack(selected, dim=0)  # (num_sel, batch, heads, seq_len, seq_len)
        mean_attn = stacked.mean(dim=(0, 2))     # (batch, seq_len, seq_len)

        batch_size, seq_len, _ = mean_attn.shape
        dispersion_scores = torch.zeros((batch_size, seq_len), device=mean_attn.device, dtype=torch.float32)

        # For tokens strictly before prompt_length, dispersion is considered neutral (0.0)
        valid_prompt_len = max(1, min(prompt_length, seq_len))

        for t in range(valid_prompt_len, seq_len):
            # Query attention distribution over all past tokens 0..t
            attn_slice = mean_attn[:, t, : t + 1]  # (batch, t + 1)
            row_sum = attn_slice.sum(dim=-1, keepdim=True) + eps
            norm_attn = attn_slice / row_sum

            # 1. Context attention ratio: fraction of attention directed to the prompt
            context_attn = norm_attn[:, :valid_prompt_len].sum(dim=-1)  # (batch,)
            context_decoupling = 1.0 - torch.clamp(context_attn, 0.0, 1.0)

            # 2. Query attention entropy (diffusion)
            attn_entropy = -torch.sum(norm_attn * torch.log(norm_attn + eps), dim=-1)
            max_entropy = torch.log(torch.tensor(float(t + 1), device=mean_attn.device)) + eps
            norm_entropy = torch.clamp(attn_entropy / max_entropy, 0.0, 1.0)

            # Blended dispersion score: 60% context decoupling + 40% distribution diffusion
            token_dispersion = 0.60 * context_decoupling + 0.40 * norm_entropy
            dispersion_scores[:, t] = torch.clamp(token_dispersion, 0.0, 1.0)

        return dispersion_scores

    @classmethod
    def fuse_signals(
        cls,
        probe_probs: torch.Tensor,
        entropy_scores: torch.Tensor,
        attention_dispersion: Optional[torch.Tensor] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Combine probe probability, predictive entropy, and attention dispersion.

        Args:
            probe_probs: Tensor of shape ``(batch, seq_len)`` with values in [0, 1].
            entropy_scores: Tensor of shape ``(batch, seq_len)`` with values in [0, 1].
            attention_dispersion: Optional tensor of shape ``(batch, seq_len)``.
            weights: Optional custom weight mapping for the signals.

        Returns:
            Tuple of (ensemble_scores, signal_dict).
        """
        w = dict(weights or cls.DEFAULT_WEIGHTS)
        w_probe = float(w.get("probe", 0.55))
        w_entropy = float(w.get("entropy", 0.25))
        w_attn = float(w.get("attention", 0.20))

        if attention_dispersion is None:
            # Rebalance weights between probe and entropy if attention is disabled/unavailable
            total = w_probe + w_entropy
            if total > 0:
                w_probe /= total
                w_entropy /= total
            else:
                w_probe, w_entropy = 0.70, 0.30
            w_attn = 0.0
            attn_tensor = torch.zeros_like(probe_probs)
        else:
            total = w_probe + w_entropy + w_attn
            if total > 0:
                w_probe /= total
                w_entropy /= total
                w_attn /= total
            attn_tensor = attention_dispersion

        ensemble = (
            w_probe * probe_probs
            + w_entropy * entropy_scores
            + w_attn * attn_tensor
        )
        ensemble = torch.clamp(ensemble, 0.0, 1.0)

        breakdown = {
            "probe": probe_probs,
            "entropy": entropy_scores,
            "attention": attn_tensor,
            "ensemble": ensemble,
        }
        return ensemble, breakdown


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
# Multi-layer probe head with attribution
# ---------------------------------------------------------------------------

class MultiLayerProbeHead(nn.Module):
    """Ensemble probe that aggregates activations from multiple transformer layers.

    Each monitored layer contributes an independent linear projection; their outputs
    are combined via a learned softmax weighted sum with per-layer attribution tracking.

    Args:
        hidden_size:  Dimensionality of each layer's hidden state.
        num_layers:   Number of layers to aggregate.
        dropout:      Dropout probability.
        fusion_type:  Fusion method ('weighted_sum' or 'gated').
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.1,
        fusion_type: str = "weighted_sum",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fusion_type = fusion_type
        self.dropout = nn.Dropout(dropout)

        # One linear projection per layer
        self.layer_probes = nn.ModuleList(
            [nn.Linear(hidden_size, 1) for _ in range(num_layers)]
        )
        # Learned scalar weights for layer importance
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        for linear in self.layer_probes:
            nn.init.normal_(linear.weight, mean=0.0, std=0.02)
            nn.init.zeros_(linear.bias)

    def forward(
        self,
        hidden_states_list: List[torch.Tensor],
        return_layer_breakdown: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Aggregate per-layer logits into a composite per-token score.

        Args:
            hidden_states_list: List of tensors each shaped
                ``(batch, seq_len, hidden_size)``, one per monitored layer.
            return_layer_breakdown: If True, also returns individual layer logits and normalized weights.

        Returns:
            If return_layer_breakdown is False:
                Float tensor of shape ``(batch, seq_len)`` with aggregated logits.
            If return_layer_breakdown is True:
                Tuple of (aggregated_logits, per_layer_logits, layer_weights).
        """
        assert len(hidden_states_list) == self.num_layers, (
            f"Expected {self.num_layers} layer activations, got {len(hidden_states_list)}"
        )
        weights = torch.softmax(self.layer_weights, dim=0)  # (num_layers,)
        per_layer_logits = torch.stack(
            [probe(self.dropout(hs)).squeeze(-1) for probe, hs in
             zip(self.layer_probes, hidden_states_list)],
            dim=0,
        )  # (num_layers, batch, seq_len)

        aggregated_logits = (weights[:, None, None] * per_layer_logits).sum(dim=0)  # (batch, seq_len)

        if return_layer_breakdown:
            return aggregated_logits, per_layer_logits, weights
        return aggregated_logits

    def get_layer_attributions(
        self,
        hidden_states_list: List[torch.Tensor],
        layer_indices: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Compute layer-by-layer hallucination probabilities and importance weights.

        Args:
            hidden_states_list: Hidden states captured for each monitored layer.
            layer_indices: Optional list of integer layer indices for labeling.

        Returns:
            Dictionary containing:
                - 'layer_weights': List of float softmax weights per layer.
                - 'layer_probs': Dict mapping layer index (str) -> tensor of probabilities.
        """
        with torch.no_grad():
            weights = torch.softmax(self.layer_weights, dim=0).cpu().tolist()
            per_layer_logits = torch.stack(
                [probe(hs).squeeze(-1) for probe, hs in zip(self.layer_probes, hidden_states_list)],
                dim=0,
            )
            per_layer_probs = torch.sigmoid(per_layer_logits)  # (num_layers, batch, seq_len)

        labels = layer_indices if (layer_indices and len(layer_indices) == self.num_layers) else list(range(self.num_layers))
        layer_prob_dict = {}
        for i, lbl in enumerate(labels):
            layer_prob_dict[str(lbl)] = per_layer_probs[i]

        return {
            "layer_weights": {str(lbl): float(weights[i]) for i, lbl in enumerate(labels)},
            "layer_probs": layer_prob_dict,
        }

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "multi_probe_head.pt")
        with open(path / "multi_probe_head_config.json", "w") as f:
            json.dump(
                {
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "fusion_type": self.fusion_type,
                },
                f, indent=2,
            )

    @classmethod
    def load(cls, path: Union[str, Path], map_location: str = "cpu") -> "MultiLayerProbeHead":
        path = Path(path)
        with open(path / "multi_probe_head_config.json") as f:
            cfg = json.load(f)
        probe = cls(
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            fusion_type=cfg.get("fusion_type", "weighted_sum"),
        )
        probe.load_state_dict(
            torch.load(path / "multi_probe_head.pt", map_location=map_location)
        )
        return probe


# ---------------------------------------------------------------------------
# Full probed model
# ---------------------------------------------------------------------------

class ProbedModel(nn.Module):
    """Wrapper combining a (LoRA-adapted) LLM with an advanced multi-signal hallucination probe.

    During inference and evaluation:
    - Base LLM computes causal language representations.
    - Probe head intercepts internal hidden states across single or multiple layers.
    - MultiSignalScorer computes predictive logit entropy and attention dispersion.
    - An ensemble score is produced, giving calibrated detection with layer attribution.

    Args:
        model:         The underlying HuggingFace causal LM (possibly PeftModel).
        probe_head:    Either a ``ProbeHead`` (single layer) or ``MultiLayerProbeHead``.
        layer_idx:     Which transformer layer to hook (for single-layer probe).
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
        self.scorer = MultiSignalScorer()

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
        output_attentions: bool = False,
        compute_multi_signal: bool = False,
        prompt_length: Optional[int] = None,
        signal_weights: Optional[Dict[str, float]] = None,
        return_layer_breakdown: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run the LLM + probe and compute multi-signal scores and losses.

        Returns a dict with keys:
            - ``logits``                      – LM logits, shape ``(batch, seq_len, vocab)``
            - ``probe_logits``                – per-token probe logits, ``(batch, seq_len)``
            - ``probe_probs``                 – sigmoid of probe_logits
            - ``entropy_scores``              – predictive uncertainty [0..1] (if compute_multi_signal)
            - ``attention_dispersion_scores``  – attention dispersion [0..1] (if compute_multi_signal)
            - ``ensemble_scores``             – fused multi-signal score (if compute_multi_signal)
            - ``layer_attributions``          – per-layer scores & weights (if multi-layer probe)
            - ``probe_loss``                  – weighted BCE loss
            - ``lm_loss``                     – language modeling cross-entropy
            - ``loss``                        – total loss
        """
        layers = get_model_layers(self.model)
        num_layers_to_hook = len(self.layer_indices)
        captured: List[Optional[torch.Tensor]] = [None] * num_layers_to_hook

        # Hook registered layers
        forward_hooks = [
            (layers[idx], self._make_hook(captured, slot))
            for slot, idx in enumerate(self.layer_indices)
        ]

        need_attentions = bool(output_attentions or compute_multi_signal)

        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=forward_hooks):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=need_attentions,
                output_hidden_states=False,
                **kwargs,
            )

        lm_logits = outputs.logits  # (batch, seq_len, vocab)

        # ---- Probe forward ----
        layer_attributions_data: Optional[Dict[str, Any]] = None

        if isinstance(self.probe_head, MultiLayerProbeHead):
            assert all(c is not None for c in captured), "Some layer hooks did not fire"
            if return_layer_breakdown or compute_multi_signal:
                probe_logits, per_layer_logits, norm_layer_weights = self.probe_head(
                    captured, return_layer_breakdown=True
                )
                layer_attributions_data = {
                    "layer_weights": {str(lbl): float(norm_layer_weights[i]) for i, lbl in enumerate(self.layer_indices)},
                    "layer_probs": {
                        str(lbl): torch.sigmoid(per_layer_logits[i])
                        for i, lbl in enumerate(self.layer_indices)
                    },
                }
            else:
                probe_logits = self.probe_head(captured)
        else:
            assert captured[0] is not None, "Layer hook did not fire"
            probe_logits = self.probe_head(captured[0])

        probe_probs = torch.sigmoid(probe_logits)

        result: Dict[str, Any] = {
            "logits": lm_logits,
            "probe_logits": probe_logits,
            "probe_probs": probe_probs,
        }

        if layer_attributions_data is not None:
            result["layer_attributions"] = layer_attributions_data

        # ---- Multi-signal ensemble scoring ----
        if compute_multi_signal:
            # 1. Logit predictive entropy
            entropy_scores = self.scorer.compute_token_entropy(lm_logits)
            result["entropy_scores"] = entropy_scores

            # 2. Attention dispersion & context decoupling
            attentions = getattr(outputs, "attentions", None)
            actual_prompt_len = prompt_length or 0
            attn_dispersion = self.scorer.compute_attention_dispersion(
                attentions, prompt_length=actual_prompt_len
            )
            result["attention_dispersion_scores"] = attn_dispersion

            # 3. Fuse signals
            ensemble_scores, signal_breakdown = self.scorer.fuse_signals(
                probe_probs=probe_probs,
                entropy_scores=entropy_scores,
                attention_dispersion=attn_dispersion,
                weights=signal_weights,
            )
            result["ensemble_scores"] = ensemble_scores
            result["signal_breakdown"] = signal_breakdown

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
                bce = F.binary_cross_entropy_with_logits(
                    valid_logits, valid_labels, weight=valid_weights, reduction="mean"
                )
                probe_loss = bce
        result["probe_loss"] = probe_loss

        # ---- LM loss ----
        lm_loss = torch.tensor(0.0, device=input_ids.device)
        if lm_labels is not None:
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = lm_labels[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
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
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(str(path / "lora_adapters"))
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
