"""Training script for hallucination detection probes.

Usage:
    python -m probe.train --config configs/train_config.yaml
    python -m probe.train --config configs/train_config.yaml --wandb

The script:
    1. Loads and tokenises training & evaluation datasets.
    2. Builds a ProbedModel (LLM + LoRA + ProbeHead).
    3. Trains with a combined probe loss + optional LM regularisation.
    4. Evaluates on all eval sets and saves metrics + ROC curves.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from utils.files_utlis import load_yaml, save_json
from utils.metrics import compute_clf_metrics, evaluate_predictions, plot_roc_curves, print_eval_metrics
from utils.model_utils import get_device, load_model_and_tokenizer, print_trainable_parameters, setup_lora_for_layers

from .config import TrainingConfig
from .dataset import TokenizedProbingDataset
from .dataset_converters import get_prepare_function
from .model import ProbedModel, ProbeHead


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset_split(
    config_dict: dict,
    tokenizer: AutoTokenizer,
) -> TokenizedProbingDataset:
    """Load a single dataset split from HuggingFace and prepare it."""
    import datasets as hf_datasets

    from .dataset import TokenizedProbingDatasetConfig

    ds_config = TokenizedProbingDatasetConfig(**config_dict)
    prepare_fn = get_prepare_function(ds_config.dataset_id)

    raw_ds = hf_datasets.load_dataset(
        ds_config.hf_repo,
        ds_config.subset,
        split=ds_config.split,
    )

    items = []
    for row in tqdm(raw_ds, desc=f"Converting {ds_config.dataset_id}"):
        item = prepare_fn(row)
        if item is not None:
            items.append(item)

    return TokenizedProbingDataset(items=items, config=ds_config, tokenizer=tokenizer)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: ProbedModel,
    dataloader: DataLoader,
    device: torch.device,
    dataset_id: str = "",
) -> Dict[str, float]:
    """Run evaluation and return a metrics dict (token-level + span-level)."""
    model.eval()

    all_probs: List[float] = []
    all_labels: List[float] = []
    all_preds: List[float] = []

    # Span-level accumulation
    all_pos_spans: List[List[int]] = []
    all_neg_spans: List[List[int]] = []
    token_offset: int = 0

    for batch in tqdm(dataloader, desc=f"Evaluating {dataset_id}"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(**batch)
        probs = outputs["probe_probs"].cpu().float()   # (batch, seq_len)
        labels = batch["classification_labels"].cpu().float()

        batch_size, seq_len = probs.shape

        valid_mask = labels != -100.0
        if not valid_mask.any():
            token_offset += batch_size * seq_len
            continue

        valid_probs = probs[valid_mask].numpy().flatten()
        valid_labels = labels[valid_mask].numpy().flatten()
        valid_preds = (valid_probs >= 0.5).astype(float)

        all_probs.extend(valid_probs.tolist())
        all_labels.extend(valid_labels.tolist())
        all_preds.extend(valid_preds.tolist())

        # Accumulate span indices with per-sample offset
        pos_spans_batch = batch.get("pos_spans", [])
        neg_spans_batch = batch.get("neg_spans", [])
        for sample_idx in range(batch_size):
            sample_offset = token_offset + sample_idx * seq_len
            if sample_idx < len(pos_spans_batch):
                for span in pos_spans_batch[sample_idx]:
                    all_pos_spans.append([i + sample_offset for i in span])
            if sample_idx < len(neg_spans_batch):
                for span in neg_spans_batch[sample_idx]:
                    all_neg_spans.append([i + sample_offset for i in span])

        token_offset += batch_size * seq_len

    if not all_labels:
        return {}

    probs_arr  = np.array(all_probs)
    labels_arr = np.array(all_labels)
    preds_arr  = np.array(all_preds)

    # Token-level metrics
    metrics: Dict = {}
    if len(np.unique(labels_arr)) >= 2:
        metrics["token"] = compute_clf_metrics(
            preds=preds_arr, labels=labels_arr, probs=probs_arr
        )

    # Span-level metrics (max aggregation, quick proxy for per-epoch logging)
    from utils.metrics import compute_span_level_metrics
    for agg in ("max", "mean"):
        span_m = compute_span_level_metrics(
            token_probs=probs_arr,
            pos_spans=all_pos_spans,
            neg_spans=all_neg_spans,
            threshold=0.5,
            aggregation=agg,
        )
        if span_m:
            metrics[f"span_{agg}"] = span_m

    return metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: TrainingConfig) -> None:
    device = get_device()
    print(f"Using device: {device}")

    # ---- Load model & tokenizer ----
    model_base, tokenizer = load_model_and_tokenizer(
        config.probe_config.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    # ---- Apply LoRA ----
    if config.probe_config.lora_layers:
        model_base = setup_lora_for_layers(
            model_base,
            layer_indices=config.probe_config.lora_layers,
            lora_r=config.probe_config.lora_r,
            lora_alpha=config.probe_config.lora_alpha,
            lora_dropout=config.probe_config.lora_dropout,
        )

    from utils.model_utils import get_model_hidden_size
    hidden_size = get_model_hidden_size(model_base)
    probe_head = ProbeHead(hidden_size=hidden_size)

    probed_model = ProbedModel(
        model=model_base,
        probe_head=probe_head,
        layer_idx=config.probe_config.layer,
    ).to(device)

    print_trainable_parameters(probed_model)

    # ---- Build datasets ----
    train_datasets = [
        load_dataset_split(cfg.__dict__, tokenizer)
        for cfg in config.train_dataset_configs
    ]
    train_dataset = train_datasets[0]
    for ds in train_datasets[1:]:
        train_dataset = train_dataset + ds

    eval_datasets = {
        cfg.dataset_id: load_dataset_split(cfg.__dict__, tokenizer)
        for cfg in config.eval_dataset_configs
    }

    def collate_fn(batch):
        """Stack tensors; keep list fields as lists."""
        out = {}
        for key in batch[0]:
            vals = [item[key] for item in batch if item is not None]
            if isinstance(vals[0], torch.Tensor):
                out[key] = torch.stack(vals)
            else:
                out[key] = vals
        return out

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # ---- Optimizer & scheduler ----
    probe_params = list(probed_model.probe_head.parameters())
    lora_params = [p for n, p in probed_model.model.named_parameters() if p.requires_grad]

    optimizer = AdamW(
        [
            {"params": probe_params, "lr": config.probe_head_lr},
            {"params": lora_params,  "lr": config.lora_lr},
        ],
        weight_decay=0.01,
    )

    total_steps = (
        config.max_steps
        if config.max_steps > 0
        else len(train_loader) * config.num_train_epochs
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps,
    )

    # ---- Optional W&B ----
    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_name or config.probe_config.probe_id,
                config=config.__dict__,
            )
        except ImportError:
            use_wandb = False

    # ---- Training ----
    global_step = 0
    best_auc = 0.0
    save_dir = Path("value_head_probes") / config.probe_config.probe_id

    probed_model.train()
    for epoch in range(config.num_train_epochs):
        epoch_probe_loss = 0.0
        epoch_lm_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_train_epochs}"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = probed_model(**batch)
            probe_loss = outputs["probe_loss"]
            lm_loss = outputs["lm_loss"]
            loss = probe_loss + config.lambda_lm * lm_loss

            # Gradient accumulation
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            if (global_step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(probed_model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_probe_loss += probe_loss.item()
            epoch_lm_loss += lm_loss.item()
            num_batches += 1
            global_step += 1

            if config.logging_steps > 0 and global_step % config.logging_steps == 0:
                log = {
                    "step": global_step,
                    "probe_loss": epoch_probe_loss / num_batches,
                    "lm_loss": epoch_lm_loss / num_batches,
                    "lr_probe": scheduler.get_last_lr()[0] if scheduler else config.probe_head_lr,
                }
                print(f"Step {global_step}: " + ", ".join(f"{k}={v:.4f}" for k, v in log.items() if isinstance(v, float)))
                if use_wandb:
                    wandb.log(log)

            if config.max_steps > 0 and global_step >= config.max_steps:
                break

        # ---- Per-epoch evaluation ----
        all_eval_metrics = {}
        for ds_id, eval_ds in eval_datasets.items():
            eval_loader = DataLoader(
                eval_ds,
                batch_size=config.per_device_eval_batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
            metrics = evaluate(probed_model, eval_loader, device, dataset_id=ds_id)
            # Flatten nested metrics for W&B / checkpoint comparison
            prefixed: Dict[str, float] = {}
            for level, level_metrics in metrics.items():
                if isinstance(level_metrics, dict):
                    for k, v in level_metrics.items():
                        prefixed[f"{ds_id}/{level}/{k}"] = v
                else:
                    prefixed[f"{ds_id}/{level}"] = level_metrics
            all_eval_metrics.update(prefixed)
            print_eval_metrics(metrics, metric_key_prefix=ds_id)

        if use_wandb and all_eval_metrics:
            wandb.log({"epoch": epoch + 1, **all_eval_metrics})

        # Save best checkpoint — use token-level AUC as the primary signal
        first_ds = list(eval_datasets.keys())[0] if eval_datasets else ""
        auc = all_eval_metrics.get(f"{first_ds}/token/auc", 0.0)
        if auc == 0.0:
            # Fallback: grab any AUC key present
            auc = next((v for k, v in all_eval_metrics.items() if k.endswith("/auc")), 0.0)
        if auc >= best_auc:
            best_auc = auc
            probed_model.save(save_dir)
            print(f"✓ Saved best model (AUC={best_auc:.4f}) to {save_dir}")

        if config.max_steps > 0 and global_step >= config.max_steps:
            break

    # ---- Final save ----
    probed_model.save(save_dir)
    if config.save_evaluation_metrics:
        save_json(all_eval_metrics, save_dir / "final_eval_metrics.json")

    if use_wandb:
        wandb.finish()

    print(f"\n✓ Training complete. Model saved to {save_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a hallucination detection probe")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML training config")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raw_cfg = load_yaml(args.config)
    cfg = TrainingConfig(**raw_cfg)
    if args.wandb:
        os.environ.setdefault("WANDB_API_KEY", os.environ.get("WANDB_API_KEY", ""))
    train(cfg)
