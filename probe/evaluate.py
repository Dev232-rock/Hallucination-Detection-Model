"""Evaluation script for hallucination detection probes.

Usage:
    python -m probe.evaluate --config configs/eval_config.yaml
    python -m probe.evaluate --config configs/eval_config.yaml --output_dir results/

The script:
    1. Loads a trained ProbedModel from disk or HuggingFace Hub.
    2. Runs inference on every configured eval dataset.
    3. Saves metrics (JSON), ROC curves (PNG), and optionally raw predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.files_utlis import load_yaml, save_json
from utils.metrics import compute_clf_metrics, plot_roc_curves, print_eval_metrics
from utils.model_utils import get_device, load_model_and_tokenizer
from utils.probe_loader import download_probe_from_hf

from .config import EvaluationConfig
from .model import ProbedModel
from .train import load_dataset_split  # reuse helper


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_dataset(
    model: ProbedModel,
    dataloader: DataLoader,
    device: torch.device,
    dataset_id: str = "",
    save_predictions: bool = False,
    output_dir: Path = None,
) -> Dict:
    """Run inference and collect token-level predictions."""
    model.eval()

    all_probs: List[float] = []
    all_labels: List[float] = []
    all_preds: List[float] = []

    for batch in tqdm(dataloader, desc=f"Evaluating {dataset_id}"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(**batch)
        probs = outputs["probe_probs"].cpu().float()
        labels = batch["classification_labels"].cpu().float()

        valid_mask = labels != -100.0
        if not valid_mask.any():
            continue

        valid_probs = probs[valid_mask].numpy().flatten().tolist()
        valid_labels = labels[valid_mask].numpy().flatten().tolist()
        valid_preds = [(p >= 0.5) * 1.0 for p in valid_probs]

        all_probs.extend(valid_probs)
        all_labels.extend(valid_labels)
        all_preds.extend(valid_preds)

    if not all_labels:
        print(f"Warning: no valid labels found for dataset '{dataset_id}'")
        return {}

    metrics = compute_clf_metrics(
        preds=np.array(all_preds),
        labels=np.array(all_labels),
        probs=np.array(all_probs),
    )

    # Optionally save raw predictions
    if save_predictions and output_dir is not None:
        raw_path = output_dir / f"{dataset_id}_predictions.json"
        save_json(
            {"probs": all_probs, "labels": all_labels, "preds": all_preds},
            raw_path,
        )
        print(f"  Saved raw predictions → {raw_path}")

    return metrics


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def evaluate(config: EvaluationConfig) -> Dict[str, Dict]:
    device = get_device()
    print(f"Using device: {device}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model weights ----
    if config.probe_config.load_from == "hf":
        print(f"Downloading probe from HuggingFace: {config.probe_config.hf_repo_id}")
        download_probe_from_hf(
            repo_id=config.probe_config.hf_repo_id,
            probe_id=config.probe_config.probe_id,
        )

    _base_model, tokenizer = load_model_and_tokenizer(config.probe_config.model_name)
    probed_model = ProbedModel.load(
        config=config.probe_config,
        path=config.probe_config.probe_path,
    ).to(device)

    def collate_fn(batch):
        out = {}
        for key in batch[0]:
            vals = [item[key] for item in batch if item is not None]
            if isinstance(vals[0], torch.Tensor):
                out[key] = torch.stack(vals)
            else:
                out[key] = vals
        return out

    # ---- Evaluate each dataset ----
    all_results: Dict[str, Dict] = {}
    roc_data: Dict[str, Dict] = {}

    for ds_config in config.dataset_configs:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_config.dataset_id}")
        print(f"{'='*60}")

        ds = load_dataset_split(ds_config.__dict__, tokenizer)
        loader = DataLoader(
            ds,
            batch_size=config.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        metrics = evaluate_dataset(
            model=probed_model,
            dataloader=loader,
            device=device,
            dataset_id=ds_config.dataset_id,
            save_predictions=config.save_predictions,
            output_dir=output_dir,
        )

        print_eval_metrics(metrics, metric_key_prefix=ds_config.dataset_id)
        all_results[ds_config.dataset_id] = metrics

    # ---- Save summary metrics ----
    summary_path = output_dir / "evaluation_metrics.json"
    save_json(all_results, summary_path)
    print(f"\n✓ Metrics saved → {summary_path}")

    # ---- ROC curves ----
    if config.save_roc_curves:
        for ds_id, metrics in all_results.items():
            auc = metrics.get("auc")
            if auc is not None:
                print(f"  {ds_id}: AUC = {auc:.4f}")
        print("  (ROC curve plotting requires raw probs/labels; run with save_predictions=True)")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a hallucination detection probe")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML evaluation config")
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Override output directory for results"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raw_cfg = load_yaml(args.config)
    if args.output_dir:
        raw_cfg["output_dir"] = args.output_dir
    cfg = EvaluationConfig(**raw_cfg)
    evaluate(cfg)
