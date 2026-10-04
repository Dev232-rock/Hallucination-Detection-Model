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
from utils.metrics import compute_clf_metrics, evaluate_predictions, plot_roc_curves, print_eval_metrics
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
    """Run inference and collect token-level **and** span-level predictions."""
    model.eval()

    all_probs: List[float] = []
    all_labels: List[float] = []
    all_preds: List[float] = []

    # Span-level collection: accumulate across the full eval set
    all_pos_spans: List[List[int]] = []
    all_neg_spans: List[List[int]] = []
    # Running offset so that span indices remain globally unique
    token_offset: int = 0

    for batch in tqdm(dataloader, desc=f"Evaluating {dataset_id}"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(**batch)
        probs = outputs["probe_probs"].cpu().float()   # (batch, seq_len)
        labels = batch["classification_labels"].cpu().float()  # (batch, seq_len)

        batch_size, seq_len = probs.shape

        valid_mask = labels != -100.0
        if not valid_mask.any():
            token_offset += batch_size * seq_len
            continue

        valid_probs = probs[valid_mask].numpy().flatten().tolist()
        valid_labels = labels[valid_mask].numpy().flatten().tolist()
        valid_preds = [(p >= 0.5) * 1.0 for p in valid_probs]

        all_probs.extend(valid_probs)
        all_labels.extend(valid_labels)
        all_preds.extend(valid_preds)

        # Accumulate span indices with per-sample token offset
        pos_spans_batch = batch.get("pos_spans", [])  # list-of-lists (batch_size items)
        neg_spans_batch = batch.get("neg_spans", [])
        for sample_idx in range(batch_size):
            sample_offset = token_offset + sample_idx * seq_len
            if sample_idx < len(pos_spans_batch):
                for span in pos_spans_batch[sample_idx]:
                    all_pos_spans.append([idx + sample_offset for idx in span])
            if sample_idx < len(neg_spans_batch):
                for span in neg_spans_batch[sample_idx]:
                    all_neg_spans.append([idx + sample_offset for idx in span])

        token_offset += batch_size * seq_len

    if not all_labels:
        print(f"Warning: no valid labels found for dataset '{dataset_id}'")
        return {}

    import numpy as np
    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_labels)

    # Unified token + span metrics
    all_token_probs_full = probs_arr  # already filtered to valid tokens
    metrics = evaluate_predictions(
        token_probs=probs_arr,
        token_labels=labels_arr,          # all valid (no -100 here)
        pos_spans=all_pos_spans,
        neg_spans=all_neg_spans,
        threshold=0.5,
    )
    # evaluate_predictions expects -100 masking; since we've pre-filtered,
    # inject a clean token-level result directly
    if "token" not in metrics:
        from utils.metrics import compute_clf_metrics
        preds_arr = np.array(all_preds)
        if len(np.unique(labels_arr)) >= 2:
            metrics["token"] = compute_clf_metrics(
                preds=preds_arr, labels=labels_arr, probs=probs_arr
            )

    # Optionally save raw predictions
    if save_predictions and output_dir is not None:
        raw_path = output_dir / f"{dataset_id}_predictions.json"
        save_json(
            {
                "probs": all_probs,
                "labels": all_labels,
                "preds": all_preds,
                "pos_spans": all_pos_spans,
                "neg_spans": all_neg_spans,
            },
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
    # For ROC curves: collect per-dataset probs + labels at token level
    roc_probs:  Dict[str, List[float]] = {}
    roc_labels: Dict[str, List[float]] = {}

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

        # Collect token-level probs/labels for ROC plotting
        if config.save_roc_curves and "token" in metrics:
            raw_path = output_dir / f"{ds_config.dataset_id}_predictions.json"
            if raw_path.exists():
                import json
                with open(raw_path) as f:
                    raw = json.load(f)
                roc_probs[ds_config.dataset_id]  = raw.get("probs", [])
                roc_labels[ds_config.dataset_id] = raw.get("labels", [])

    # ---- Save summary metrics ----
    summary_path = output_dir / "evaluation_metrics.json"
    save_json(all_results, summary_path)
    print(f"\n✓ Metrics saved → {summary_path}")

    # ---- ROC curves ----
    if config.save_roc_curves and roc_probs:
        import numpy as np
        # plot_roc_curves expects {agg_level: [...]} dicts; we pass token-level
        # probs/labels for every dataset as the "all" bucket
        preds_dict  = {ds_id: (np.array(p) >= 0.5).astype(float).tolist()
                       for ds_id, p in roc_probs.items()}
        plot_roc_curves(
            all_preds={"all": [v for vs in preds_dict.values()  for v in vs]},
            all_labels={"all": [v for vs in roc_labels.values() for v in vs]},
            all_probs={"all": [v for vs in roc_probs.values()   for v in vs]},
            save_dir=str(output_dir),
            prefix="eval",
        )
        print(f"  ROC curves saved → {output_dir / 'eval_roc_curves.png'}")

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
