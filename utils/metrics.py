import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve


def compute_clf_metrics(
    preds: np.ndarray,
    labels: np.ndarray, 
    probs: np.ndarray
) -> Dict[str, float]:
    # Compute classification metrics
    assert all((labels == 0.0) | (labels == 1.0)), "labels must be either 0 or 1"

 
    # Basic classification metrics
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auc_score = roc_auc_score(labels, probs) if len(np.unique(labels)) == 2 else float('nan')

    # Find optimal threshold
    optimal_threshold = float('nan')
    threshold_optimized_accuracy = float('nan')
    recall_at_01_fpr = float('nan')

    if len(np.unique(labels)) == 2:
        # ROC curve
        fpr, tpr, thresholds = roc_curve(labels, probs)
        # Find optimal threshold for accuracy
        unique_probs = np.unique(probs)
        if len(unique_probs) > 100:
            percentiles = np.linspace(0, 100, 100)
            threshold_candidates = np.percentile(unique_probs, percentiles)
        else:
            threshold_candidates = unique_probs
        
        best_accuracy = 0.0
        optimal_threshold = 0.5

        for threshold in threshold_candidates:
            y_pred = (probs >= threshold).astype(int)
            acc = accuracy_score(labels, y_pred)
            if acc > best_accuracy:
                best_accuracy = acc
                optimal_threshold = threshold
        
        threshold_optimized_accuracy = best_accuracy

        # Calculate recall at 0.1 FPR
        target_fpr = 0.1
        idx = np.where(fpr <= target_fpr)[0]
        if len(idx) > 0:
            recall_at_01_fpr = tpr[idx[-1]]
        else:
            recall_at_01_fpr = 0.0
        
        #Calculate recall at 0.6 FPR
        target_fpr = 0.6
        idx = np.where(fpr <= target_fpr)[0]
        if len(idx) > 0:
            recall_at_06_fpr = tpr[idx[-1]]
        else:
            recall_at_06_fpr = 0.1

    # Count distributions
    true_positive_count = int(np.sum(labels == 1.0))
    true_negative_count = int(np.sum(labels == 0.0))
    pred_positive_count = int(np.sum(preds == 1.0))
    pred_negative_count = int(np.sum(preds == 0.0))
    total_samples = len(labels)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc_score),
        "optimal_threshold": float(optimal_threshold),
        "threshold_optimized_accuracy": float(threshold_optimized_accuracy),
        "recall_at_0.1_fpr": float(recall_at_01_fpr),
        "recall_at_0.6_fpr": float(recall_at_06_fpr),
        "true_positive_count": true_positive_count,
        "true_negative_count": true_negative_count,
        "pred_positive_count": pred_positive_count,
        "pred_negative_count": pred_negative_count,
        "total_samples": total_samples
    }

def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute evaluation metrics (thin wrapper around compute_clf_metrics)."""
    if probabilities is None:
        probabilities = predictions
    return compute_clf_metrics(predictions, labels, probabilities)


def bootstrap_confidence_intervals(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Compute bootstrap confidence intervals for key classification metrics.

    Resamples ``(probs, labels)`` with replacement *n_bootstrap* times,
    computes each metric on every resample, then reports the
    ``[alpha/2, 1-alpha/2]`` percentiles as the CI.

    Args:
        probs:       1-D array of predicted probabilities.
        labels:      1-D binary ground-truth labels (0 or 1).
        threshold:   Decision threshold for converting probs to binary preds.
        n_bootstrap: Number of bootstrap iterations (default 1000).
        alpha:       Significance level; 0.05 gives 95% CIs.
        seed:        Random seed for reproducibility.

    Returns:
        Dict mapping metric name → ``(lower_bound, upper_bound)``.
        Metrics included: ``auc``, ``f1``, ``precision``, ``recall``,
        ``accuracy``.

    Example::

        cis = bootstrap_confidence_intervals(probs, labels)
        print(f"AUC: {auc:.4f}  95% CI [{cis['auc'][0]:.4f}, {cis['auc'][1]:.4f}]")
    """
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score, recall_score, accuracy_score,
    )

    rng = np.random.default_rng(seed)
    n = len(labels)

    metric_samples: Dict[str, List[float]] = {
        "auc": [], "f1": [], "precision": [], "recall": [], "accuracy": [],
    }

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        b_probs  = probs[idx]
        b_labels = labels[idx]
        b_preds  = (b_probs >= threshold).astype(float)

        # Skip resamples with only one class (AUC undefined)
        if len(np.unique(b_labels)) < 2:
            continue

        metric_samples["auc"].append(roc_auc_score(b_labels, b_probs))
        metric_samples["f1"].append(f1_score(b_labels, b_preds, zero_division=0))
        metric_samples["precision"].append(precision_score(b_labels, b_preds, zero_division=0))
        metric_samples["recall"].append(recall_score(b_labels, b_preds, zero_division=0))
        metric_samples["accuracy"].append(accuracy_score(b_labels, b_preds))

    lo_pct = 100.0 * (alpha / 2)
    hi_pct = 100.0 * (1.0 - alpha / 2)

    cis: Dict[str, Tuple[float, float]] = {}
    for metric, samples in metric_samples.items():
        if not samples:
            cis[metric] = (float("nan"), float("nan"))
        else:
            arr = np.array(samples)
            cis[metric] = (float(np.percentile(arr, lo_pct)), float(np.percentile(arr, hi_pct)))

    return cis


def compute_span_level_metrics(
    token_probs: np.ndarray,
    pos_spans: List[List[int]],
    neg_spans: List[List[int]],
    threshold: float = 0.5,
    aggregation: str = "max",
    compute_ci: bool = False,
    n_bootstrap: int = 1000,
) -> Dict[str, any]:
    """Compute span-level hallucination detection metrics.

    For each span (positive = hallucinated, negative = factual) we aggregate
    the per-token probabilities within that span into a single score, then
    compute standard binary classification metrics at the span level.

    Args:
        token_probs:  1-D array of per-token hallucination probabilities for
                      the *entire* sequence (length == seq_len).
        pos_spans:    List of positive (hallucinated) spans.  Each element is
                      a list of **token indices** belonging to that span.
        neg_spans:    List of negative (factual) spans.  Same format.
        threshold:    Decision threshold applied to span scores.
        aggregation:  How to pool token probs within a span — ``"max"`` or
                      ``"mean"``.
        compute_ci:   Whether to compute bootstrap confidence intervals.
        n_bootstrap:  Number of bootstrap iterations for CI calculation.

    Returns:
        Dict with the same keys as :func:`compute_clf_metrics`, prefixed with
        nothing (caller can add a prefix).  Returns an empty dict when there
        are no spans with both classes.
    """
    if not pos_spans and not neg_spans:
        return {}

    agg_fn = np.max if aggregation == "max" else np.mean

    span_probs: List[float] = []
    span_labels: List[float] = []

    for indices in pos_spans:
        if not indices:
            continue
        valid = [i for i in indices if i < len(token_probs)]
        if not valid:
            continue
        span_probs.append(float(agg_fn(token_probs[valid])))
        span_labels.append(1.0)

    for indices in neg_spans:
        if not indices:
            continue
        valid = [i for i in indices if i < len(token_probs)]
        if not valid:
            continue
        span_probs.append(float(agg_fn(token_probs[valid])))
        span_labels.append(0.0)

    if not span_labels:
        return {}

    probs_arr = np.array(span_probs)
    labels_arr = np.array(span_labels)
    preds_arr = (probs_arr >= threshold).astype(float)

    # Need at least one positive and one negative to compute AUC
    if len(np.unique(labels_arr)) < 2:
        return {}

    metrics = compute_clf_metrics(preds=preds_arr, labels=labels_arr, probs=probs_arr)
    if compute_ci:
        metrics["ci"] = bootstrap_confidence_intervals(
            probs=probs_arr, labels=labels_arr, threshold=threshold, n_bootstrap=n_bootstrap
        )
    return metrics


def evaluate_predictions(
    token_probs: np.ndarray,
    token_labels: np.ndarray,
    pos_spans: List[List[int]],
    neg_spans: List[List[int]],
    threshold: float = 0.5,
    compute_ci: bool = False,
    n_bootstrap: int = 1000,
) -> Dict[str, Dict[str, any]]:
    """Return token-level **and** span-level (mean & max) metrics in one call.

    Returns a dict with three keys: ``"token"``, ``"span_mean"``,
    ``"span_max"``, each mapping to a metrics dict from
    :func:`compute_clf_metrics` / :func:`compute_span_level_metrics`.
    """
    valid_mask = token_labels != -100.0
    results: Dict[str, Dict[str, any]] = {}

    if valid_mask.any():
        valid_probs = token_probs[valid_mask]
        valid_labels = token_labels[valid_mask]
        valid_preds = (valid_probs >= threshold).astype(float)
        if len(np.unique(valid_labels)) >= 2:
            token_metrics = compute_clf_metrics(
                preds=valid_preds, labels=valid_labels, probs=valid_probs
            )
            if compute_ci:
                token_metrics["ci"] = bootstrap_confidence_intervals(
                    probs=valid_probs, labels=valid_labels, threshold=threshold, n_bootstrap=n_bootstrap
                )
            results["token"] = token_metrics

    for agg in ("max", "mean"):
        span_metrics = compute_span_level_metrics(
            token_probs=token_probs,
            pos_spans=pos_spans,
            neg_spans=neg_spans,
            threshold=threshold,
            aggregation=agg,
            compute_ci=compute_ci,
            n_bootstrap=n_bootstrap,
        )
        if span_metrics:
            results[f"span_{agg}"] = span_metrics

    return results

def plot_roc_curves(
    all_preds: Dict[str, List[float]],
    all_labels: Dict[str, List[float]], 
    all_probs: Dict[str, List[float]],
    save_dir: str,
    prefix: Optional[str] = None
) -> None:
     
    #Plot ROC curves for different aggregation levels.
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(18, 6))
    
    fpr_targets = [0.05, 0.1, 0.2, 0.5]
    dot_color = "black"
    dot_size = 40

    for i, agg_level in enumerate(['all', 'span', 'span_max']):
        plt.subplot(1, 3, i+1)
        
        if agg_level not in all_labels or len(all_labels[agg_level]) == 0:
            plt.title(f"{agg_level.replace('_', ' ').title()}\nInsufficient data")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            continue
        
        labels = np.array(all_labels[agg_level])
        probs = np.array(all_probs[agg_level])

        if len(np.unique(labels)) < 2:
            plt.title(f"{agg_level.replace('_', ' ').title()}\nInsufficient label diversity")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            continue
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = roc_auc_score(labels, probs)

        plt.fill_between(fpr, tpr, color="#f9c97d", alpha=0.5)
        plt.plot(fpr, tpr, lw=2, color="black", label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'w--', lw=2, alpha=0.7)

        # Mark TPR at specific FPRs
        for fpr_target in fpr_targets:
            idx = np.argmin(np.abs(fpr - fpr_target))
            plt.scatter(fpr[idx], tpr[idx], s=dot_size, color=dot_color, zorder=5)
            plt.text(fpr[idx], tpr[idx]+0.03, f"{tpr[idx]:.4f}", fontsize=10, ha="center", color=dot_color)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f"{agg_level.replace('_', ' ').title()}")
        plt.legend(loc="lower right")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    filename = f"{prefix}_roc_curves.png" if prefix else "roc_curves.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curves saved to {os.path.join(save_dir, filename)}")

def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, save_path: str) -> None:
    "Plot a single ROC curve."
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_threshold_analysis(
    probabilities: np.ndarray,
    labels: np.ndarray, 
    save_path: str
) -> None:
    #Plot metrics vs threshold.
    thresholds = np.linspace(0, 1, 100)
    accuracies = []
    precisions = []
    recalls = []
    for threshold in thresholds:
        preds = (probabilities >= threshold).astype(int)
        accuracies.append(accuracy_score(labels, preds))
        precisions.append(precision_score(labels, preds, zero_division=0))
        recalls.append(recall_score(labels, preds, zero_division=0))

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, label='Accuracy')
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title('Metrics vs Classification Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def print_eval_metrics(
    metrics: dict,
    metric_key_prefix: str = "",
    all_labels: Optional[dict] = None,
    include_random_baseline: bool = True,
    seed: int = 42,
) -> None:
    """Pretty-print evaluation metrics produced by compute_clf_metrics or evaluate_predictions."""
    header = f"Evaluation Metrics ({metric_key_prefix})" if metric_key_prefix else "Evaluation Metrics"
    print(f"\n===== {header} =====")

    prefix = metric_key_prefix + "/" if metric_key_prefix else ""

    # Loss metrics
    if f"{prefix}lm_loss" in metrics:
        print("\nLoss Metrics:")
        print(f"  - LM Loss:    {metrics.get(f'{prefix}lm_loss', 0):.4f}")
        print(f"  - Probe Loss: {metrics.get(f'{prefix}probe_loss', 0):.4f}")

    # Support both flat dicts (legacy) and nested dicts from evaluate_predictions
    def _print_level(level_name: str, m: dict) -> None:
        print(f"\n  [{level_name}]")
        ci_dict = m.get("ci", {})
        for key in ("accuracy", "precision", "recall", "f1", "auc",
                    "recall_at_0.1_fpr", "recall_at_0.6_fpr",
                    "threshold_optimized_accuracy", "optimal_threshold"):
            if key in m:
                ci_str = ""
                if isinstance(ci_dict, dict) and key in ci_dict:
                    lo, hi = ci_dict[key]
                    if not (np.isnan(lo) or np.isnan(hi)):
                        ci_str = f"  (95% CI: [{lo:.4f}, {hi:.4f}])"
                print(f"    - {key:<30s}: {m[key]:.4f}{ci_str}")
        counts = {k: m[k] for k in ("total_samples", "true_positive_count",
                                     "true_negative_count") if k in m}
        if counts:
            print(f"    - samples: {counts.get('total_samples', '?')}  "
                  f"(pos={counts.get('true_positive_count', '?')}, "
                  f"neg={counts.get('true_negative_count', '?')})")

    # Nested format: {"token": {...}, "span_max": {...}, "span_mean": {...}}
    nested_keys = {"token", "span_max", "span_mean"}
    if any(k in metrics for k in nested_keys):
        for level in ("token", "span_mean", "span_max"):
            if level in metrics:
                _print_level(level, metrics[level])
    else:
        # Flat format: backward-compatible with existing callers
        for agg_level in ["all", "span", "span_max"]:
            if f"{prefix}{agg_level}_accuracy" in metrics:
                _print_level(agg_level.replace("_", " ").title(), {
                    k.replace(f"{prefix}{agg_level}_", ""): v
                    for k, v in metrics.items()
                    if k.startswith(f"{prefix}{agg_level}_")
                })
        # Plain flat (e.g. direct compute_clf_metrics output)
        if "accuracy" in metrics and not any(
            f"{prefix}{a}_accuracy" in metrics for a in ["all", "span", "span_max"]
        ):
            _print_level("token", metrics)

    print("\n" + "=" * 40 + "\n")