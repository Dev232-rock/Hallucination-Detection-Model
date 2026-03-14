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