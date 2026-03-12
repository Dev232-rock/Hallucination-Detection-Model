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