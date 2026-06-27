from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation


def _bin(edge: np.ndarray) -> np.ndarray:
    return edge > 0


def edge_metrics_tolerance(pred: np.ndarray, gt: np.ndarray, tolerance: int = 2) -> dict[str, float]:
    p = _bin(pred)
    g = _bin(gt)
    if tolerance > 0:
        structure = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=bool)
        g_d = binary_dilation(g, structure=structure)
        p_d = binary_dilation(p, structure=structure)
    else:
        g_d = g
        p_d = p

    tp_pred = np.logical_and(p, g_d).sum()
    pred_n = p.sum()
    precision = tp_pred / pred_n if pred_n else 0.0

    tp_gt = np.logical_and(g, p_d).sum()
    gt_n = g.sum()
    recall = tp_gt / gt_n if gt_n else 0.0

    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    iou = inter / union if union else 0.0

    return {
        "precision_at_k": float(precision),
        "recall_at_k": float(recall),
        "f1_at_k": float(f1),
        "iou_strict": float(iou),
        "pred_edge_pixels": int(pred_n),
        "gt_edge_pixels": int(gt_n),
    }
