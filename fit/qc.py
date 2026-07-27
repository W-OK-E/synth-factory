"""Section 5 of .agent/plan.md: quality control for a fitted mesh.

Three independent signals, deliberately not blended into one score (same
reasoning as prior_qc.scoring.PlausibilityScorer: different scales, and
collapsing them hides which one is driving a bad fit):
  - silhouette_iou: does the mesh's outline explain the image's silhouette?
  - mean_keypoint_error_px: does it explain the detected 2D keypoints?
  - prior_neg_log_prob: is the result anatomically plausible on its own
    terms, independent of whether it happens to also fit this image well
    (catches the "keypoint-accurate but anatomically absurd" failure mode
    this pipeline hit and fixed during development -- see optimize.py).
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.append("/home/om/mpi/data-gen/prior-qc")
from prior_qc.torch_prior import TorchPlausibilityPrior  # noqa: E402


def qc_report(
    final_vertices: np.ndarray,
    final_silhouette: np.ndarray,
    target_mask: np.ndarray,
    projected_keypoints: np.ndarray,
    target_keypoints: np.ndarray,
    target_confidence: np.ndarray,
    betas: np.ndarray,
    global_orient: np.ndarray,
    pose: np.ndarray,
    prior: TorchPlausibilityPrior,
    kpt_thr: float = 0.3,
) -> dict:
    sil_binary = final_silhouette > 0.5
    mask_binary = target_mask > 0.5
    intersection = (sil_binary & mask_binary).sum()
    union = (sil_binary | mask_binary).sum()
    iou = float(intersection / max(union, 1))

    visible = target_confidence > kpt_thr
    if visible.any():
        err = np.linalg.norm(projected_keypoints[visible] - target_keypoints[visible], axis=-1)
        mean_kp_err = float(err.mean())
        max_kp_err = float(err.max())
    else:
        mean_kp_err = max_kp_err = float("nan")

    with torch.no_grad():
        prior_device = prior.shape.mean.device  # TorchPlausibilityPrior's stats are buffers, not
        # nn.Parameters (they're fixed, not learned) -- next(prior.parameters()) is empty
        fullpose = torch.cat(
            [
                torch.as_tensor(global_orient, dtype=torch.float32, device=prior_device),
                torch.as_tensor(pose, dtype=torch.float32, device=prior_device),
            ],
            dim=-1,
        )
        betas_t = torch.as_tensor(betas, dtype=torch.float32, device=prior_device)
        neg_log_prob = prior(betas_t, fullpose).item()

    return {
        "silhouette_iou": iou,
        "mean_keypoint_error_px": mean_kp_err,
        "max_keypoint_error_px": max_kp_err,
        "n_visible_keypoints": int(visible.sum()),
        "prior_neg_log_prob": neg_log_prob,
    }
