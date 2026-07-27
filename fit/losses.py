"""Losses for the image -> VAREN mesh optimization: reprojection, silhouette,
and the pose/shape plausibility prior from prior-qc.
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

sys.path.append("/home/om/mpi/data-gen/prior-qc")
from prior_qc.torch_prior import TorchPlausibilityPrior  # noqa: E402


def reprojection_loss(
    projected: torch.Tensor, target: torch.Tensor, confidence: torch.Tensor, kpt_thr: float = 0.3
) -> torch.Tensor:
    """projected: (B, K, 2) pixel coords from the mesh. target: (B, K, 2) detected pixel
    coords. confidence: (B, K) detection scores -- doubles as the per-keypoint weight AND
    a soft mask (near-zero weight for low-confidence/undetected keypoints, so a missed
    detection doesn't drag the fit toward (0, 0))."""
    weight = confidence * (confidence > kpt_thr).float()
    sq_err = ((projected - target) ** 2).sum(dim=-1)  # (B, K)
    return (weight * sq_err).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1e-6)


def silhouette_loss(rendered: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """Soft IoU loss: 1 - intersection/union, both (B, H, W) in [0, 1]. Differentiable,
    scale-invariant (unlike raw MSE, doesn't just reward shrinking the silhouette)."""
    intersection = (rendered * target_mask).sum(dim=(-1, -2))
    union = (rendered + target_mask - rendered * target_mask).sum(dim=(-1, -2))
    return 1.0 - intersection / union.clamp_min(1e-6)


def silhouette_iou(rendered_binary: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """Hard IoU (for QC reporting, not optimization -- thresholded, not differentiable)."""
    rendered_binary = rendered_binary > 0.5
    target_mask = target_mask > 0.5
    intersection = (rendered_binary & target_mask).float().sum(dim=(-1, -2))
    union = (rendered_binary | target_mask).float().sum(dim=(-1, -2))
    return intersection / union.clamp_min(1e-6)


def build_prior(
    shape_prior_path: str = "/home/om/mpi/data-gen/prior-qc/outputs/shape_prior.pkl",
    pose_prior_path: str = "/home/om/mpi/data-gen/prior-qc/outputs/pose_prior.pkl",
    shape_weight: float = 1.0,
    pose_weight: float = 1.0,
) -> TorchPlausibilityPrior:
    return TorchPlausibilityPrior.from_files(
        shape_prior_path=shape_prior_path,
        pose_prior_path=pose_prior_path,
        shape_weight=shape_weight,
        pose_weight=pose_weight,
    )
