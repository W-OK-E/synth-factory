"""Sample VAREN pose/shape parameters around a reference mean.

The reference mean (global_orient, pose, betas) is the one shipped with
VAREN at `VAREN/examples/example_params.json`. New samples are drawn as
isotropic Gaussian noise around that mean, per the shape/pose sampling
idea in `data-gen/.agent/3d2img.md`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from synth3d2img.mocap_poses import HSMAL_POSE_DIM, load_hsmal_poses


@dataclass
class VarenParams:
    """A batch of VAREN input parameters."""

    global_orient: np.ndarray  # (N, 3)
    pose: np.ndarray  # (N, 111)
    betas: np.ndarray  # (N, 39)
    # Per-sample provenance for the joints hSMAL mocap doesn't cover, e.g.
    # "zero" / "sampled"; empty for the plain Gaussian sampler.
    extra_joint_fill: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return self.pose.shape[0]


def load_mean_params(params_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the (global_orient, pose, betas) mean from a VAREN params json."""
    with open(params_path) as f:
        raw = json.load(f)
    global_orient = np.asarray(raw["global_orient"], dtype=np.float32)
    pose = np.asarray(raw["pose"], dtype=np.float32)
    betas = np.asarray(raw["betas"], dtype=np.float32)
    return global_orient, pose, betas


def sample_params(
    n: int,
    params_path: str,
    pose_std: float = 0.15,
    global_orient_std: float = 0.2,
    betas_std: float = 0.03,
    num_active_betas: int = 10,
    seed: int | None = None,
) -> VarenParams:
    """Sample `n` VAREN params around the mean in `params_path`.

    `num_active_betas` limits shape noise to the leading beta components,
    matching the reference params where only the first ~10 of 39 betas are
    non-zero.
    """
    global_orient_mean, pose_mean, betas_mean = load_mean_params(params_path)
    rng = np.random.default_rng(seed)

    global_orient = global_orient_mean + rng.normal(
        0.0, global_orient_std, size=(n, global_orient_mean.shape[0])
    ).astype(np.float32)

    pose = pose_mean + rng.normal(
        0.0, pose_std, size=(n, pose_mean.shape[0])
    ).astype(np.float32)

    betas = np.tile(betas_mean, (n, 1))
    num_active_betas = min(num_active_betas, betas_mean.shape[0])
    betas[:, :num_active_betas] += rng.normal(
        0.0, betas_std, size=(n, num_active_betas)
    ).astype(np.float32)

    return VarenParams(global_orient=global_orient, pose=pose, betas=betas)


def sample_params_from_mocap(
    model_data_dir: str,
    params_path: str,
    global_orient_std: float = 0.2,
    betas_std: float = 0.03,
    num_active_betas: int = 10,
    extra_joint_std: float = 0.1,
    max_samples: int | None = None,
    seed: int | None = None,
) -> VarenParams:
    """Build one VAREN sample per real hSMAL mocap frame.

    hSMAL's 108 pose values/frame (36 joints incl. root) fill VAREN's
    body_pose[:108] directly, positionally. VAREN's remaining body_pose
    (the one joint hSMAL has no data for) has, per frame, an independent
    coin flip: zeros, or Gaussian noise around the reference pose for that
    joint -- covering both filling strategies across the dataset without
    doubling its size. global_orient and betas are sampled the same way as
    `sample_params` (unchanged, per data-gen/.agent/3d2img.md).

    `max_samples`: if set and smaller than the total number of valid mocap
    frames, a random subset (without replacement) is used instead of every
    frame -- e.g. for a partial run short of the full dataset.
    """
    global_orient_mean, pose_mean, betas_mean = load_mean_params(params_path)
    hsmal_poses = load_hsmal_poses(model_data_dir)
    rng = np.random.default_rng(seed)

    if max_samples is not None and max_samples < hsmal_poses.shape[0]:
        chosen = np.sort(
            rng.choice(hsmal_poses.shape[0], size=max_samples, replace=False)
        )
        hsmal_poses = hsmal_poses[chosen]

    n = hsmal_poses.shape[0]

    global_orient = global_orient_mean + rng.normal(
        0.0, global_orient_std, size=(n, global_orient_mean.shape[0])
    ).astype(np.float32)

    extra_dim = pose_mean.shape[0] - HSMAL_POSE_DIM
    use_sampled_fill = rng.random(n) < 0.5
    extra = np.where(
        use_sampled_fill[:, None],
        pose_mean[HSMAL_POSE_DIM:]
        + rng.normal(0.0, extra_joint_std, size=(n, extra_dim)),
        0.0,
    ).astype(np.float32)
    print("Adding extra data with dimension:",extra.shape)
    pose = np.concatenate([hsmal_poses, extra], axis=1)
    extra_joint_fill = ["sampled" if s else "zero" for s in use_sampled_fill]

    betas = np.tile(betas_mean, (n, 1))
    num_active_betas = min(num_active_betas, betas_mean.shape[0])
    betas[:, :num_active_betas] += rng.normal(
        0.0, betas_std, size=(n, num_active_betas)
    ).astype(np.float32)

    return VarenParams(
        global_orient=global_orient,
        pose=pose,
        betas=betas,
        extra_joint_fill=extra_joint_fill,
    )
