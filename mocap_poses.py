"""Load real horse mocap poses (hSMAL) to drive VAREN pose sampling.

Per data-gen/.agent/3d2img.md: hSMAL poses (36 joints incl. root, 108
values/frame) are used as VAREN poses (37 joints, 111 values/frame) instead
of Gaussian noise. hSMAL's 108 values fill VAREN's body_pose[:108]
positionally; VAREN's remaining body_pose[108:111] (the one extra joint
hSMAL doesn't have) has no mocap source, so each frame independently gets
either zeros or noise sampled around the reference pose for that joint.
"""
from __future__ import annotations

import glob
import os
import pickle

import numpy as np

HSMAL_POSE_DIM = 108  # 36 joints (incl. root) x 3, per data-gen/.agent/3d2img.md


def load_hsmal_poses(model_data_dir: str) -> np.ndarray:
    """Concatenate 'poses' from every *.npz or *.pkl in `model_data_dir`, dropping frames
    listed in each file's 'missing_frame' (placeholder all-zero rows)."""
    valid_chunks = []
    
    # collect both .npz and .pkl files
    paths = sorted(glob.glob(os.path.join(model_data_dir, "*.npz")))
    paths += sorted(glob.glob(os.path.join(model_data_dir, "*.pkl")))

    for path in paths:
        if "horse" in path: #Skipping horse_stagei.pkl which does not contain poses
            continue
        if path.lower().endswith(".npz"):
            data = np.load(path)
        else:
            with open(path, "rb") as f:
                data = pickle.load(f)

        # data may be a dict-like with key 'fullpose', or the array itself
        if  "fullpose" in data:
            poses = np.array(data["fullpose"], dtype=np.float32)
        else:
            print("Loaded", path,"data type is:",type(data))
            poses = np.array(data, dtype=np.float32)

        # ensure shape is (frames, dims)
        if poses.ndim == 1:
            poses = poses[np.newaxis, ...]

        # If an extra 3 global position coordinates exist, drop them (trim to HSMAL_POSE_DIM)
        if poses.shape[1] > HSMAL_POSE_DIM:
            poses = poses[:, :HSMAL_POSE_DIM]

        valid_chunks.append(poses)

    if not valid_chunks:
        raise FileNotFoundError(f"No .npz pose files found in {model_data_dir}")

    poses = np.concatenate(valid_chunks, axis=0)
    if poses.shape[1] != HSMAL_POSE_DIM:
        raise ValueError(
            f"Expected hSMAL poses with {HSMAL_POSE_DIM} values/frame, "
            f"got {poses.shape[1]}"
        )
    return poses
