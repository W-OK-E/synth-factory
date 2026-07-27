"""Correspondence between AP10K/APT36K's 17-keypoint scheme (what
ViTPose+/vitpose_ap10k and vitpose_apt36k predict -- confirmed identical
keypoint_names for both, see Horse-image-data/*/keypoints_vitpose_*.json)
and VAREN's joints/surface vertices.

Two kinds of VAREN point are used:
  - ("joint", i): one of the 38 kinematic-tree joints varen_layer.VARENLayer's
    `output.joints[:, i]` returns (indices from VAREN.pkl's `parts` dict,
    which names all 38 -- Pelvis=0 ... REar=37; `parts["Face"]=38` is a
    39th, *segmentation-only* label with no kinematic-tree column in
    `kintree_table`, not a real joint, and isn't used here).
  - ("vertex", name): a single named mesh vertex from VAREN.pkl's
    `vertex_ids['varen']` dict, taken directly from `output.vertices`.

Joints are the natural choice for rotation-pivot concepts (shoulder, elbow,
hip, knee/stifle, neck, tail root). Eyes/nose/paw-tips are surface
locations, not rotation pivots, so a named vertex is the better match.

This mapping is necessarily approximate in a couple of places: AP10K/APT36K's
17-keypoint scheme is a generic quadruped skeleton, not horse-specific, and
doesn't line up 1:1 with horse anatomy for the front leg -- what it calls
"Elbow" lands closest to VAREN's LFLeg2 joint, which is anatomically a
horse's true elbow, but lay/generic-dataset annotators (and this
correspondence) can't be assumed to distinguish that precisely from the
carpus one joint further down. Verified by rendering markers on the
standing-idle reference mesh and checking each lands in the anatomically
sensible place -- see verify_keypoint_map.py's output
(outputs/keypoint_map_check.png), not just asserted here.
"""
from __future__ import annotations

# name -> ("joint", index) | ("vertex", vertex_ids['varen'] key)
AP10K_TO_VAREN: dict[str, tuple[str, int | str]] = {
    "L_Eye": ("vertex", "leye"),
    "R_Eye": ("vertex", "reye"),
    "Nose": ("vertex", "between_nostrils_base"),
    "Neck": ("joint", 26),  # Neck
    "Root of tail": ("joint", 30),  # Tail1
    "L_Shoulder": ("joint", 6),  # LFLeg1
    "L_Elbow": ("joint", 7),  # LFLeg2
    "L_F_Paw": ("vertex", "lfront_foot_tip"),
    "R_Shoulder": ("joint", 11),  # RFLeg1
    "R_Elbow": ("joint", 12),  # RFLeg2
    "R_F_Paw": ("vertex", "rfront_foot_tip"),
    "L_Hip": ("joint", 16),  # LBLeg1
    "L_Knee": ("joint", 17),  # LBLeg2 (stifle -- the hind "knee")
    "L_B_Paw": ("vertex", "lback_foot_tip"),
    "R_Hip": ("joint", 21),  # RBLeg1
    "R_Knee": ("joint", 22),  # RBLeg2
    "R_B_Paw": ("vertex", "rback_foot_tip"),
}

VAREN_PARTS = {
    "Pelvis": 0, "Spine": 1, "Spine1": 2, "Spine2": 3, "LScapula": 4, "RScapula": 5,
    "LFLeg1": 6, "LFLeg2": 7, "LFLeg3": 8, "LFLeg4": 9, "LFToe": 10,
    "RFLeg1": 11, "RFLeg2": 12, "RFLeg3": 13, "RFLeg4": 14, "RFToe": 15,
    "LBLeg1": 16, "LBLeg2": 17, "LBLeg3": 18, "LBLeg4": 19, "LBToe": 20,
    "RBLeg1": 21, "RBLeg2": 22, "RBLeg3": 23, "RBLeg4": 24, "RBToe": 25,
    "Neck": 26, "Neck1": 27, "Neck2": 28, "Head": 29,
    "Tail1": 30, "Tail2": 31, "Tail3": 32, "Tail4": 33, "Tail5": 34,
    "Mouth": 35, "LEar": 36, "REar": 37,
}


def get_keypoint_positions(vertices, joints, vertex_ids: dict, keypoint_names: list[str]):
    """vertices: (..., V, 3), joints: (..., 38, 3) -- same batch shape.
    keypoint_names: e.g. the AP10K/APT36K 17 names, in the order predictions
    come in. Returns (..., 17, 3) VAREN points, same order."""
    import torch

    points = []
    for name in keypoint_names:
        kind, ref = AP10K_TO_VAREN[name]
        if kind == "joint":
            points.append(joints[..., ref, :])
        else:
            points.append(vertices[..., vertex_ids[ref], :])
    return torch.stack(points, dim=-2)
