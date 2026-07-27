"""Image -> VAREN mesh optimization: silhouette + reprojection + pose/shape prior.

Works in a bbox-centered square crop (standard top-down-pose-style practice,
same idea ViTPose itself uses to crop before inference) rather than the full
image, so the renderer/camera only ever deal with one square working
resolution regardless of the input photo's aspect ratio.
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.append("/home/om/mpi/data-gen")
sys.path.append("/home/om/mpi/data-gen/genzoo")
from synth3d2img.mesh import VarenMeshGenerator, axis_angle_to_rotmat  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fit.camera import WeakPerspectiveCamera  # noqa: E402
from fit.keypoint_map import AP10K_TO_VAREN, get_keypoint_positions  # noqa: E402
from fit.losses import build_prior, reprojection_loss, silhouette_iou, silhouette_loss  # noqa: E402
from fit.render import DifferentiableSilhouetteRenderer  # noqa: E402

sys.path.append("/home/om/mpi/VAREN")
from varen.vertex_ids import vertex_ids as VERTEX_IDS  # noqa: E402

VERTEX_IDS_VAREN = VERTEX_IDS["varen"]


def crop_and_resize(image: np.ndarray, bbox_xywh: list[float], mask: np.ndarray, padding: float, out_size: int):
    """bbox_xywh: COCO [x, y, w, h]. mask: same HxW as image, {0,1}. Returns
    (cropped_image, cropped_mask, crop_origin_xy, crop_size_px, scale_to_out)."""
    x, y, w, h = bbox_xywh
    cx, cy = x + w / 2, y + h / 2
    crop_size = max(w, h) * padding
    x0, y0 = cx - crop_size / 2, cy - crop_size / 2

    H, W = image.shape[:2]
    # Pad the source image so the crop is always fully in-bounds (edge-of-photo horses).
    pad = int(np.ceil(crop_size))
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    padded_mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    x0p, y0p = int(round(x0 + pad)), int(round(y0 + pad))
    size_i = int(round(crop_size))
    crop_img = padded[y0p : y0p + size_i, x0p : x0p + size_i]
    crop_mask = padded_mask[y0p : y0p + size_i, x0p : x0p + size_i]

    crop_img = cv2.resize(crop_img, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    crop_mask = cv2.resize(crop_mask, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    scale_to_out = out_size / crop_size
    return crop_img, crop_mask, (x0, y0), crop_size, scale_to_out


def rasterize_mask(height: int, width: int, segmentation: list[list[float]]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in segmentation:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


class FitResult:
    def __init__(self, betas, pose, global_orient, camera_scale, camera_translation, losses, iou):
        self.betas = betas
        self.pose = pose
        self.global_orient = global_orient
        self.camera_scale = camera_scale
        self.camera_translation = camera_translation
        self.losses = losses
        self.iou = iou


def fit_single_instance(
    image_path: str,
    bbox_xywh: list[float],
    segmentation: list[list[float]],
    keypoints_xys: np.ndarray,  # (17, 3): x, y, score, AP10K/APT36K order
    reference_params_path: str = "reference_params/standing_idle.json",
    device: str = "cuda",
    image_size: int = 256,
    padding: float = 1.3,
    num_iters: int = 300,
    lr: float = 0.02,
    w_kp: float = 1.0,
    w_sil: float = 5.0,
    w_prior: float = 0.2,
    log_every: int = 50,
) -> tuple[FitResult, dict]:
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]
    gt_mask = rasterize_mask(H, W, segmentation)

    crop_img, crop_mask, (x0, y0), crop_size, scale_to_out = crop_and_resize(
        image, bbox_xywh, gt_mask, padding, image_size
    )
    target_mask = torch.as_tensor(crop_mask, dtype=torch.float32, device=device).unsqueeze(0)

    kp_xy = keypoints_xys[:, :2].copy()
    kp_conf = keypoints_xys[:, 2].copy()
    kp_xy = (kp_xy - np.array([x0, y0])) * scale_to_out
    target_kp = torch.as_tensor(kp_xy, dtype=torch.float32, device=device).unsqueeze(0)
    target_conf = torch.as_tensor(kp_conf, dtype=torch.float32, device=device).unsqueeze(0)
    keypoint_names = list(AP10K_TO_VAREN.keys())

    with open(reference_params_path) as f:
        ref = json.load(f)
    mesh_gen = VarenMeshGenerator("/home/om/mpi/VAREN/models", device=device)

    betas = torch.tensor([ref["betas"]], dtype=torch.float32, device=device, requires_grad=True)
    pose = torch.tensor([ref["pose"]], dtype=torch.float32, device=device, requires_grad=True)

    # Camera init: scale the mesh (in its own rest pose) so its projected extent roughly
    # fills 1/padding of the crop (the crop was built with that same padding around the
    # GT bbox) -- a reasonable starting point rather than an arbitrary/zero one.
    with torch.no_grad():
        init_pose_aa = pose.reshape(1, 37, 3)
        init_out = mesh_gen.model(
            body_pose=axis_angle_to_rotmat(init_pose_aa),
            global_orient=axis_angle_to_rotmat(torch.zeros(1, 1, 3, device=device)),
            betas=betas,
        )
        v = init_out.vertices[0].cpu().numpy()
        extent = (v.max(0) - v.min(0)).max()
        mesh_center_xy = (v.min(0) + v.max(0))[:2] / 2  # NOT (0, 0) -- see below
    init_scale = (2.0 / padding) / extent
    # The reference mesh's own (x, y) center is nowhere near the origin (verified:
    # ~(0.51, 0.44), not (0, 0)) -- translation starting at zero, uncorrected, was the real
    # bug behind an earlier failed fit: Adam "fixed" the resulting large keypoint error not
    # by moving translation (a weak, 2-parameter gradient) but by contorting the 111-dim
    # pose to drag individual joints into place instead -- keypoint loss went down while
    # the mesh's silhouette never moved into the right region at all (final IoU 0.27,
    # silhouette loss stuck flat). Centering the initial translation removes the incentive
    # for that shortcut.
    init_translation = -init_scale * mesh_center_xy

    # Global *orientation* has the same problem one level up: gradient descent on an
    # axis-angle from a single arbitrary starting facing direction is prone to getting
    # stuck (rotating "the wrong way round" is a poor local minimum, not something a few
    # Adam steps reliably escape). Standard fix (SMPLify-style multi-init): try a handful
    # of candidate starting yaws (rotation about VAREN's up axis, Y -- see genzoo/varen_layer
    # for why Y is up in this convention), run a very short keypoint-only optimization from
    # each, and keep whichever converges to the lowest keypoint loss before committing to
    # the full staged fit below.
    candidate_yaws = [0.0, np.pi / 2, np.pi, -np.pi / 2]
    best = None
    for yaw in candidate_yaws:
        cand_orient = torch.tensor([[0.0, yaw, 0.0]], device=device, requires_grad=True)
        cand_camera = WeakPerspectiveCamera(batch_size=1, init_scale=float(init_scale)).to(device)
        with torch.no_grad():
            cand_camera.translation.data = torch.tensor(
                [init_translation], dtype=torch.float32, device=device
            )
        opt = torch.optim.Adam([cand_orient, cand_camera.log_scale, cand_camera.translation], lr=lr)
        for _ in range(30):
            opt.zero_grad()
            orient_aa = cand_orient.reshape(1, 1, 3)
            out = mesh_gen.model(
                body_pose=axis_angle_to_rotmat(init_pose_aa),
                global_orient=axis_angle_to_rotmat(orient_aa),
                betas=betas,
            )
            kp_3d = get_keypoint_positions(out.vertices, out.joints, VERTEX_IDS_VAREN, keypoint_names)
            kp_2d = cand_camera.project_to_pixels(kp_3d, image_size)
            l_kp = reprojection_loss(kp_2d, target_kp, target_conf).mean()
            l_kp.backward()
            opt.step()
        final_loss = l_kp.item()
        print(f"  yaw={np.degrees(yaw):+6.1f}deg -> keypoint loss after 30 steps: {final_loss:.2f}")
        if best is None or final_loss < best[0]:
            best = (final_loss, cand_orient.detach().clone(), cand_camera.log_scale.detach().clone(),
                    cand_camera.translation.detach().clone())

    _, best_orient, best_log_scale, best_translation = best
    global_orient = best_orient.clone().requires_grad_(True)
    camera = WeakPerspectiveCamera(batch_size=1, init_scale=float(init_scale)).to(device)
    with torch.no_grad():
        camera.log_scale.data = best_log_scale
        camera.translation.data = best_translation

    renderer = DifferentiableSilhouetteRenderer(image_size=image_size, device=device)
    prior = build_prior().to(device)
    faces = torch.as_tensor(mesh_gen.faces.astype("int64"), device=device)

    def forward_and_losses(compute_sil: bool):
        pose_aa = pose.reshape(1, 37, 3)
        orient_aa = global_orient.reshape(1, 1, 3)
        out = mesh_gen.model(
            body_pose=axis_angle_to_rotmat(pose_aa),
            global_orient=axis_angle_to_rotmat(orient_aa),
            betas=betas,
        )
        vertices, joints = out.vertices, out.joints

        kp_3d = get_keypoint_positions(vertices, joints, VERTEX_IDS_VAREN, keypoint_names)
        kp_2d = camera.project_to_pixels(kp_3d, image_size)
        l_kp = reprojection_loss(kp_2d, target_kp, target_conf).mean()

        if compute_sil:
            sil = renderer.render(vertices, faces, camera)
            l_sil = silhouette_loss(sil, target_mask).mean()
        else:
            sil, l_sil = None, torch.zeros((), device=device)

        fullpose_114 = torch.cat([global_orient, pose], dim=-1)
        l_prior = prior(betas, fullpose_114).mean()
        return vertices, joints, sil, l_kp, l_sil, l_prior

    # Staged optimization (per AniMerPlus's own Stage 4 spec -- this project's fitting
    # target): a one-shot joint optimization over everything at once was tried first and
    # failed exactly the way that spec predicts an unstaged fit would -- keypoint loss
    # (initially ~2000, dwarfing the [0,1]-bounded silhouette term) drove the optimizer to
    # contort the pose until individual joints matched target 2D points, while the whole
    # mesh's silhouette never moved into place (final IoU 0.27, silhouette loss stuck at
    # ~0.75, prior loss ballooning from -323 to +1092 -- i.e. converging to something
    # keypoint-accurate but anatomically absurd). Splitting into stages that each solve a
    # narrower problem first (where/how is the whole body placed and oriented -> then let
    # it pose -> then let it change shape) fixes this the same way it does in SMPLify-style
    # fitting generally.
    loss_history = []
    it_counter = 0

    def run_stage(stage_name: str, trainable_params, n_iters: int, w_kp_stage, w_sil_stage, w_prior_stage, use_sil: bool):
        nonlocal it_counter
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        for _ in range(n_iters):
            optimizer.zero_grad()
            vertices, joints, sil, l_kp, l_sil, l_prior = forward_and_losses(compute_sil=use_sil)
            loss = w_kp_stage * l_kp + w_sil_stage * l_sil + w_prior_stage * l_prior
            loss.backward()
            optimizer.step()
            loss_history.append(
                {
                    "iter": it_counter, "stage": stage_name, "total": loss.item(),
                    "kp": l_kp.item(), "sil": l_sil.item() if use_sil else None, "prior": l_prior.item(),
                }
            )
            if it_counter % log_every == 0 or _ == n_iters - 1:
                sil_str = f"{l_sil.item():.4f}" if use_sil else "  n/a "
                print(f"[{stage_name:6s} {it_counter:4d}] total={loss.item():.4f} kp={l_kp.item():.4f} sil={sil_str} prior={l_prior.item():.4f}")
            it_counter += 1
        return vertices, joints, sil

    # Stage 1 gets the largest share deliberately: the multi-yaw search above only tries
    # discrete rotations about one axis, but the residual rotation needed to match a real
    # photo is a general 3D one -- verified (standalone gradient check) that keypoint loss
    # was *still* trending down at 50/50 iterations, i.e. not converged. Handing off to
    # stage 2/3 before this actually settles lets pose/shape freedom "cheat" an incomplete
    # placement instead.
    n1 = max(1, num_iters // 2)
    n2 = max(1, num_iters // 3)
    n3 = num_iters - n1 - n2

    # Stage 1: global placement only (camera + global_orient), pose/betas frozen at the
    # reference. Keypoint loss only -- silhouette gradients are unreliable before the mesh
    # is even roughly in the right place (soft-rasterization has limited "reach").
    run_stage("place", [camera.log_scale, camera.translation, global_orient], n1,
              w_kp_stage=1.0, w_sil_stage=0.0, w_prior_stage=0.0, use_sil=False)

    # Stage 2: let pose deform to match keypoints + stay plausible, betas still frozen.
    run_stage("pose", [camera.log_scale, camera.translation, global_orient, pose], n2,
              w_kp_stage=w_kp, w_sil_stage=0.0, w_prior_stage=w_prior, use_sil=False)

    # Stage 3: full loss, everything free -- silhouette now has a sensible starting point.
    vertices, joints, sil = run_stage(
        "full", [camera.log_scale, camera.translation, global_orient, pose, betas], n3,
        w_kp_stage=w_kp, w_sil_stage=w_sil, w_prior_stage=w_prior, use_sil=True,
    )
    if sil is None:  # n3 could be 0 for a very small num_iters
        vertices, joints, sil, _, _, _ = forward_and_losses(compute_sil=True)

    with torch.no_grad():
        sil_final = renderer.render(vertices, faces, camera)
        iou = silhouette_iou(sil_final, target_mask).item()

    result = FitResult(
        betas=betas.detach().cpu().numpy(),
        pose=pose.detach().cpu().numpy(),
        global_orient=global_orient.detach().cpu().numpy(),
        camera_scale=camera.scale.detach().cpu().numpy(),
        camera_translation=camera.translation.detach().cpu().numpy(),
        losses=loss_history,
        iou=iou,
    )
    debug = {
        "crop_img": crop_img,
        "crop_mask": crop_mask,
        "target_kp": kp_xy,
        "target_conf": kp_conf,
        "keypoint_names": keypoint_names,
        "final_vertices": vertices.detach().cpu().numpy(),
        "final_silhouette": sil_final[0].detach().cpu().numpy(),
        "faces": mesh_gen.faces,
        "image_size": image_size,
    }
    return result, debug
