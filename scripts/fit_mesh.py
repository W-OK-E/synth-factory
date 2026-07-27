"""Image -> VAREN mesh fitting CLI: the pipeline described in .agent/plan.md,
steps 3-5 (mesh generation, silhouette+reprojection+prior optimization, QC).
Step 2 (markers) is scripts/run_keypoints.py, run beforehand.

    cd /home/om/mpi/data-gen/synth_2d_to_3d
    micromamba run -n animer2 python scripts/fit_mesh.py \\
        --split test --file horse-pic_jpg.rf.3a18d24d1937f82c0e34dbc70ab1365f.jpg \\
        --keypoint_model vitpose_apt36k --out_dir outputs/fits

    # or fit every image in a split:
    micromamba run -n animer2 python scripts/fit_mesh.py --split test --out_dir outputs/fits
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PIL import Image, ImageDraw

from fit.losses import build_prior
from fit.optimize import fit_single_instance
from fit.qc import qc_report

DATA_ROOT = "/home/om/mpi/data-gen/Horse-image-data"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True, choices=["train", "valid", "test"])
    p.add_argument("--file", default=None, help="Single file_name to fit; omit to fit the whole split.")
    p.add_argument("--keypoint_model", default="vitpose_apt36k", choices=["vitpose_apt36k", "vitpose_ap10k", "horse10"])
    p.add_argument("--out_dir", default="outputs/fits")
    p.add_argument("--num_iters", type=int, default=500)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def fit_one(coco, kp_by_file, split_dir, file_name, args, prior):
    img_entry = next(i for i in coco["images"] if i["file_name"] == file_name)
    anns = [a for a in coco["annotations"] if a["image_id"] == img_entry["id"]]
    if not anns:
        return None
    ann = anns[0]  # one horse per image in this dataset (max 1 GT box seen in practice)

    if file_name not in kp_by_file:
        print(f"  skip {file_name}: no {args.keypoint_model} keypoints found")
        return None
    keypoints = np.array(kp_by_file[file_name])

    result, debug = fit_single_instance(
        image_path=os.path.join(split_dir, file_name),
        bbox_xywh=ann["bbox"],
        segmentation=ann["segmentation"],
        keypoints_xys=keypoints,
        device=args.device,
        image_size=args.image_size,
        num_iters=args.num_iters,
        log_every=args.num_iters,  # only start/end
    )

    from fit.camera import WeakPerspectiveCamera
    from fit.keypoint_map import AP10K_TO_VAREN, get_keypoint_positions
    import torch

    sys.path.append("/home/om/mpi/data-gen")
    sys.path.append("/home/om/mpi/data-gen/genzoo")
    from synth3d2img.mesh import VarenMeshGenerator, axis_angle_to_rotmat

    sys.path.append("/home/om/mpi/VAREN")
    from varen.vertex_ids import vertex_ids as VERTEX_IDS

    mesh_gen = VarenMeshGenerator("/home/om/mpi/VAREN/models", device=args.device)
    pose_t = torch.tensor(result.pose, dtype=torch.float32, device=args.device)
    orient_t = torch.tensor(result.global_orient, dtype=torch.float32, device=args.device)
    betas_t = torch.tensor(result.betas, dtype=torch.float32, device=args.device)
    out = mesh_gen.model(
        body_pose=axis_angle_to_rotmat(pose_t.reshape(1, 37, 3)),
        global_orient=axis_angle_to_rotmat(orient_t.reshape(1, 1, 3)),
        betas=betas_t,
    )
    keypoint_names = list(AP10K_TO_VAREN.keys())
    kp_3d = get_keypoint_positions(out.vertices, out.joints, VERTEX_IDS["varen"], keypoint_names)
    cam = WeakPerspectiveCamera(batch_size=1).to(args.device)
    with torch.no_grad():
        cam.log_scale.data = torch.log(torch.tensor(result.camera_scale, device=args.device))
        cam.translation.data = torch.tensor(result.camera_translation, device=args.device)
    kp_2d = cam.project_to_pixels(kp_3d, args.image_size)[0].detach().cpu().numpy()

    report = qc_report(
        final_vertices=debug["final_vertices"],
        final_silhouette=debug["final_silhouette"],
        target_mask=debug["crop_mask"],
        projected_keypoints=kp_2d,
        target_keypoints=debug["target_kp"],
        target_confidence=debug["target_conf"],
        betas=result.betas,
        global_orient=result.global_orient,
        pose=result.pose,
        prior=prior,
    )
    report["file_name"] = file_name

    stem = os.path.splitext(file_name)[0]
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, f"{stem}_params.npz"),
        betas=result.betas, pose=result.pose, global_orient=result.global_orient,
        camera_scale=result.camera_scale, camera_translation=result.camera_translation,
    )

    crop_img = debug["crop_img"]
    sil = debug["final_silhouette"]
    red = np.zeros_like(crop_img); red[..., 0] = 255
    alpha = (sil[..., None] * 0.5)
    overlay = (crop_img * (1 - alpha) + red * alpha).astype(np.uint8)
    img = Image.fromarray(overlay)
    draw = ImageDraw.Draw(img)
    for (x, y), c in zip(kp_2d, debug["target_conf"]):
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(0, 255, 0))
    for (x, y), c in zip(debug["target_kp"], debug["target_conf"]):
        if c > 0.3:
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 255, 0))
    img.save(os.path.join(out_dir, f"{stem}_overlay.png"))

    return report


def main() -> None:
    args = parse_args()
    split_dir = os.path.join(DATA_ROOT, args.split)
    coco = json.load(open(os.path.join(split_dir, "_annotations.coco.json")))
    kp_data = json.load(open(os.path.join(split_dir, f"keypoints_{args.keypoint_model}.json")))
    kp_by_file = {r["file_name"]: r["keypoints"] for r in kp_data["results"]}

    prior = build_prior().to(args.device)

    files = [args.file] if args.file else [im["file_name"] for im in coco["images"]]
    os.makedirs(args.out_dir, exist_ok=True)
    reports = []
    for i, file_name in enumerate(files):
        print(f"[{i + 1}/{len(files)}] {file_name}")
        try:
            report = fit_one(coco, kp_by_file, split_dir, file_name, args, prior)
        except Exception as e:
            print(f"  FAILED: {e}")
            report = {"file_name": file_name, "error": str(e)}
        if report is not None:
            reports.append(report)
            print(f"  IoU={report.get('silhouette_iou', 'n/a')} "
                  f"kp_err={report.get('mean_keypoint_error_px', 'n/a')} "
                  f"prior={report.get('prior_neg_log_prob', 'n/a')}")

    with open(os.path.join(args.out_dir, "qc_report.json"), "w") as f:
        json.dump(reports, f, indent=2)
    ious = [r["silhouette_iou"] for r in reports if "silhouette_iou" in r]
    if ious:
        print(f"\n{len(ious)} fits, mean IoU={np.mean(ious):.3f}, median={np.median(ious):.3f}")
    print(f"Wrote {args.out_dir}/qc_report.json")


if __name__ == "__main__":
    main()
