"""Generate a synthetic VAREN (params, mesh renders) dataset.

For each sample: get betas/pose/global_orient, run them through VAREN to
get a mesh, and render a silhouette, mask, and depth map. Downstream
texturing/background synthesis (e.g. data-gen/utils/flux2img.py) is out of
scope here.

Two pose sources (see data-gen/.agent/3d2img.md):
  - Gaussian (default): noise around VAREN's reference example_params.json.
  - `--mocap-dir`: one sample per real hSMAL mocap frame (count is then
    however many valid frames exist, --num-samples is ignored).

Run inside the `animer2` micromamba env, which has varen/pyrender/amr
installed:

    micromamba run -n animer2 python -m synth3d2img.generate \
        --output-dir /path/to/dataset --num-samples 10000

    micromamba run -n animer2 python -m synth3d2img.generate \
        --output-dir /path/to/dataset \
        --mocap-dir /home/om/mpi/PFERD/hSMALdata/MODEL_DATA
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import sys
sys.path.append('/home/om/mpi/data-gen/')

from synth3d2img.mesh import VarenMeshGenerator
from synth3d2img.render import MeshRenderer
from synth3d2img.sampling import sample_params, sample_params_from_mocap

DEFAULT_MODEL_PATH = "/home/om/mpi/VAREN/models"
DEFAULT_PARAMS_PATH = "/home/om/mpi/VAREN/examples/example_params.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output_dir", required=True, help="Dataset output folder.")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Gaussian mode: sample count (default 10 if unset). Mocap mode: "
        "if unset, one sample per valid mocap frame (the full run); if set "
        "and smaller than the total frame count, a random subset of this "
        "size is used instead.",
    )
    parser.add_argument(
        "--mocap_dir",
        default="/home/om/mpi/PFERD/hSMALdata/VAREN-poses",
        help="Directory of hSMAL *.npz pose files; if set, one sample per "
        "valid mocap frame instead of Gaussian pose sampling (--num-samples "
        "is ignored).",
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--params", default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--batch_size", type=int, default=32, help="VAREN forward-pass batch size.")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--pose_std", type=float, default=0.30)
    parser.add_argument("--global_orient_std", type=float, default=0.2)
    parser.add_argument("--betas_std", type=float, default=0.03)
    parser.add_argument("--num_active_betas", type=int, default=10)
    parser.add_argument(
        "--extra-joint-std",
        type=float,
        default=0.1,
        help="[--mocap-dir only] std for the one VAREN joint hSMAL has no mocap for.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min_elevation_deg",
        type=float,
        default=-25.0,
        help="Camera orbit elevation lower bound (0 = horse-eye level).",
    )
    parser.add_argument(
        "--max_elevation_deg",
        type=float,
        default=25.0,
        help="Camera orbit elevation upper bound. Kept well away from "
        "+-90 so renders never come from directly overhead/underneath.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--no-muscle-deformations",
        action="store_true",
        help="Disable VAREN's neural muscle deformer (faster, less realistic).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    if args.mocap_dir:
        params = sample_params_from_mocap(
            model_data_dir=args.mocap_dir,
            params_path=args.params,
            global_orient_std=args.global_orient_std,
            betas_std=args.betas_std,
            num_active_betas=args.num_active_betas,
            extra_joint_std=args.extra_joint_std,
            max_samples=args.num_samples,
            seed=args.seed,
        )
    else:
        params = sample_params(
            n=args.num_samples if args.num_samples is not None else 10,
            params_path=args.params,
            pose_std=args.pose_std,
            global_orient_std=args.global_orient_std,
            betas_std=args.betas_std,
            num_active_betas=args.num_active_betas,
            seed=args.seed,
        )

    mesh_gen = VarenMeshGenerator(
        args.model_path,
        use_muscle_deformations=not args.no_muscle_deformations,
        device=args.device,
    )
    renderer = MeshRenderer(mesh_gen.faces, image_size=args.image_size)

    # Camera orbits freely in azimuth around the horse; elevation is kept
    # within a band around the horizon so renders never come from directly
    # overhead or underneath.
    camera_rng = np.random.default_rng(args.seed)
    azimuths = camera_rng.uniform(0.0, 2 * np.pi, size=len(params))
    elevations = camera_rng.uniform(
        np.radians(args.min_elevation_deg), np.radians(args.max_elevation_deg),
        size=len(params),
    )

    n_digits = max(8, len(str(len(params))))
    manifest_path = os.path.join(args.output_dir, "manifest.jsonl")

    with open(manifest_path, "w") as manifest:
        for start in range(0, len(params), args.batch_size):
            end = min(start + args.batch_size, len(params))
            vertices_batch = mesh_gen.generate(
                params.global_orient[start:end],
                params.pose[start:end],
                params.betas[start:end],
            )

            for offset, sample_idx in enumerate(range(start, end)):
                render = renderer.render(
                    vertices_batch[offset],
                    azimuth=azimuths[sample_idx],
                    elevation=elevations[sample_idx],
                )
                sample_id = f"{sample_idx:0{n_digits}d}"

                sil_rel = f"images/{sample_id}_silhouette.png"
                mask_rel = f"images/{sample_id}_mask.png"
                depth_rel = f"images/{sample_id}_depth.png"
                render.silhouette.save(os.path.join(args.output_dir, sil_rel))
                render.mask.save(os.path.join(args.output_dir, mask_rel))
                render.depth.save(os.path.join(args.output_dir, depth_rel))

                entry = {
                    "id": sample_id,
                    "global_orient": params.global_orient[sample_idx].tolist(),
                    "pose": params.pose[sample_idx].tolist(),
                    "betas": params.betas[sample_idx].tolist(),
                    "silhouette": sil_rel,
                    "mask": mask_rel,
                    "depth": depth_rel,
                    "depth_min": render.depth_min,
                    "depth_max": render.depth_max,
                    "camera_azimuth_deg": float(np.degrees(azimuths[sample_idx])),
                    "camera_elevation_deg": float(np.degrees(elevations[sample_idx])),
                }
                if params.extra_joint_fill:
                    entry["extra_joint_fill"] = params.extra_joint_fill[sample_idx]
                manifest.write(json.dumps(entry) + "\n")

            print(f"[{end}/{len(params)}] samples rendered", flush=True)

    renderer.delete()


if __name__ == "__main__":
    main()
