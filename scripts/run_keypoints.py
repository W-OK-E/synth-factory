"""Run ViTPose keypoint models over Horse-image-data (train/valid/test) using
the dataset's own GT bboxes (Roboflow COCO export -- xywh format, single
'hourses' category), and save both the numeric keypoint results and
visualizations back into Horse-image-data itself.

The stock demo (ViTPose/demo/top_down_img_demo.py) only saves visualization
images -- pose_results (the actual (x, y, score) keypoints) are computed but
never written anywhere. This script is that same inference pattern, plus
actually persisting the numbers, since the downstream mesh-fitting pipeline
needs the coordinates, not just a picture.

Run from the ViTPose directory (needs its mmpose/mmcv on the path), with the
project venv:

    cd /home/om/mpi/data-gen/synth_2d_to_3d/ViTPose
    ../.venv/bin/python ../scripts/run_keypoints.py --model horse10
    ../.venv/bin/python ../scripts/run_keypoints.py --model vitpose_ap10k
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

from xtcocotools.coco import COCO

from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
from mmpose.datasets import DatasetInfo

DATA_ROOT = "/home/om/mpi/data-gen/Horse-image-data"
SPLITS = ["train", "valid", "test"]

MODELS = {
    "vitpose_apt36k": dict(
        config="configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/apt36k/ViTPose_base_apt36k_256x192.py",
        checkpoint="checkpoints/apt36k.pth",
    ),
    "horse10": dict(
        config="configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/res50_horse10_256x256-split1.py",
        checkpoint="https://download.openmmlab.com/mmpose/animal/resnet/res50_horse10_256x256_split1-3a3dc37e_20210405.pth",
    ),
    "vitpose_ap10k": dict(
        config="configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ViTPose_base_ap10k_256x192.py",
        checkpoint="checkpoints/ap10k.pth",  # split out of vitpose+_base.pth via tools/model_split.py
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kpt-thr", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_cfg = MODELS[args.model]

    pose_model = init_pose_model(model_cfg["config"], model_cfg["checkpoint"], device=args.device)
    dataset = pose_model.cfg.data["test"]["type"]
    dataset_info = pose_model.cfg.data["test"].get("dataset_info", None)
    if dataset_info is None:
        warnings.warn("no dataset_info in config, falling back to legacy keypoint order")
    else:
        dataset_info = DatasetInfo(dataset_info)

    for split in args.splits:
        split_dir = os.path.join(DATA_ROOT, split)
        json_file = os.path.join(split_dir, "_annotations.coco.json")
        coco = COCO(json_file)

        out_vis_dir = os.path.join(split_dir, f"vis_{args.model}")
        os.makedirs(out_vis_dir, exist_ok=True)
        results = []

        img_ids = list(coco.imgs.keys())
        for i, image_id in enumerate(img_ids):
            image = coco.loadImgs(image_id)[0]
            image_path = os.path.join(split_dir, image["file_name"])
            ann_ids = coco.getAnnIds(image_id)

            person_results = [{"bbox": coco.anns[a]["bbox"]} for a in ann_ids]
            if not person_results:
                continue

            pose_results, _ = inference_top_down_pose_model(
                pose_model,
                image_path,
                person_results,
                bbox_thr=None,
                format="xywh",
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=False,
                outputs=None,
            )

            out_file = os.path.join(out_vis_dir, image["file_name"])
            vis_pose_result(
                pose_model, image_path, pose_results, dataset=dataset, dataset_info=dataset_info,
                kpt_score_thr=args.kpt_thr, out_file=out_file,
            )

            for ann_id, pose in zip(ann_ids, pose_results):
                results.append(
                    {
                        "image_id": image_id,
                        "file_name": image["file_name"],
                        "annotation_id": ann_id,
                        "bbox": coco.anns[ann_id]["bbox"],
                        "keypoints": pose["keypoints"].tolist(),  # (K, 3): x, y, score
                    }
                )
            print(f"[{split}/{args.model}] {i + 1}/{len(img_ids)}: {image['file_name']}")

        keypoint_names = None
        if dataset_info is not None:
            keypoint_names = [dataset_info.keypoint_id2name[k] for k in sorted(dataset_info.keypoint_id2name)]

        out_json = os.path.join(split_dir, f"keypoints_{args.model}.json")
        with open(out_json, "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "config": model_cfg["config"],
                    "checkpoint": model_cfg["checkpoint"],
                    "keypoint_names": keypoint_names,
                    "results": results,
                },
                f,
            )
        print(f"Wrote {out_json} ({len(results)} instances) and {out_vis_dir}/")


if __name__ == "__main__":
    main()
