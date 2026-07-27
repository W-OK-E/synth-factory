"""Renders the standing-idle reference mesh with the AP10K/APT36K<->VAREN
keypoint correspondence (keypoint_map.py) overlaid as actual 3D sphere
markers *in the same pyrender scene* as the horse mesh -- not a separately
re-derived 2D projection (which is exactly what went wrong on the first
attempt at this: a hand-rolled projection had a coordinate-convention bug,
producing markers that kept their correct *relative* clustering but floated
disconnected from the mesh -- see git history / session notes). Putting real
3D geometry into one scene and letting pyrender's own camera project
everything sidesteps re-implementing that math at all.

    cd /home/om/mpi/data-gen/synth_2d_to_3d
    micromamba run -n animer2 python fit/verify_keypoint_map.py
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.append("/home/om/mpi/data-gen")
sys.path.append("/home/om/mpi/data-gen/genzoo")
sys.path.append("/home/om/mpi/VAREN")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyrender
import torch
import trimesh
from PIL import Image
from varen.vertex_ids import vertex_ids as VERTEX_IDS

from fit.keypoint_map import AP10K_TO_VAREN, get_keypoint_positions
from synth3d2img.mesh import VarenMeshGenerator, axis_angle_to_rotmat

COLORS = {
    "L_Eye": (255, 0, 0), "R_Eye": (0, 255, 0), "Nose": (0, 0, 255), "Neck": (255, 255, 0),
    "Root of tail": (255, 0, 255), "L_Shoulder": (0, 255, 255), "L_Elbow": (255, 128, 0),
    "L_F_Paw": (128, 0, 255), "R_Shoulder": (0, 128, 255), "R_Elbow": (128, 255, 0),
    "R_F_Paw": (255, 0, 128), "L_Hip": (0, 255, 128), "L_Knee": (128, 128, 0),
    "L_B_Paw": (0, 128, 128), "R_Hip": (128, 0, 0), "R_Knee": (0, 0, 128), "R_B_Paw": (200, 200, 200),
}


def main() -> None:
    vertex_ids = VERTEX_IDS["varen"]

    with open("reference_params/standing_idle.json") as f:
        ref = json.load(f)

    mesh_gen = VarenMeshGenerator("/home/om/mpi/VAREN/models", device="cpu")
    pose_aa = torch.as_tensor(np.array([ref["pose"]]), dtype=torch.float32).reshape(1, 37, 3)
    orient_aa = torch.as_tensor(np.array([ref["global_orient"]]), dtype=torch.float32).reshape(1, 1, 3)
    output = mesh_gen.model(
        body_pose=axis_angle_to_rotmat(pose_aa),
        global_orient=axis_angle_to_rotmat(orient_aa),
        betas=torch.as_tensor(np.array([ref["betas"]]), dtype=torch.float32),
    )
    vertices, joints = output.vertices, output.joints
    keypoint_names = list(AP10K_TO_VAREN.keys())
    kp_positions = get_keypoint_positions(vertices, joints, vertex_ids, keypoint_names)[0].detach().numpy()

    # Same 180deg-about-X flip MeshRenderer.render() applies internally to the horse mesh
    # (see render.py) -- applied here to the markers too, so both live in the same frame.
    flip = np.array([1.0, -1.0, -1.0])
    verts_np = vertices[0].detach().numpy() * flip
    kp_np = kp_positions * flip
    faces = mesh_gen.faces

    horse_mesh = trimesh.Trimesh(verts_np, faces, process=False)
    horse_mesh.visual.face_colors = [200, 200, 200, 255]

    scene = pyrender.Scene(bg_color=[80, 80, 80, 255], ambient_light=(0.6, 0.6, 0.6))
    scene.add(pyrender.Mesh.from_trimesh(horse_mesh, smooth=False))

    radius_marker = 0.04
    for name in keypoint_names:
        sphere = trimesh.creation.icosphere(radius=radius_marker)
        sphere.apply_translation(kp_np[keypoint_names.index(name)])
        color = COLORS[name]
        sphere.visual.face_colors = [*color, 255]
        scene.add(pyrender.Mesh.from_trimesh(sphere, smooth=False))

    for pos in [[0, -1, 1], [0, 1, 1], [1, 1, 2]]:
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5)
        light_pose = np.eye(4)
        light_pose[:3, 3] = pos
        scene.add(light, pose=light_pose)

    center = (verts_np.min(axis=0) + verts_np.max(axis=0)) / 2
    radius = np.linalg.norm(verts_np - center, axis=1).max()
    fov = np.radians(45.0)
    distance = radius * 1.2 / np.sin(fov / 2)
    azimuth, elevation = 1.55, 0.15
    direction = np.array(
        [np.cos(elevation) * np.sin(azimuth), np.sin(elevation), np.cos(elevation) * np.cos(azimuth)]
    )
    eye = center + distance * direction
    world_up = np.array([0.0, 1.0, 0.0])
    z_axis = direction
    x_axis = np.cross(world_up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    cam_pose = np.eye(4)
    cam_pose[:3, 0] = x_axis
    cam_pose[:3, 1] = y_axis
    cam_pose[:3, 2] = z_axis
    cam_pose[:3, 3] = eye

    camera = pyrender.PerspectiveCamera(yfov=fov, aspectRatio=1.0)
    scene.add(camera, pose=cam_pose)

    renderer = pyrender.OffscreenRenderer(768, 768)
    color, _ = renderer.render(scene)
    renderer.delete()

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/keypoint_map_check.png"
    Image.fromarray(color).save(out_path)
    print(f"Wrote {out_path}")
    print("legend:", COLORS)


if __name__ == "__main__":
    main()
