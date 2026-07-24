"""Render VAREN meshes into images usable as image-gen conditioning.

Produces, per mesh:
  - silhouette: a matte-shaded RGB render on a flat background (the format
    expected by data-gen/utils/flux2img.py as its init image).
  - mask: binary foreground/background mask.
  - depth: normalized depth map (near = bright), an alternative
    conditioning signal.

Coordinate/lighting conventions mirror AniMerPlus/amr/utils/mesh_renderer.py
(the pyrender setup already used elsewhere in this codebase), except the
camera auto-frames each mesh via its bounding sphere instead of assuming a
fixed human-scale focal length -- VAREN meshes vary in scale with betas.
"""
from __future__ import annotations

import os

if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

from dataclasses import dataclass

import numpy as np
import pyrender
import trimesh
from PIL import Image

from amr.utils.mesh_renderer import create_raymond_lights

BACKGROUND_COLOR = (128, 128, 128)
MESH_COLOR = (0.65, 0.55, 0.45, 1.0)


@dataclass
class Render:
    silhouette: Image.Image
    mask: Image.Image
    depth: Image.Image
    depth_min: float
    depth_max: float


class MeshRenderer:
    """Auto-framing offscreen renderer, one mesh at a time."""

    def __init__(
        self,
        faces: np.ndarray,
        image_size: int = 1024,
        fov_degrees: float = 45.0,
        margin: float = 1.2,
    ) -> None:
        self.faces = faces
        self.fov = np.radians(fov_degrees)
        self.margin = margin
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=image_size, viewport_height=image_size, point_size=1.0
        )
        self.material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, roughnessFactor=0.9, baseColorFactor=MESH_COLOR
        )

    def _camera_pose(
        self, vertices: np.ndarray, azimuth: float, elevation: float
    ) -> np.ndarray:
        center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
        radius = np.linalg.norm(vertices - center, axis=1).max()
        distance = (radius * self.margin) / np.sin(self.fov / 2.0)

        direction = np.array(
            [
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
                np.cos(elevation) * np.cos(azimuth),
            ]
        )
        eye = center + distance * direction

        # Look-at basis: camera looks down local -Z, local +Y is up.
        world_up = np.array([0.0, 1.0, 0.0])
        z_axis = direction  # eye - center, normalized (direction already is)
        if abs(np.dot(z_axis, world_up)) > 0.999:  # near top/bottom, guard anyway
            world_up = np.array([0.0, 0.0, 1.0])
        x_axis = np.cross(world_up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        pose = np.eye(4)
        pose[:3, 0] = x_axis
        pose[:3, 1] = y_axis
        pose[:3, 2] = z_axis
        pose[:3, 3] = eye
        return pose

    def render(
        self, vertices: np.ndarray, azimuth: float = 0.0, elevation: float = 0.0
    ) -> Render:
        mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy(), process=False)
        # VAREN's raw frame needs a -90 deg rotation about X to stand the
        # horse upright facing the camera (empirically checked against
        # example_params.json's mean pose; VAREN's convention differs from
        # the 180 deg flip used for human SMPL meshes in AniMerPlus).
        flip = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        mesh.apply_transform(flip)
        py_mesh = pyrender.Mesh.from_trimesh(mesh, material=self.material)

        scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.4, 0.4, 0.4)
        )
        scene.add(py_mesh, "mesh")

        camera = pyrender.PerspectiveCamera(yfov=self.fov, aspectRatio=1.0)
        camera_node = scene.add(
            camera, pose=self._camera_pose(mesh.vertices, azimuth, elevation)
        )

        # Lights are attached to the camera (not the scene root) so
        # illumination stays consistent as the camera orbits around the
        # horse -- otherwise far-side views would be backlit/underexposed.
        for light_node in create_raymond_lights():
            scene.add_node(light_node, parent_node=camera_node)

        color, depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

        # Some antialiased edge pixels get partial alpha with no depth
        # sample written; depth > 0 is the reliable "is there a surface
        # here" signal.
        mask = depth > 0
        rgb = color[:, :, :3].astype(np.float32)
        background = np.array(BACKGROUND_COLOR, dtype=np.float32)
        composited = np.where(mask[:, :, None], rgb, background).astype(np.uint8)

        depth_img = np.zeros_like(depth, dtype=np.uint8)
        depth_min = depth_max = 0.0
        if mask.any():
            fg_depth = depth[mask]
            depth_min, depth_max = float(fg_depth.min()), float(fg_depth.max())
            span = max(depth_max - depth_min, 1e-6)
            normalized = 1.0 - (depth - depth_min) / span  # near = bright
            depth_img[mask] = np.clip(normalized[mask] * 255.0, 0, 255).astype(np.uint8)

        return Render(
            silhouette=Image.fromarray(composited, mode="RGB"),
            mask=Image.fromarray((mask * 255).astype(np.uint8), mode="L"),
            depth=Image.fromarray(depth_img, mode="L"),
            depth_min=depth_min,
            depth_max=depth_max,
        )

    def delete(self) -> None:
        self.renderer.delete()
