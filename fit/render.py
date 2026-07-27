"""Differentiable silhouette rendering (pytorch3d), driven by the exact same
WeakPerspectiveCamera used for keypoint reprojection (camera.py) -- so the
silhouette and reprojection losses can't be satisfied by two different,
mutually-inconsistent camera interpretations of the mesh.

Rasterizer/shader settings follow AniMerPlus's `amr/utils/mesh_renderer.py`
SilhouetteRenderer (SoftSilhouetteShader + MeshRasterizer, the standard
pytorch3d differentiable-silhouette recipe), swapped from its
FoVPerspectiveCameras (needs an assumed focal length) to OrthographicCameras
(matches the weak-perspective convention used throughout this fitting
pipeline and genzoo's own rendering).

pytorch3d's default camera looks down +Z in a left-handed frame (+X left),
which does not match VAREN's (post `varen_layer.VARENLayer` remap) X=width/
Y=up/Z=forward convention -- rendered naively, the horse comes out sideways
and upside down. Fixed by negating X and Y before rasterizing (verified by
actually rendering the standing-idle reference and checking it comes out
upright and forward-facing, not assumed) -- coincidentally the exact same
correction AniMerPlus's own SilhouetteRenderer applies for its meshes
(`R=[[-1,0,0],[0,-1,0],[0,0,1]]`), which makes sense since both are Y-up
mesh conventions targeting the same pytorch3d camera. This uses
`camera.WORLD_FLIP` (not a locally-redefined copy) so keypoint reprojection
and the mesh silhouette always apply the identical correction -- an earlier
version defined this flip separately in each file, which silently drifted
out of sync for the *screen-space* conversion (a related but distinct bug
from this world-space one -- see camera.py's docstring) and produced a
silhouette that landed nowhere near its own mesh's reprojected keypoints in
a real optimization run before being caught and fixed.
"""
from __future__ import annotations

import torch
from pytorch3d.renderer import (
    BlendParams,
    MeshRasterizer,
    RasterizationSettings,
    SoftSilhouetteShader,
)
from pytorch3d.renderer import MeshRenderer as Pytorch3dMeshRenderer
from pytorch3d.structures import Meshes

from .camera import WORLD_FLIP, WeakPerspectiveCamera


class DifferentiableSilhouetteRenderer:
    def __init__(self, image_size: int, device: str = "cuda", faces_per_pixel: int = 50) -> None:
        self.device = device
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=torch.log(torch.tensor(1.0 / 1e-4 - 1.0)).item() * blend_params.sigma,
            faces_per_pixel=faces_per_pixel,
            bin_size=0,
        )
        # cameras get swapped in per-call (they depend on the current WeakPerspectiveCamera
        # state, which changes every optimization step) -- placeholder here is unused.
        self.rasterizer = MeshRasterizer(raster_settings=raster_settings)
        self.shader = SoftSilhouetteShader(blend_params=blend_params)
        self.renderer = Pytorch3dMeshRenderer(rasterizer=self.rasterizer, shader=self.shader)

    def render(self, vertices: torch.Tensor, faces: torch.Tensor, camera: WeakPerspectiveCamera) -> torch.Tensor:
        """vertices: (B, V, 3) in VAREN's (post-remap) convention, faces: (B, F, 3) or (F, 3).
        Returns (B, H, W) soft silhouette in [0, 1]."""
        if faces.dim() == 2:
            faces = faces.unsqueeze(0).expand(vertices.shape[0], -1, -1)
        mesh = Meshes(verts=vertices * WORLD_FLIP.to(vertices.device), faces=faces)
        cameras = camera.to_pytorch3d_cameras(device=vertices.device)
        images = self.renderer(mesh, cameras=cameras)
        return images[..., 3]
