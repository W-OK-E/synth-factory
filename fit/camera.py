"""Weak-perspective camera: standard choice for optimization-based mesh fitting
to real, uncalibrated photos (SMPLify/HMR-family convention) -- avoids needing
a focal-length assumption, at the cost of not modeling real perspective
distortion. VAREN's own `global_orient` parameter serves as the
object-to-camera rotation (optimized directly), so the "camera" here only
ever needs scale + 2D translation on top of that.

Both keypoint reprojection and the differentiable silhouette (render.py) go
through this *same* pytorch3d OrthographicCameras object and its own
`transform_points_screen`, deliberately -- not two independently hand-rolled
projection formulas. That replaced an earlier, buggier version that used a
manual `scale * xy + translation` formula for keypoints while the silhouette
went through pytorch3d's own camera: the two disagreed for any nonzero
translation (verified: PyTorch3D's actual NDC/screen convention has both
axes reversed relative to the naive assumption -- `transform_points_screen`
gives `size - naive_formula(x, y)`, not `naive_formula(x, y)` -- and a
hand-derived translation-flip fix on top of that made it *worse*, not
better, because it stacked with the wrong sign). Routing both paths through
the identical pytorch3d call removes the whole class of "did I get the sign
right" bugs by construction rather than by re-derivation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from pytorch3d.renderer import OrthographicCameras

# World-space orientation fix (see render.py's docstring for the derivation):
# VAREN's (post varen_layer remap) convention doesn't match what this camera
# setup expects without it. Applied to *any* 3D point -- keypoints or mesh
# vertices -- before it ever reaches the pytorch3d camera, so both paths see
# the same world.
WORLD_FLIP = torch.tensor([-1.0, -1.0, 1.0])


class WeakPerspectiveCamera(nn.Module):
    def __init__(self, batch_size: int = 1, init_scale: float = 1.0) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.full((batch_size,), float(torch.tensor(init_scale).log())))
        self.translation = nn.Parameter(torch.zeros(batch_size, 2))

    @property
    def scale(self) -> torch.Tensor:
        return self.log_scale.exp()  # keep scale positive under unconstrained optimization

    def to_pytorch3d_cameras(self, device) -> OrthographicCameras:
        batch_size = self.log_scale.shape[0]
        R = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        T = torch.zeros(batch_size, 3, device=device)
        return OrthographicCameras(
            device=device, R=R, T=T, focal_length=self.scale.to(device), principal_point=self.translation.to(device)
        )

    def project_to_pixels(self, points_3d: torch.Tensor, image_size: int) -> torch.Tensor:
        """points_3d: (B, N, 3), raw VAREN convention. Returns (B, N, 2) pixel
        coordinates (x right, y down, origin top-left) via pytorch3d's own
        transform_points_screen -- the same conversion render.py's silhouette
        rasterizer uses internally, guaranteed consistent by construction."""
        device = points_3d.device
        flipped = points_3d * WORLD_FLIP.to(device)
        cameras = self.to_pytorch3d_cameras(device)
        screen = cameras.transform_points_screen(flipped, image_size=((image_size, image_size),) * flipped.shape[0])
        return screen[..., :2]
