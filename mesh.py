"""Batched mesh generation from VAREN parameters."""
from __future__ import annotations

import numpy as np
import torch
from varen import VAREN


class VarenMeshGenerator:
    """Loads VAREN once and turns batches of params into mesh vertices."""

    def __init__(
        self,
        model_path: str,
        use_muscle_deformations: bool = True,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = VAREN(
            model_path, use_muscle_deformations=use_muscle_deformations
        ).to(self.device)
        self.model.eval()
        self.faces = self.model.faces

    @torch.no_grad()
    def generate(
        self,
        global_orient: np.ndarray,
        pose: np.ndarray,
        betas: np.ndarray,
    ) -> np.ndarray:
        """Run a batched VAREN forward pass.

        Args:
            global_orient: (B, 3) array.
            pose: (B, 111) array.
            betas: (B, 39) array.

        Returns:
            vertices: (B, V, 3) array.
        """
        batch_size = pose.shape[0]
        transl = torch.zeros(batch_size, 3, dtype=torch.float32, device=self.device)

        output = self.model(
            body_pose=torch.as_tensor(pose, dtype=torch.float32, device=self.device),
            betas=torch.as_tensor(betas, dtype=torch.float32, device=self.device),
            global_orient=torch.as_tensor(
                global_orient, dtype=torch.float32, device=self.device
            ),
            transl=transl,
        )
        return output.vertices.detach().cpu().numpy()
