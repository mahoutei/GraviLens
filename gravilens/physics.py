"""SIS lensing: the only physics path in the model.

Conventions
-----------
* ``theta_e`` is the Einstein radius in arcsec, shape ``[B]``.
* Image and source grids span ``[-(S-1)/2, +(S-1)/2] * pixel_scale`` arcsec.
* ``sis_invert_bilinear`` is the differentiable inverse used by the model.
* ``sis_invert_scatter`` is the nearest-pixel reference used only for oracle
  comparisons; it is deliberately not differentiable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

THETA_E_MIN = 0.5
THETA_E_MAX = 2.0
PIXEL_SCALE = 0.101

__all__ = [
    "THETA_E_MIN",
    "THETA_E_MAX",
    "PIXEL_SCALE",
    "theta_from_unit",
    "coordinate_grid",
    "sis_image_positions",
    "sis_invert_bilinear",
    "sis_invert_scatter",
]


def theta_from_unit(raw: torch.Tensor, low: float = THETA_E_MIN,
                    high: float = THETA_E_MAX) -> torch.Tensor:
    """Map an unconstrained scalar to the simulator's valid theta_E interval."""
    return low + (high - low) * torch.sigmoid(raw)


def coordinate_grid(size: int, pixel_scale: float = PIXEL_SCALE,
                    device=None, dtype=None) -> tuple[torch.Tensor, torch.Tensor]:
    half = (size - 1) / 2.0
    c = (torch.arange(size, device=device, dtype=dtype) - half) * pixel_scale
    return torch.meshgrid(c, c, indexing="ij")


def sis_image_positions(beta_x: torch.Tensor, beta_y: torch.Tensor,
                        theta_e: torch.Tensor):
    """The two image positions of a source at (beta_x, beta_y), in arcsec."""
    r = torch.sqrt(beta_x ** 2 + beta_y ** 2).clamp_min(1e-8)
    ux, uy = beta_x / r, beta_y / r
    return (beta_x + theta_e * ux, beta_y + theta_e * uy,
            beta_x - theta_e * ux, beta_y - theta_e * uy)


def sis_invert_bilinear(images: torch.Tensor, theta_e: torch.Tensor,
                        pixel_scale: float = PIXEL_SCALE,
                        core: float = 0.01) -> torch.Tensor:
    """Differentiable SIS inversion.

    For every source-plane point beta we sample the lensed image at
    ``beta +/- theta_E * beta_hat`` and take the surface-brightness-weighted
    mean. Gradients flow from the output to ``theta_e`` through ``grid_sample``.

    Args:
        images: ``[B, S, S]`` or ``[B, 1, S, S]``.
        theta_e: ``[B]`` or ``[B, 1]`` arcsec.
    Returns:
        ``[B, 1, S, S]``, max-normalised per image.
    """
    squeeze = images.dim() == 4
    if squeeze:
        images = images[:, 0]
    if images.dim() != 3:
        raise ValueError(f"expected [B,S,S] or [B,1,S,S], got {tuple(images.shape)}")

    B, S, _ = images.shape
    half = (S - 1) / 2.0
    c = (torch.arange(S, device=images.device, dtype=images.dtype) - half) * pixel_scale
    bi, bj = torch.meshgrid(c, c, indexing="ij")
    bi, bj = bi[None], bj[None]                       # [1,S,S]

    te = theta_e.reshape(B, 1, 1).to(images.dtype)
    b = torch.sqrt(bi ** 2 + bj ** 2 + core ** 2)
    ui, uj = bi / b, bj / b
    maxc = half * pixel_scale
    img4 = images.unsqueeze(1)                        # [B,1,S,S]

    def sample(ti: torch.Tensor, tj: torch.Tensor):
        grid = torch.stack([tj / maxc, ti / maxc], dim=-1)   # x = axis 1, y = axis 0
        v = F.grid_sample(img4, grid, mode="bilinear", padding_mode="zeros",
                          align_corners=True)[:, 0]
        inside = ((ti.abs() <= maxc) & (tj.abs() <= maxc)).to(images.dtype)
        return v, inside

    v1, m1 = sample(bi + te * ui, bj + te * uj)
    v2, m2 = sample(bi - te * ui, bj - te * uj)
    # Second image exists only inside the caustic (|beta| < theta_E).
    w2 = m2 * torch.sigmoid((te - b) / (0.5 * pixel_scale))
    src = (m1 * v1 + w2 * v2) / (m1 + w2).clamp_min(1e-6)
    src = src / src.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    return src.unsqueeze(1)


def sis_invert_scatter(images: torch.Tensor, theta_e: torch.Tensor,
                       pixel_scale: float = PIXEL_SCALE,
                       core: float = 0.01) -> torch.Tensor:
    """Nearest-pixel reference inversion (oracle comparison only)."""
    squeeze = images.dim() == 4
    if squeeze:
        images = images[:, 0]
    B, S, _ = images.shape
    half = (S - 1) / 2.0
    c = (torch.arange(S, device=images.device, dtype=images.dtype) - half) * pixel_scale
    gx, gy = torch.meshgrid(c, c, indexing="ij")
    gx, gy = gx.flatten()[None].expand(B, -1), gy.flatten()[None].expand(B, -1)
    k = theta_e.reshape(B, 1).to(images.dtype)
    r = torch.sqrt(gx ** 2 + gy ** 2 + core ** 2)
    sx, sy = gx - k * gx / r, gy - k * gy / r
    maxc = half * pixel_scale
    px = torch.clamp(((sx + maxc) / (2 * maxc) * (S - 1)).round().long(), 0, S - 1)
    py = torch.clamp(((sy + maxc) / (2 * maxc) * (S - 1)).round().long(), 0, S - 1)
    idx = px * S + py
    flat = images.reshape(B, -1)
    out = torch.zeros(B, S * S, device=images.device, dtype=images.dtype).scatter_add_(1, idx, flat)
    cnt = torch.zeros_like(out).scatter_add_(1, idx, torch.ones_like(flat))
    out = torch.where(cnt > 0, out / cnt.clamp_min(1), out)
    out = out / out.amax(dim=1, keepdim=True).clamp_min(1e-8)
    return out.view(B, 1, S, S)
