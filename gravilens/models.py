"""Models: one physics model (GraviLensP v2) and three parameter-matched baselines.

GraviLensP forward:
    init tokens -> physics stack -> theta_E -> SIS inverse -> source tokens
                -> cross-attention at the FRONT of the classifier stack
                -> classifier stack -> pooled head.

The source therefore conditions every classifier block, and ``no_sis`` swaps
exactly one tensor (the source image). No hidden state, no dead heads.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import ModelConfig
from .physics import sis_invert_bilinear, theta_from_unit

VARIANTS = ("full", "no_sup", "no_sis", "dist")


def _mlp(in_dim: int, hidden_dim: int, out_dim: int,
         hidden_layers: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
    for _ in range(max(0, hidden_layers - 1)):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Tokenizer + window-attention trunk (GraviLens)
# ---------------------------------------------------------------------------

class ShiftedPatchTokenizer(nn.Module):
    """SPT: concatenate four shifted copies, then a strided convolution."""

    def __init__(self, image_size: int, patch_size: int, embed_dim: int,
                 in_channels: int = 1):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.delta = patch_size // 2
        self.projection = nn.Conv2d(in_channels * 5, embed_dim,
                                    kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.positional = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.positional, std=0.02)

    @staticmethod
    def _shift(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
        out = torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
        if dy > 0:
            out[..., :dy, :] = 0
        elif dy < 0:
            out[..., dy:, :] = 0
        if dx > 0:
            out[..., :, :dx] = 0
        elif dx < 0:
            out[..., :, dx:] = 0
        return out

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        d = self.delta
        shifted = torch.cat([
            image,
            self._shift(image, -d, -d),
            self._shift(image, d, -d),
            self._shift(image, -d, d),
            self._shift(image, d, d),
        ], dim=1)
        tokens = self.projection(shifted).flatten(2).transpose(1, 2)
        return self.norm(tokens) + self.positional


class WindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int,
                 dropout: float = 0.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        coords_flat = coords.flatten(1)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", rel.sum(-1))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)]
        bias = bias.reshape(N, N, -1).permute(2, 0, 1)
        attn = (attn + bias.unsqueeze(0)).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.drop(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    @staticmethod
    def _partition(x: torch.Tensor, ws: int) -> torch.Tensor:
        B, H, W, C = x.shape
        x = x.view(B, H // ws, ws, W // ws, ws, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, C)

    @staticmethod
    def _reverse(windows: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
        B = windows.shape[0] // (H * W // ws // ws)
        x = windows.view(B, H // ws, W // ws, ws, ws, -1)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * W, -1)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x).view(x.shape[0], H, W, -1)
        x = self.attn(self._partition(x, self.window_size))
        x = self._reverse(x, self.window_size, H, W)
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class TransformerStack(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, window_size, mlp_ratio, dropout)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, H, W)
        return x


def distortion_map(image: torch.Tensor) -> torch.Tensor:
    """|tanh(d2/dxdy [log(Imax/I)]^2)|, mirroring the simulator's feature."""
    imax = image.amax(dim=(1, 2), keepdim=True)
    log_img = torch.log(imax.clamp_min(1e-12) / image.clamp_min(1e-12)).square()
    grad = torch.gradient(torch.gradient(log_img, dim=1)[0], dim=2)[0]
    return torch.where(imax > 0, torch.tanh(grad).abs(), torch.zeros_like(grad))


class GraviLensP(nn.Module):
    """Physics-informed classifier. Variants differ in exactly one tensor."""

    def __init__(self, cfg: ModelConfig, variant: str = "full",
                 theta_min: float = 0.5, theta_max: float = 2.0,
                 pixel_scale: float = 0.101, core_radius: float = 0.01):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if cfg.image_size % cfg.patch_size or cfg.window_size <= 0:
            raise ValueError("invalid patch/window configuration")
        self.cfg = cfg
        self.variant = variant
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.pixel_scale = pixel_scale
        self.core_radius = core_radius
        self.H = self.W = cfg.image_size // cfg.patch_size
        if self.H % cfg.window_size:
            raise ValueError(f"H={self.H} must be divisible by window_size={cfg.window_size}")

        self.tokenizer = ShiftedPatchTokenizer(
            cfg.image_size, cfg.patch_size, cfg.embed_dim, cfg.in_channels)
        self.physics_stack = TransformerStack(
            cfg.embed_dim, cfg.num_blocks, cfg.num_heads, cfg.window_size,
            cfg.mlp_ratio, cfg.dropout)
        self.theta_head = nn.Sequential(
            nn.Linear(cfg.embed_dim, 512), nn.GELU(), nn.LayerNorm(512), nn.Dropout(cfg.dropout),
            nn.Linear(512, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(cfg.dropout),
            nn.Linear(256, 1),
        )
        self.source_tokenizer = ShiftedPatchTokenizer(
            cfg.image_size, cfg.patch_size, cfg.embed_dim, cfg.in_channels)
        self.cross_norm = nn.LayerNorm(cfg.embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            cfg.embed_dim, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
        self.classifier_stack = TransformerStack(
            cfg.embed_dim, cfg.num_blocks, cfg.num_heads, cfg.window_size,
            cfg.mlp_ratio, cfg.dropout)
        self.class_head = _mlp(
            cfg.embed_dim, cfg.hidden_dim, cfg.num_classes,
            cfg.hidden_layers, cfg.dropout)

        if variant == "dist":
            self.dist_tokenizer = ShiftedPatchTokenizer(
                cfg.image_size, cfg.patch_size, cfg.embed_dim, cfg.in_channels)

    def forward(self, images: torch.Tensor, return_aux: bool = False):
        if images.dim() == 3:
            images = images.unsqueeze(1)
        if images.dim() != 4 or images.shape[1] != self.cfg.in_channels:
            raise ValueError(f"expected [B,{self.cfg.in_channels},H,W], got {tuple(images.shape)}")
        B = images.shape[0]

        init = self.tokenizer(images)
        if self.variant == "dist":
            with torch.no_grad():
                dist = distortion_map(images[:, 0])
            init = init + self.dist_tokenizer(dist.unsqueeze(1))

        encoded = self.physics_stack(init, self.H, self.W)
        theta = theta_from_unit(
            self.theta_head(encoded.mean(dim=1)).squeeze(-1),
            self.theta_min, self.theta_max)

        if self.variant == "no_sis":
            flat = images.flatten(1)
            source = images / flat.amax(dim=1).clamp_min(1e-8).view(B, 1, 1, 1)
        else:
            source = sis_invert_bilinear(images, theta, self.pixel_scale, self.core_radius)

        source_tokens = self.source_tokenizer(source)
        query = self.cross_norm(init)
        fused = init + self.cross_attn(query, source_tokens, source_tokens,
                                       need_weights=False)[0]
        logits = self.class_head(self.classifier_stack(fused, self.H, self.W).mean(dim=1))
        return (logits, theta, source) if return_aux else logits


# ---------------------------------------------------------------------------
# Baselines (parameter-matched by match_budget)
# ---------------------------------------------------------------------------

class _PatchEmbed(nn.Module):
    def __init__(self, cfg: ModelConfig, dim: int, cls_token: bool = True):
        super().__init__()
        self.proj = nn.Conv2d(cfg.in_channels, dim, cfg.patch_size, cfg.patch_size)
        self.num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) if cls_token else None
        n = self.num_patches + int(cls_token)
        self.positional = nn.Parameter(torch.zeros(1, n, dim))
        nn.init.trunc_normal_(self.positional, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(x).flatten(2).transpose(1, 2)
        if self.cls_token is not None:
            tokens = torch.cat([self.cls_token.expand(tokens.shape[0], -1, -1), tokens], dim=1)
        return tokens + self.positional


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class ViTModel(nn.Module):
    def __init__(self, cfg: ModelConfig, dim: int = 320, depth: int = 3,
                 num_heads: int = 8, mlp_dim: int | None = None,
                 dropout: float = 0.1):
        super().__init__()
        self.embed = _PatchEmbed(cfg, dim, cls_token=True)
        self.blocks = nn.ModuleList([
            _Block(dim, num_heads, mlp_dim or 4 * dim, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        tokens = self.embed(x)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.norm(tokens[:, 0]))


class _ClassBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, cls: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        query = self.norm_q(cls)
        kv = self.norm_kv(patches)
        cls = cls + self.attn(query, kv, kv, need_weights=False)[0]
        return cls + self.mlp(self.norm_mlp(cls))


class CaiTModel(nn.Module):
    def __init__(self, cfg: ModelConfig, dim: int = 320, depth: int = 3,
                 cls_depth: int = 1, num_heads: int = 8,
                 mlp_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        mlp_dim = mlp_dim or 4 * dim
        self.embed = _PatchEmbed(cfg, dim, cls_token=False)
        self.blocks = nn.ModuleList([
            _Block(dim, num_heads, mlp_dim, dropout) for _ in range(depth)
        ])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.cls_blocks = nn.ModuleList([
            _ClassBlock(dim, num_heads, mlp_dim, dropout) for _ in range(cls_depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        patches = self.embed(x)
        for block in self.blocks:
            patches = block(patches)
        cls = self.cls_token.expand(patches.shape[0], -1, -1)
        for block in self.cls_blocks:
            cls = block(cls, patches)
        return self.head(self.norm(cls[:, 0]))


class ViTSDModel(nn.Module):
    """ViT with locally self-attention: the diagonal is masked out."""

    def __init__(self, cfg: ModelConfig, dim: int = 320, depth: int = 3,
                 num_heads: int = 8, mlp_dim: int | None = None,
                 dropout: float = 0.1):
        super().__init__()
        self.embed = _PatchEmbed(cfg, dim, cls_token=True)
        self.blocks = nn.ModuleList([
            _Block(dim, num_heads, mlp_dim or 4 * dim, dropout) for _ in range(depth)
        ])
        self.register_buffer(
            "attention_mask",
            torch.eye(1 + self.embed.num_patches, dtype=torch.bool))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        tokens = self.embed(x)
        for block in self.blocks:
            tokens = block(tokens, attn_mask=self.attention_mask)
        return self.head(self.norm(tokens[:, 0]))


# ---------------------------------------------------------------------------
# Construction + parameter budget
# ---------------------------------------------------------------------------

BASELINES = ("vit", "cait", "vitsd")

# Chosen by match_budget() against GraviLensP v2 (4,111,684 params); all within 2%.
BASELINE_SPECS = {
    "vit": {"dim": 288, "depth": 4, "num_heads": 8},
    "cait": {"dim": 288, "depth": 3, "num_heads": 8},
    "vitsd": {"dim": 288, "depth": 4, "num_heads": 8},
}


def build_model(name: str, cfg: ModelConfig, variant: str = "full",
                dim: int = 320, depth: int = 3, **kwargs) -> nn.Module:
    if name == "gravilens":
        return GraviLensP(cfg, variant, **kwargs)
    if name == "vit":
        return ViTModel(cfg, dim=dim, depth=depth, **kwargs)
    if name == "cait":
        return CaiTModel(cfg, dim=dim, depth=depth, **kwargs)
    if name == "vitsd":
        return ViTSDModel(cfg, dim=dim, depth=depth, **kwargs)
    raise ValueError(f"unknown model {name!r}; expected 'gravilens' or {BASELINES}")


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def match_budget(cfg: ModelConfig, name: str, target: int,
                 tolerance: float = 0.10,
                 dims: tuple[int, ...] = (192, 224, 256, 288, 320, 352, 384),
                 depths: tuple[int, ...] = (2, 3, 4)) -> dict:
    """Pick (dim, depth) for a baseline closest to ``target`` trainable params."""
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}")
    best: tuple[float, int, int, int] | None = None
    for dim in dims:
        heads = 8 if dim % 8 == 0 else 4
        for depth in depths:
            model = build_model(name, cfg, dim=dim, depth=depth, num_heads=heads)
            n = count_trainable_params(model)
            err = abs(n - target) / target
            if best is None or err < best[0]:
                best = (err, dim, depth, n)
            del model
    assert best is not None
    err, dim, depth, n = best
    if err > tolerance:
        raise ValueError(
            f"{name}: best match {n:,} params is {err:.1%} from target {target:,} "
            f"(tolerance {tolerance:.0%}); widen the search grid")
    return {"name": name, "dim": dim, "depth": depth, "params": n, "rel_err": err}
