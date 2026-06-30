"""Training-free NP-LoRA fusion primitives.

The implementation follows Eq. 12 of NP-LoRA (arXiv:2511.11051v3):
    D_merged = D_style + D_content @ (I - mu/(1 + mu) * V @ V.T)
where V is the right-singular-vector basis of the style update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


_DOWN_SUFFIXES = (".lora_down.weight", ".lora_A.weight")
_UP_SUFFIXES = (".lora_up.weight", ".lora_B.weight")


@dataclass(frozen=True)
class AdapterPair:
    """One conventional LoRA adapter represented by its checkpoint keys."""

    base: str
    down_key: str
    up_key: str
    alpha_key: str | None


def _find_key_with_suffix(state_dict: Mapping[str, torch.Tensor], base: str, suffixes: tuple[str, ...]) -> str | None:
    for suffix in suffixes:
        key = base + suffix
        if key in state_dict:
            return key
    return None


def find_adapter_pairs(state_dict: Mapping[str, torch.Tensor]) -> dict[str, AdapterPair]:
    """Return conventional up/down LoRA pairs, supporting Kohya and Diffusers names."""
    pairs: dict[str, AdapterPair] = {}
    for key in state_dict:
        suffix = next((s for s in _DOWN_SUFFIXES if key.endswith(s)), None)
        if suffix is None:
            continue
        base = key[: -len(suffix)]
        up_key = _find_key_with_suffix(state_dict, base, _UP_SUFFIXES)
        if up_key is None:
            continue
        alpha_key = next((candidate for candidate in (base + ".alpha", base + ".lora_alpha") if candidate in state_dict), None)
        pairs[base] = AdapterPair(base, key, up_key, alpha_key)
    return pairs


def _alpha_value(state_dict: Mapping[str, torch.Tensor], pair: AdapterPair) -> float:
    if pair.alpha_key is None:
        return float(state_dict[pair.down_key].shape[0])
    alpha = state_dict[pair.alpha_key]
    return float(alpha.detach().float().reshape(-1)[0].item())


def adapter_delta(state_dict: Mapping[str, torch.Tensor], pair: AdapterPair) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    """Materialize one LoRA update as a 2D FP32 matrix.

    1x1 convolution LoRAs are supported. Spatial LoCon kernels are deliberately
    rejected because a generic SVD factorisation cannot preserve their operator.
    """
    down = state_dict[pair.down_key]
    up = state_dict[pair.up_key]
    if not isinstance(down, torch.Tensor) or not isinstance(up, torch.Tensor):
        raise TypeError(f"{pair.base}: LoRA factors must be tensors")
    if down.ndim not in (2, 4) or up.ndim not in (2, 4):
        raise ValueError(f"{pair.base}: only linear or 1x1-convolution LoRA factors are supported")
    if up.ndim == 4 and tuple(up.shape[2:]) != (1, 1):
        raise ValueError(f"{pair.base}: spatial lora_up kernels are not supported")
    if up.shape[1] != down.shape[0]:
        raise ValueError(f"{pair.base}: incompatible LoRA rank ({up.shape[1]} vs {down.shape[0]})")
    up_2d = up.detach().to(device="cpu", dtype=torch.float32).reshape(up.shape[0], up.shape[1])
    down_2d = down.detach().to(device="cpu", dtype=torch.float32).reshape(down.shape[0], -1)
    scale = _alpha_value(state_dict, pair) / float(down.shape[0])
    return up_2d.matmul(down_2d).mul_(scale), tuple(up.shape), tuple(down.shape)


def np_project_content(content_delta: torch.Tensor, style_delta: torch.Tensor, mu: float) -> torch.Tensor:
    """Project content away from the style right-singular subspace (Eq. 12)."""
    if content_delta.shape != style_delta.shape:
        raise ValueError(f"Mismatched update shapes: {tuple(content_delta.shape)} and {tuple(style_delta.shape)}")
    if mu < 0:
        raise ValueError("mu must be non-negative")
    if mu == 0:
        return content_delta
    # full_matrices=False yields all non-zero-capable directions for a low-rank update.
    _, singular_values, vh = torch.linalg.svd(style_delta, full_matrices=False)
    # ``full_matrices=False`` still returns min(m, n) vectors, including null
    # directions when the update's true LoRA rank is lower.  NP-LoRA uses the
    # style update's rank-r_s subspace, so retain only numerically non-zero
    # singular directions.
    tolerance = torch.finfo(singular_values.dtype).eps * max(style_delta.shape) * singular_values[0]
    rank = int((singular_values > tolerance).sum().item())
    if rank == 0:
        return content_delta
    vh = vh[:rank]
    attenuation = float(mu) / (1.0 + float(mu))
    return content_delta - attenuation * (content_delta.matmul(vh.transpose(-2, -1))).matmul(vh)


def style_right_basis(state_dict: Mapping[str, torch.Tensor], pair: AdapterPair) -> torch.Tensor:
    """Construct the style right-singular subspace without a large matrix SVD.

    If ``D = L @ R`` is a LoRA update, QR on ``R.T`` followed by an SVD of the
    tiny rank-by-rank core gives the same non-zero right-singular subspace as
    an SVD of D.  This is the efficient QR route described in the paper's
    implementation appendix.
    """
    down = state_dict[pair.down_key].detach().to(device="cpu", dtype=torch.float32).reshape(
        state_dict[pair.down_key].shape[0], -1
    )
    up = state_dict[pair.up_key].detach().to(device="cpu", dtype=torch.float32).reshape(
        state_dict[pair.up_key].shape[0], state_dict[pair.up_key].shape[1]
    )
    q, r = torch.linalg.qr(down.transpose(0, 1), mode="reduced")
    core = up.matmul(r.transpose(0, 1))
    _, singular_values, vh = torch.linalg.svd(core, full_matrices=False)
    if singular_values.numel() == 0:
        return q[:, :0]
    tolerance = torch.finfo(singular_values.dtype).eps * max(core.shape) * singular_values[0]
    rank = int((singular_values > tolerance).sum().item())
    return q.matmul(vh[:rank].transpose(0, 1))


def _scaled_factors(
    state_dict: Mapping[str, torch.Tensor], pair: AdapterPair, strength: float
) -> tuple[torch.Tensor, torch.Tensor]:
    down = state_dict[pair.down_key].detach().to(device="cpu", dtype=torch.float32)
    up = state_dict[pair.up_key].detach().to(device="cpu", dtype=torch.float32)
    return up * (_alpha_value(state_dict, pair) / float(down.shape[0]) * strength), down


def _factors_are_compatible(
    content_up: torch.Tensor, content_down: torch.Tensor, style_up: torch.Tensor, style_down: torch.Tensor
) -> bool:
    return (
        content_up.shape[0] == style_up.shape[0]
        and tuple(content_up.shape[2:]) == tuple(style_up.shape[2:])
        and tuple(content_down.shape[1:]) == tuple(style_down.shape[1:])
    )


def factorize_delta(delta: torch.Tensor, up_shape: tuple[int, ...], down_shape: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return LoRA factors whose alpha/rank is one and whose product is ``delta``."""
    if delta.ndim != 2:
        raise ValueError("delta must be a matrix")
    u, s, vh = torch.linalg.svd(delta, full_matrices=False)
    # Put singular values in the up factor. This is exact up to normal FP32 SVD error.
    rank = int(s.numel())
    up = (u * s.unsqueeze(0)).reshape(up_shape[0], rank, *up_shape[2:]).contiguous()
    down = vh.reshape(rank, *down_shape[1:]).contiguous()
    alpha = torch.tensor(float(rank), dtype=torch.float32)
    return up, down, alpha


def _is_clip_adapter(base: str) -> bool:
    base = base.lower()
    return any(token in base for token in ("lora_te", "text_encoder", "clip_l", "clip_g"))


def _strength_for(base: str, model_strength: float, clip_strength: float) -> float:
    return float(clip_strength if _is_clip_adapter(base) else model_strength)


def fuse_lora_state_dicts(
    content: Mapping[str, torch.Tensor],
    style: Mapping[str, torch.Tensor],
    *,
    mu: float,
    content_model_strength: float = 1.0,
    style_model_strength: float = 1.0,
    content_clip_strength: float = 1.0,
    style_clip_strength: float = 1.0,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Fuse two conventional LoRA state dicts into one state dict for ComfyUI.

    Non-overlapping adapters are retained (with their requested strength). For
    overlapping adapters the result is re-factorised, so it remains a standard
    LoRA checkpoint and can be applied by ComfyUI's native loader.
    """
    if mu < 0:
        raise ValueError("mu must be non-negative")
    content_pairs = find_adapter_pairs(content)
    style_pairs = find_adapter_pairs(style)
    output: dict[str, torch.Tensor] = {}
    warnings: list[str] = []

    for base in sorted(set(content_pairs) | set(style_pairs)):
        c_pair = content_pairs.get(base)
        s_pair = style_pairs.get(base)
        template = s_pair or c_pair
        assert template is not None
        try:
            c_up = c_down = s_up = s_down = None
            if c_pair:
                c_up, c_down = _scaled_factors(
                    content, c_pair, _strength_for(base, content_model_strength, content_clip_strength)
                )
            if s_pair:
                s_up, s_down = _scaled_factors(
                    style, s_pair, _strength_for(base, style_model_strength, style_clip_strength)
                )

            if c_up is not None and s_up is not None:
                if not _factors_are_compatible(c_up, c_down, s_up, s_down):
                    raise ValueError("content and style adapter geometries differ")
                basis = style_right_basis(style, s_pair)
                attenuation = float(mu) / (1.0 + float(mu))
                content_down_2d = c_down.reshape(c_down.shape[0], -1)
                projected_down = content_down_2d - attenuation * (content_down_2d.matmul(basis)).matmul(basis.T)
                c_down = projected_down.reshape_as(c_down)
                up = torch.cat((s_up, c_up), dim=1)
                down = torch.cat((s_down, c_down), dim=0)
            elif s_up is not None:
                up, down = s_up, s_down
            else:
                up, down = c_up, c_down

            # Factors already represent the complete delta, hence alpha/rank=1.
            alpha = torch.tensor(float(down.shape[0]), dtype=torch.float32)
            output[template.down_key] = down
            output[template.up_key] = up
            output[template.alpha_key or (base + ".alpha")] = alpha
        except (TypeError, ValueError, RuntimeError) as error:
            warnings.append(f"Skipped {base}: {error}")

    if not output:
        raise ValueError("No compatible conventional LoRA up/down pairs were found in the selected files.")
    return output, warnings
