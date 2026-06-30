"""SSR-Merge primitives for ComfyUI LoRA state dicts.

This follows the public SSR-Merge formulation:

* concatenate LoRA down projections into a unified subspace,
* collect second-order activation statistics during a calibration pass,
* solve the subspace router, and
* absorb the router into the up projection so the result is a normal LoRA.

The calibration runner lives in ``ssr_nodes.py`` because it needs ComfyUI's
sampler, CLIP, and model patcher objects.  This module is deliberately small
and tensor-focused so it can be regression tested without launching ComfyUI.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Mapping

import torch


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SSRLayer:
    """One LoRA layer in ComfyUI's target-weight namespace."""

    lora_base: str
    target_base: str
    down: torch.Tensor
    up: torch.Tensor
    alpha: float

    @property
    def rank(self) -> int:
        return int(self.down.shape[0])

    @property
    def is_conv1x1(self) -> bool:
        return self.down.ndim == 4

    def down_2d(self) -> torch.Tensor:
        return self.down.detach().to(device="cpu", dtype=torch.float32).reshape(self.rank, -1)

    def scaled_up_2d(self, strength: float) -> torch.Tensor:
        up = self.up.detach().to(device="cpu", dtype=torch.float32).reshape(self.up.shape[0], self.up.shape[1])
        return up * (float(self.alpha) / float(self.rank) * float(strength))


@dataclass
class SSRMergedLoRA:
    """Merged LoRA state dict plus human-readable calibration metadata."""

    state_dict: dict[str, torch.Tensor]
    cache_key: str
    info: str
    warnings: list[str]


class SSRStats:
    """Accumulate and solve SSR statistics for multiple LoRAs."""

    def __init__(
        self,
        layers_per_task: list[Mapping[str, SSRLayer]],
        *,
        lambda_reg: float,
        model_strengths: list[float],
    ) -> None:
        if len(layers_per_task) < 2:
            raise ValueError("SSR-Merge requires at least two LoRAs.")
        if lambda_reg < 0:
            raise ValueError("lambda_reg must be non-negative.")
        if len(model_strengths) != len(layers_per_task):
            raise ValueError("model_strengths length must match the number of LoRAs.")

        self.layers_per_task = [dict(task) for task in layers_per_task]
        self.lambda_reg = float(lambda_reg)
        self.model_strengths = [float(x) for x in model_strengths]
        self.warnings: list[str] = []
        self._current_task: int | None = None
        self.plan: dict[str, dict] = {}
        self._build_plan()

    def _build_plan(self) -> None:
        all_targets = sorted(set().union(*(set(task) for task in self.layers_per_task)))
        for target_base in all_targets:
            active: list[int] = []
            layers: list[SSRLayer] = []
            for index, task in enumerate(self.layers_per_task):
                layer = task.get(target_base)
                if layer is None:
                    continue
                if layer.down.ndim not in (2, 4) or layer.up.ndim not in (2, 4):
                    self.warnings.append(f"Skipped {target_base}: only linear and 1x1-conv LoRA are supported.")
                    continue
                if layer.up.ndim == 4 and tuple(layer.up.shape[2:]) != (1, 1):
                    self.warnings.append(f"Skipped {target_base}: spatial lora_up kernels are unsupported.")
                    continue
                if layer.down.ndim == 4 and tuple(layer.down.shape[2:]) != (1, 1):
                    self.warnings.append(f"Skipped {target_base}: spatial lora_down kernels are unsupported.")
                    continue
                if layer.up.shape[1] != layer.down.shape[0]:
                    self.warnings.append(f"Skipped {target_base}: incompatible LoRA rank.")
                    continue
                active.append(index)
                layers.append(layer)

            if not layers:
                continue

            if len(layers) == 1:
                self.plan[target_base] = {
                    "passthrough": True,
                    "active": active,
                    "layers": layers,
                }
                continue

            down = torch.cat([layer.down_2d() for layer in layers], dim=0)
            rank = int(down.shape[0])
            self.plan[target_base] = {
                "passthrough": False,
                "active": active,
                "layers": layers,
                "down": down,
                "rank_dims": [layer.rank for layer in layers],
                "g": torch.zeros((rank, rank), dtype=torch.float32),
                "q": torch.zeros((rank, rank), dtype=torch.float32),
                "count": 0,
            }

        if not self.plan:
            raise ValueError("No compatible LoRA layers were found for SSR-Merge.")

    def module_targets(self) -> set[str]:
        """Return module names that need forward hooks."""
        return {target for target, entry in self.plan.items() if not entry["passthrough"]}

    def activation_summary(self) -> str:
        """Return a compact summary of collected calibration statistics."""
        routed = [entry for entry in self.plan.values() if not entry["passthrough"]]
        active = sum(1 for entry in routed if int(entry["count"]) > 0)
        samples = sum(int(entry["count"]) for entry in routed)
        return f"active_layers={active}/{len(routed)}, activation_samples={samples}"

    def set_active_task(self, task_index: int) -> None:
        if task_index < 0 or task_index >= len(self.layers_per_task):
            raise IndexError(f"task_index out of range: {task_index}")
        self._current_task = int(task_index)

    def accumulate(self, target_base: str, activation: torch.Tensor) -> None:
        """Accumulate statistics from one hooked module input."""
        if self._current_task is None:
            return
        entry = self.plan.get(target_base)
        if entry is None or entry["passthrough"]:
            return
        if self._current_task not in entry["active"]:
            return

        down = entry["down"]
        in_dim = int(down.shape[1])
        x = _activation_to_matrix(activation, in_dim)
        if x is None:
            return
        local_task = entry["active"].index(self._current_task)
        z = down.to(device=x.device, dtype=torch.float32).matmul(x)
        entry["g"] += z.matmul(z.t()).cpu()

        start = sum(entry["rank_dims"][:local_task])
        end = start + entry["rank_dims"][local_task]
        entry["q"][start:end, :] += z[start:end, :].matmul(z.t()).cpu()
        entry["count"] += int(x.shape[1])

    def solve(self) -> dict[str, torch.Tensor]:
        """Return a flat ComfyUI-loadable LoRA state dict."""
        LOGGER.info("[SSR-Merge] Solving SSR routers: %s", self.activation_summary())
        output: dict[str, torch.Tensor] = {}
        no_activation_layers: list[str] = []
        solved_layers = 0
        passthrough_layers = 0
        for target_base, entry in self.plan.items():
            layers: list[SSRLayer] = entry["layers"]
            template = layers[0]

            if entry["passthrough"]:
                passthrough_layers += 1
                task_index = entry["active"][0]
                layer = layers[0]
                up = layer.scaled_up_2d(self.model_strengths[task_index])
                down = layer.down_2d()
                rank = layer.rank
            else:
                down = entry["down"]
                up = torch.cat(
                    [
                        layer.scaled_up_2d(self.model_strengths[task_index])
                        for task_index, layer in zip(entry["active"], layers)
                    ],
                    dim=1,
                )
                rank = int(down.shape[0])
                count = int(entry["count"])
                if count == 0:
                    router = torch.eye(rank, dtype=torch.float32)
                    no_activation_layers.append(target_base)
                else:
                    g = entry["g"] / float(count)
                    q = entry["q"] / float(count)
                    g = g + torch.eye(rank, dtype=torch.float32) * self.lambda_reg
                    try:
                        router = torch.linalg.solve(g.to(torch.float64), q.t().to(torch.float64)).t().to(torch.float32)
                        solved_layers += 1
                    except RuntimeError as error:
                        router = torch.eye(rank, dtype=torch.float32)
                        self.warnings.append(f"{target_base}: router solve failed; used identity router ({error}).")
                up = up.matmul(router)

            up_shape = (template.up.shape[0], rank, *template.up.shape[2:])
            down_shape = (rank, *template.down.shape[1:])
            output[f"{template.lora_base}.lora_up.weight"] = up.reshape(up_shape).contiguous()
            output[f"{template.lora_base}.lora_down.weight"] = down.reshape(down_shape).contiguous()
            output[f"{template.lora_base}.alpha"] = torch.tensor(float(rank), dtype=torch.float32)

        if no_activation_layers:
            preview = ", ".join(no_activation_layers[:8])
            suffix = "" if len(no_activation_layers) <= 8 else f", ... +{len(no_activation_layers) - 8} more"
            self.warnings.append(
                f"{len(no_activation_layers)} layer(s) received no calibration activation; "
                f"used identity router for: {preview}{suffix}."
            )
        LOGGER.info(
            "[SSR-Merge] Router solve complete: emitted_layers=%d, solved_layers=%d, "
            "passthrough_layers=%d, identity_fallback_layers=%d",
            len(output) // 3,
            solved_layers,
            passthrough_layers,
            len(no_activation_layers),
        )
        return output


def _activation_to_matrix(activation: torch.Tensor, in_dim: int) -> torch.Tensor | None:
    """Convert Linear/Conv1x1 module input to ``(in_dim, tokens)``."""
    x = activation.detach()
    if x.ndim >= 2 and x.shape[-1] == in_dim:
        return x.reshape(-1, in_dim).t().to(dtype=torch.float32)
    if x.ndim == 4 and x.shape[1] == in_dim:
        return x.permute(1, 0, 2, 3).reshape(in_dim, -1).to(dtype=torch.float32)
    return None
