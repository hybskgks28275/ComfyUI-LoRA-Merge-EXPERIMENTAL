"""ComfyUI node that applies a fused NP-LoRA at runtime."""

from __future__ import annotations

import logging

import folder_paths
import comfy.sd
import comfy.utils

from .comfy_api_compat import io, resolve_type
from .np_lora import fuse_lora_state_dicts


LOGGER = logging.getLogger(__name__)
_MODEL = resolve_type("MODEL")
_CLIP = resolve_type("CLIP")


class NPLoRALoader(io.ComfyNode):
    """Fuse a subject/content LoRA with a style LoRA using NP-LoRA."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NPLoRALoader",
            display_name="NP-LoRA Loader (Subject + Style)",
            category="loaders/LoRA",
            description=(
                "Fuse a subject/content LoRA with a style LoRA using NP-LoRA's "
                "asymmetric null-space projection."
            ),
            search_aliases=["np lora", "lora merge", "style lora", "lora loader"],
            inputs=[
                _MODEL.Input("model"),
                _CLIP.Input("clip"),
                io.Combo.Input(
                    "content_lora",
                    options=folder_paths.get_filename_list("loras"),
                    tooltip="Subject, character, or content LoRA to preserve.",
                ),
                io.Combo.Input(
                    "style_lora",
                    options=folder_paths.get_filename_list("loras"),
                    tooltip="Style LoRA whose right-singular subspace is protected.",
                ),
                io.Float.Input(
                    "mu", default=0.5, min=0.0, max=100.0, step=0.05,
                    tooltip="0 = direct merge; higher values protect style more strongly.",
                ),
                io.Float.Input("content_strength_model", default=1.0, min=-20.0, max=20.0, step=0.05),
                io.Float.Input("style_strength_model", default=1.0, min=-20.0, max=20.0, step=0.05),
                io.Float.Input("content_strength_clip", default=1.0, min=-20.0, max=20.0, step=0.05),
                io.Float.Input("style_strength_clip", default=1.0, min=-20.0, max=20.0, step=0.05),
            ],
            outputs=[
                _MODEL.Output(display_name="model"),
                _CLIP.Output(display_name="clip"),
                io.String.Output(display_name="merge info"),
            ],
        )

    @classmethod
    def validate_inputs(cls, content_lora, style_lora, **_kwargs):
        available = set(folder_paths.get_filename_list("loras"))
        for label, filename in (("content_lora", content_lora), ("style_lora", style_lora)):
            if filename not in available:
                return f"Unknown {label}: {filename}"
        return True

    @classmethod
    def execute(
        cls, model, clip, content_lora, style_lora, mu,
        content_strength_model, style_strength_model,
        content_strength_clip, style_strength_clip,
    ) -> io.NodeOutput:
        content_path = folder_paths.get_full_path_or_raise("loras", content_lora)
        style_path = folder_paths.get_full_path_or_raise("loras", style_lora)
        content = comfy.utils.load_torch_file(content_path, safe_load=True)
        style = comfy.utils.load_torch_file(style_path, safe_load=True)
        merged, warnings = fuse_lora_state_dicts(
            content, style, mu=float(mu),
            content_model_strength=float(content_strength_model),
            style_model_strength=float(style_strength_model),
            content_clip_strength=float(content_strength_clip),
            style_clip_strength=float(style_strength_clip),
        )
        for warning in warnings:
            LOGGER.warning("[NP-LoRA] %s", warning)
        model, clip = comfy.sd.load_lora_for_models(model, clip, merged, 1.0, 1.0)
        info = f"NP-LoRA: content={content_lora}, style={style_lora}, mu={float(mu):g}; adapters={len(merged) // 3}"
        if warnings:
            info += f"; skipped={len(warnings)} (see console)"
        return io.NodeOutput(model, clip, info)
