"""ComfyUI V3 nodes for SSR-Merge calibration and loading."""

from __future__ import annotations

import gc
import logging
import time

import torch

import comfy.lora
import comfy.lora_convert
import comfy.model_management
import comfy.sample
import comfy.sd
import comfy.samplers
import comfy.utils
import folder_paths

from .comfy_api_compat import io, resolve_type
from .ssr_merge import SSRLayer, SSRMergedLoRA, SSRStats


LOGGER = logging.getLogger(__name__)
_MODEL = resolve_type("MODEL")
_CLIP = resolve_type("CLIP")
_SSR_MERGE = resolve_type("SSR_MERGE")
_FAST_CALIBRATION_STEPS = 1
_FAST_CALIBRATION_CFG = 1.0


class SSRMergeCalibration(io.ComfyNode):
    """Run one-shot SSR calibration and emit a merged LoRA payload."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SSRMergeCalibration",
            display_name="SSR-Merge Calibration",
            category="loaders/LoRA",
            description=(
                "Run one-step SSR-Merge calibration and emit a merged LoRA payload."
            ),
            search_aliases=["ssr merge", "subspace signal routing", "lora calibration"],
            inputs=[
                _MODEL.Input("model"),
                _CLIP.Input("clip"),
                io.Combo.Input("lora_1", options=folder_paths.get_filename_list("loras")),
                io.Combo.Input("lora_2", options=folder_paths.get_filename_list("loras")),
                io.String.Input("prompt_1", default="", multiline=True, dynamic_prompts=True),
                io.String.Input("prompt_2", default="", multiline=True, dynamic_prompts=True),
                io.String.Input("negative_prompt", default="", multiline=True, dynamic_prompts=True),
                io.Float.Input("lora_1_strength_model", default=1.0, min=-20.0, max=20.0, step=0.05),
                io.Float.Input("lora_2_strength_model", default=1.0, min=-20.0, max=20.0, step=0.05),
                io.Float.Input(
                    "lambda_reg",
                    default=0.0001,
                    min=0.0,
                    max=1.0,
                    step=0.0001,
                    tooltip="Ridge regularization for the SSR correlation matrix.",
                ),
                io.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Int.Input("width", default=1024, min=64, max=8192, step=8),
                io.Int.Input("height", default=1024, min=64, max=8192, step=8),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS),
            ],
            outputs=[
                _SSR_MERGE.Output(display_name="ssr merge"),
                io.String.Output(display_name="calibration info"),
            ],
        )

    @classmethod
    def validate_inputs(cls, lora_1, lora_2, **_kwargs):
        available = set(folder_paths.get_filename_list("loras"))
        for label, filename in (("lora_1", lora_1), ("lora_2", lora_2)):
            if filename not in available:
                return f"Unknown {label}: {filename}"
        if lora_1 == lora_2:
            return "SSR-Merge requires two different LoRA files."
        return True

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        """Ensure calibration is evaluated for each queue run."""
        return time.time_ns()

    @classmethod
    def execute(
        cls,
        model,
        clip,
        lora_1,
        lora_2,
        prompt_1,
        prompt_2,
        negative_prompt,
        lora_1_strength_model,
        lora_2_strength_model,
        lambda_reg,
        seed,
        width,
        height,
        sampler_name,
        scheduler,
    ) -> io.NodeOutput:
        if clip is None:
            raise ValueError("SSR-Merge Calibration requires a CLIP input to encode calibration prompts.")
        paths = [
            folder_paths.get_full_path_or_raise("loras", lora_1),
            folder_paths.get_full_path_or_raise("loras", lora_2),
        ]
        prompts = [str(prompt_1), str(prompt_2)]
        strengths = [float(lora_1_strength_model), float(lora_2_strength_model)]
        LOGGER.info(
            "[SSR-Merge] Requested calibration: loras=(%s, %s), steps=%d, size=%dx%d, "
            "cfg=%g, sampler=%s, scheduler=%s",
            lora_1,
            lora_2,
            _FAST_CALIBRATION_STEPS,
            int(width),
            int(height),
            _FAST_CALIBRATION_CFG,
            sampler_name,
            scheduler,
        )
        LOGGER.info("[SSR-Merge] Loading LoRAs for calibration.")
        raw_loras = [comfy.utils.load_torch_file(path, safe_load=True) for path in paths]
        layers_per_task = [_extract_model_lora_layers(raw, model) for raw in raw_loras]
        LOGGER.info(
            "[SSR-Merge] Extracted model-side LoRA layers: %s=%d, %s=%d",
            lora_1,
            len(layers_per_task[0]),
            lora_2,
            len(layers_per_task[1]),
        )
        stats = SSRStats(layers_per_task, lambda_reg=float(lambda_reg), model_strengths=strengths)
        LOGGER.info(
            "[SSR-Merge] Merge plan: total_layers=%d, routed_layers=%d, passthrough_layers=%d",
            len(stats.plan),
            len(stats.module_targets()),
            len(stats.plan) - len(stats.module_targets()),
        )
        hooked_count = 0
        for index, (raw, prompt, lora_name) in enumerate(zip(raw_loras, prompts, (lora_1, lora_2))):
            hooked_count += _calibrate_one_task(
                model=model,
                raw_lora=raw,
                clip=clip,
                stats=stats,
                task_index=index,
                task_label=lora_name,
                prompt=prompt,
                negative_prompt=str(negative_prompt),
                seed=int(seed),
                width=int(width),
                height=int(height),
                steps=_FAST_CALIBRATION_STEPS,
                cfg=_FAST_CALIBRATION_CFG,
                sampler_name=sampler_name,
                scheduler=scheduler,
            )

        LOGGER.info("[SSR-Merge] Solving routers after calibration. %s", stats.activation_summary())
        merged = stats.solve()
        warnings = stats.warnings
        for warning in warnings:
            LOGGER.warning("[SSR-Merge] %s", warning)
        info = (
            f"SSR-Merge: loras=({lora_1}, {lora_2}); layers={len(merged) // 3}; "
            f"hooks={hooked_count}; lambda={float(lambda_reg):g}; seed={int(seed)}"
        )
        if warnings:
            info += f"; warnings={len(warnings)}"
        result = SSRMergedLoRA(state_dict=merged, cache_key=f"calibration:{time.time_ns()}", info=info, warnings=warnings)
        LOGGER.info("[SSR-Merge] Calibration complete: %s", info)
        return io.NodeOutput(result, info)


class SSRMergeLoader(io.ComfyNode):
    """Apply an SSR-Merge calibration result as a normal LoRA."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SSRMergeLoader",
            display_name="SSR-Merge Loader",
            category="loaders/LoRA",
            description="Apply the merged LoRA emitted by SSR-Merge Calibration.",
            search_aliases=["ssr merge loader", "lora loader"],
            inputs=[
                _MODEL.Input("model"),
                _CLIP.Input("clip"),
                _SSR_MERGE.Input("ssr_merge"),
                io.Float.Input("strength_model", default=1.0, min=-20.0, max=20.0, step=0.05),
            ],
            outputs=[
                _MODEL.Output(display_name="model"),
                _CLIP.Output(display_name="clip"),
                io.String.Output(display_name="merge info"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, ssr_merge: SSRMergedLoRA, strength_model) -> io.NodeOutput:
        started = time.perf_counter()
        patch_started = time.perf_counter()
        key_map: dict[str, str] = comfy.lora.model_lora_keys_unet(model.model, {})
        converted = comfy.lora_convert.convert_lora(ssr_merge.state_dict)
        loaded = comfy.lora.load_lora(converted, key_map)
        patch_seconds = time.perf_counter() - patch_started

        apply_started = time.perf_counter()
        patched_model = model.clone()
        applied = set(patched_model.add_patches(loaded, float(strength_model)))
        for key in loaded:
            if key not in applied:
                LOGGER.warning("[SSR-Merge Loader] NOT LOADED %s", key)
        apply_seconds = time.perf_counter() - apply_started
        total_seconds = time.perf_counter() - started

        LOGGER.info(
            "[SSR-Merge Loader] Applied patch dict: key_map=%d, patches=%d, "
            "patch_build=%.3fs, apply=%.3fs, total=%.3fs, strength=%g",
            len(key_map),
            len(loaded),
            patch_seconds,
            apply_seconds,
            total_seconds,
            float(strength_model),
        )
        info = (
            f"{ssr_merge.info}; loader_strength={float(strength_model):g}; "
            f"loader_time={total_seconds:.3f}s"
        )
        clip = clip
        model = patched_model
        return io.NodeOutput(model, clip, info)

def _extract_model_lora_layers(raw_lora: dict[str, torch.Tensor], model) -> dict[str, SSRLayer]:
    key_map: dict[str, str] = comfy.lora.model_lora_keys_unet(model.model, {})
    converted = comfy.lora_convert.convert_lora(raw_lora)
    loaded = comfy.lora.load_lora(converted, key_map, log_missing=False)
    reverse_targets = _reverse_key_map(key_map)
    layers: dict[str, SSRLayer] = {}
    for target_weight, adapter in loaded.items():
        if getattr(adapter, "name", None) != "lora":
            continue
        weights = getattr(adapter, "weights", None)
        if not weights or len(weights) < 3:
            continue
        up, down, alpha = weights[:3]
        if not isinstance(up, torch.Tensor) or not isinstance(down, torch.Tensor):
            continue
        if not target_weight.endswith(".weight"):
            continue
        lora_base = reverse_targets.get(target_weight)
        if lora_base is None:
            continue
        alpha_value = float(alpha) if alpha is not None else float(down.shape[0])
        target_base = target_weight[: -len(".weight")]
        layers[target_base] = SSRLayer(
            lora_base=lora_base,
            target_base=target_base,
            down=down,
            up=up,
            alpha=alpha_value,
        )
    if not layers:
        raise ValueError("No conventional model-side LoRA layers were found for SSR-Merge.")
    return layers


def _reverse_key_map(key_map: Mapping[str, str]) -> dict[str, str]:
    reverse: dict[str, str] = {}
    # Prefer generic diffusion_model keys because they are stable for ComfyUI's loader.
    for lora_base, target in sorted(key_map.items(), key=lambda item: 0 if item[0].startswith("diffusion_model.") else 1):
        reverse.setdefault(target, lora_base)
    return reverse


def _calibrate_one_task(
    *,
    model,
    raw_lora: dict[str, torch.Tensor],
    clip,
    stats: SSRStats,
    task_index: int,
    task_label: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
) -> int:
    started = time.perf_counter()
    LOGGER.info(
        "[SSR-Merge] Task %d calibration start: lora=%s, steps=%d, prompt_chars=%d",
        task_index + 1,
        task_label,
        steps,
        len(prompt),
    )
    calibration_model, _ = comfy.sd.load_lora_for_models(model, None, raw_lora, 1.0, 0.0)
    comfy.model_management.load_model_gpu(calibration_model)
    handles: list[torch.utils.hooks.RemovableHandle] = []
    hooked = 0
    named_modules = dict(calibration_model.model.named_modules())
    for target_base in stats.module_targets():
        module = _find_module(named_modules, target_base)
        if module is None:
            stats.warnings.append(f"{target_base}: no matching module found during calibration.")
            continue
        handle = module.register_forward_hook(_make_hook(stats, target_base))
        handles.append(handle)
        hooked += 1
    LOGGER.info("[SSR-Merge] Task %d hooks registered: %d", task_index + 1, hooked)

    stats.set_active_task(task_index)
    try:
        positive = _encode_prompt(clip, prompt)
        negative = _encode_prompt(clip, negative_prompt)
        latent = torch.zeros((1, 4, max(1, height // 8), max(1, width // 8)), dtype=torch.float32)
        latent = comfy.sample.fix_empty_latent_channels(calibration_model, latent, 8)
        noise = comfy.sample.prepare_noise(latent, seed + task_index)
        with torch.no_grad():
            comfy.sample.sample(
                calibration_model,
                noise,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent,
                denoise=1.0,
                disable_pbar=True,
                seed=seed + task_index,
            )
    finally:
        for handle in handles:
            handle.remove()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    LOGGER.info(
        "[SSR-Merge] Task %d calibration done in %.2fs. %s",
        task_index + 1,
        time.perf_counter() - started,
        stats.activation_summary(),
    )
    return hooked


def _make_hook(stats: SSRStats, target_base: str):
    def hook(_module, inputs, _outputs):
        if not inputs:
            return
        activation = inputs[0]
        if isinstance(activation, torch.Tensor):
            with torch.no_grad():
                stats.accumulate(target_base, activation)

    return hook


def _find_module(named_modules: Mapping[str, torch.nn.Module], target_base: str):
    candidates = [target_base]
    if target_base.startswith("diffusion_model."):
        candidates.append(target_base[len("diffusion_model.") :])
    else:
        candidates.append("diffusion_model." + target_base)
    for name in candidates:
        module = named_modules.get(name)
        if module is not None:
            return module
    return None


def _encode_prompt(clip, prompt: str):
    tokens = clip.tokenize(prompt)
    return clip.encode_from_tokens_scheduled(tokens)
