# ComfyUI-LoRA-Merge-EXPERIMENTAL

[日本語README](README_ja.md)

Custom nodes for ComfyUI that support [NP-LoRA](https://arxiv.org/html/2511.11051v3) and [SSR-Merge](https://arxiv.org/abs/2606.10617) based LoRA merging.

This extension currently provides:

- **NP-LoRA Loader (Subject + Style)**
  - Fuses a subject/character LoRA with a style LoRA using NP-LoRA's asymmetric null-space projection.
- **SSR-Merge Calibration**
  - Runs a lightweight calibration pass for SSR-Merge and builds a merged LoRA from internal activation statistics.
- **SSR-Merge Loader**
  - Applies the result emitted by `SSR-Merge Calibration` to a `MODEL`.

Internally, the extension uses ComfyUI's current custom-node API.

## Installation

Place this directory under ComfyUI's `custom_nodes` directory. From the current development location:

```powershell
Copy-Item -Recurse D:\tools\dev\ComfyUI-LoRA-Merge-EXPERIMENTAL D:\tools\ComfyUI\ComfyUI\custom_nodes\
```

Restart ComfyUI, then add the nodes from `loaders/LoRA`.

## NP-LoRA Loader

The NP-LoRA loader follows Eq. 12 of the NP-LoRA paper. It attenuates the content LoRA component that overlaps with the right-singular-vector subspace of the style LoRA.

`D_merged = D_style + D_content (I - mu/(1+mu) V V^T)`

Basic usage:

1. Set `content_lora` to the subject, person, or character LoRA you want to preserve.
2. Set `style_lora` to the style LoRA.
3. Start with `mu = 0.5`.
4. Increase `mu` if the style is too weak. Decrease it if subject details are lost. `mu = 0` is equivalent to a direct additive merge.

## SSR-Merge

SSR-Merge performs a short calibration inference before merging. It collects internal activation statistics, solves an analytic router, and absorbs that router into the up-projection so the result can be applied as a normal LoRA.

This extension exposes SSR-Merge as two nodes:

1. **SSR-Merge Calibration**
   - `model` / `clip`
   - `lora_1` / `lora_2`
   - `prompt_1` / `prompt_2`
   - `negative_prompt`
   - seed, resolution, `lambda_reg`
   - sampler / scheduler

2. **SSR-Merge Loader**
   - Takes the `ssr merge` output from `SSR-Merge Calibration` and applies it to `MODEL`.

### Calibration behavior

`SSR-Merge Calibration` runs a fresh calibration pass when the node is executed.

The calibration inference uses fixed lightweight values for steps and CFG:

- steps: `1`
- cfg: `1.0`

The sampler and scheduler are user-selectable in the node UI.

SSR-Merge logs normal progress at `INFO` level. The logs include:

- extracted LoRA layer counts
- routed and passthrough layer counts
- calibration start and end for each LoRA
- registered hook count
- activation statistics coverage
- router solve summary
- Loader timing

### Current scope

- Model-side U-Net/DiT LoRA
- Linear and 1x1 convolution LoRA
- SSR merge for two LoRAs

CLIP-side LoRA, spatial LoCon kernels, DoRA, LoHa, and LoKr are intentionally excluded from SSR-Merge for safety.

## References

- [NP-LoRA: Null Space Projection for Subject-Style LoRA Fusion](https://arxiv.org/html/2511.11051v3)
- [SSR-Merge: Subspace Signal Routing for Training-Free LoRA Merging in Diffusion Models](https://arxiv.org/abs/2606.10617)
- [SSR-Merge official implementation](https://github.com/nagara214/SSR-Merge)
