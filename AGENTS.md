# AGENTS.md

## Project overview

This repository contains experimental ComfyUI custom nodes for LoRA merging.

Supported merge methods:

- NP-LoRA for subject/style LoRA fusion.
- SSR-Merge for calibration-based LoRA merging.

The project uses ComfyUI's current V3 custom-node API through `comfy_entrypoint`, `ComfyExtension`, and `io.ComfyNode`.

## Important files

- `__init__.py` registers the ComfyUI extension and node classes.
- `np_lora.py` contains NP-LoRA tensor merge logic.
- `np_lora_loader.py` exposes the NP-LoRA ComfyUI node.
- `ssr_merge.py` contains SSR-Merge tensor/statistics logic.
- `ssr_nodes.py` exposes SSR-Merge calibration and loader nodes.
- `comfy_api_compat.py` handles ComfyUI API import compatibility.
- `tests/` contains lightweight regression tests that do not require launching ComfyUI.

## Validation commands

Run these from the repository root.

```powershell
@'
import pathlib, runpy, sys
root = pathlib.Path.cwd()
sys.path.insert(0, str(root))
np_mod = runpy.run_path(str(root / 'tests' / 'test_np_lora.py'))
ssr_mod = runpy.run_path(str(root / 'tests' / 'test_ssr_merge.py'))
for mod, names in [
    (np_mod, ['test_mu_zero_is_direct_merge', 'test_higher_mu_suppresses_style_subspace_component']),
    (ssr_mod, ['test_ssr_solve_emits_standard_lora_keys', 'test_ssr_identity_fallback_when_no_activation']),
]:
    for name in names:
        mod[name]()
print('Core tests: PASS')
'@ | D:\tools\ComfyUI\venv\Scripts\python.exe -
```

```powershell
@'
import asyncio, importlib.util, pathlib, sys
package = pathlib.Path.cwd()
sys.path.insert(0, r'D:\tools\ComfyUI\ComfyUI')
spec = importlib.util.spec_from_file_location(
    'lora_merge_experimental_custom',
    package / '__init__.py',
    submodule_search_locations=[str(package)],
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
extension = asyncio.run(module.comfy_entrypoint())
nodes = asyncio.run(extension.get_node_list())
ids = [node.GET_SCHEMA().node_id for node in nodes]
assert ids == ['NPLoRALoader', 'SSRMergeCalibration', 'SSRMergeLoader'], ids
print('V3 schema: PASS', ids)
'@ | D:\tools\ComfyUI\venv\Scripts\python.exe -
```

```powershell
D:\tools\ComfyUI\venv\Scripts\python.exe -m compileall -q .
```

After running `compileall`, remove generated `__pycache__` directories before committing.

```powershell
Get-ChildItem -Path . -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
```

## Development notes

- Keep node code in the ComfyUI V3 style. Do not reintroduce legacy `NODE_CLASS_MAPPINGS`, `INPUT_TYPES`, `RETURN_TYPES`, or `FUNCTION` declarations.
- SSR-Merge calibration currently uses `steps=1` and `cfg=1.0`; sampler and scheduler are user-selectable.
- SSR-Merge calibration and loader behavior should stay explicit and easy to reason about.
- Avoid adding rank compression unless the user explicitly requests it, because it can affect output quality.
- Do not add support for LoHa, LoKr, DoRA, or spatial LoCon kernels unless proper handling is implemented and tested.

## Documentation

- Main README: `README.md`
- Japanese README: `README_ja.md`
- License: Apache License 2.0
