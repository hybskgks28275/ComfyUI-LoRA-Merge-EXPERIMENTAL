"""LoRA merge custom nodes for ComfyUI's V3 extension API."""

from typing_extensions import override

from .comfy_api_compat import ComfyExtension, io
from .np_lora_loader import NPLoRALoader
from .ssr_nodes import SSRMergeCalibration, SSRMergeLoader


class LoRAMergeToolsExtension(ComfyExtension):
    """Register LoRA merge nodes through the V3 extension entrypoint."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [NPLoRALoader, SSRMergeCalibration, SSRMergeLoader]


async def comfy_entrypoint() -> ComfyExtension:
    return LoRAMergeToolsExtension()


__all__ = ["comfy_entrypoint"]
