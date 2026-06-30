"""Imports for the current ComfyUI V3 custom-node API."""

from __future__ import annotations

try:
    from comfy_api.latest import ComfyExtension, io
except ImportError:  # pragma: no cover - supports pre-latest V3 builds
    from comfy_api import ComfyExtension, io


def resolve_type(name: str):
    """Resolve built-in V3 socket types with a Custom fallback."""
    direct = getattr(io, name, None)
    if direct is not None:
        return direct
    camel = name[:1].upper() + name[1:].lower()
    return getattr(io, camel, None) or io.Custom(name)


__all__ = ["ComfyExtension", "io", "resolve_type"]
