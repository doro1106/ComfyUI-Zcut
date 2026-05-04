try:
    from .install import ensure_requirements

    ensure_requirements()
except Exception as exc:
    raise RuntimeError(
        "[Zcut] Dependency auto-install failed. Install dependencies in this ComfyUI Python environment with "
        "`pip install -r custom_nodes/Zcut/requirements.txt` and restart ComfyUI."
    ) from exc

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
