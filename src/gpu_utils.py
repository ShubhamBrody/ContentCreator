"""
ContentCreator - GPU & VRAM Utilities

Handles model loading/unloading and GPU memory management.
"""

import gc
import torch
from rich.console import Console

console = Console()


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def free_vram() -> None:
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def log_vram(label: str = "") -> None:
    """Log current VRAM usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        console.print(
            f"[dim]VRAM {label}: "
            f"{allocated:.1f}GB allocated / "
            f"{reserved:.1f}GB reserved / "
            f"{total:.1f}GB total[/dim]"
        )


def unload_model(model: object) -> None:
    """Move a model to CPU and free VRAM."""
    if hasattr(model, "to"):
        model.to("cpu")  # type: ignore[union-attr]
    del model
    free_vram()
