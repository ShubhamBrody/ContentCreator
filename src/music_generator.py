"""
ContentCreator - Music Generator

Generates background music using Meta's MusicGen (local, free).
Supports text-to-music with configurable duration and model sizes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import torch
from rich.console import Console

from src.config import Config
from src.gpu_utils import free_vram, log_vram, unload_model

console = Console()


class MusicGenerator:
    """Generates background music using MusicGen (Meta)."""

    def __init__(self, config: Config):
        self.config = config
        self.engine = config.music.get("engine", "musicgen")
        self.model_id = config.music.get("model", "facebook/musicgen-small")
        self.duration = config.music.get("duration", 30)
        self.volume = config.music.get("volume", 0.3)
        self._model: Any = None
        self._processor: Any = None

    def _load_model(self, device: Optional[str] = None) -> None:
        """Load MusicGen model.

        Args:
            device: Override device ('cuda' or 'cpu').  When *None* the
                    config device is used.
        """
        if self._model is not None:
            return

        if self.engine == "none":
            console.print("[dim]Music generation disabled.[/dim]")
            return

        target = device or str(self.config.device)

        # Aggressively clear CUDA state left by previous stages (SDXL, video)
        free_vram()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.ipc_collect()

        console.print(f"[cyan]Loading MusicGen model on {target}...[/cyan]")
        log_vram("before MusicGen load")

        from transformers import AutoProcessor, MusicgenForConditionalGeneration  # type: ignore[import-untyped]

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = MusicgenForConditionalGeneration.from_pretrained(self.model_id)

        # MusicGen-small is only ~300 MB — always use float32 to avoid
        # cuBLAS float16 numerical instability & CUDA indexing corruption.
        self._model = self._model.to(torch.float32).to(target)
        self._active_device = target  # remember for tokenisation

        log_vram("after MusicGen load")
        console.print(f"[green]✓ MusicGen model loaded on {target}[/green]")

    def _force_unload(self) -> None:
        """Drop model references WITHOUT touching CUDA.

        Use this when the CUDA context is poisoned and ``model.to('cpu')``
        would itself throw.
        """
        self._model = None
        self._processor = None
        import gc
        gc.collect()

    def unload(self) -> None:
        """Unload MusicGen model and free VRAM."""
        if self._model is not None:
            console.print("[dim]Unloading MusicGen model...[/dim]")
            unload_model(self._model)
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        free_vram()
        log_vram("after MusicGen unload")

    async def generate_music(
        self,
        prompt: str,
        output_path: str,
        duration: Optional[float] = None,
    ) -> Optional[str]:
        """
        Generate background music from a text prompt.

        Args:
            prompt: Music style description (e.g., "upbeat electronic background music")
            output_path: Where to save the generated audio
            duration: Duration in seconds (overrides config)

        Returns:
            Path to generated music file, or None if disabled
        """
        if self.engine == "none":
            console.print("[dim]Music generation skipped (disabled).[/dim]")
            return None

        self._load_model()

        if self._model is None or self._processor is None:
            raise RuntimeError("MusicGen model failed to load.")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        duration = duration or self.duration

        console.print(
            f"[cyan]Generating {duration}s of music: '{prompt}'[/cyan]"
        )

        import scipy.io.wavfile  # type: ignore[import-untyped]

        # Tokenize the prompt
        inputs = self._processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        )
        dev = getattr(self, "_active_device", str(self.config.device))
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        # Calculate max_new_tokens from duration
        # MusicGen generates at ~50 tokens/second (depends on model)
        sample_rate = self._model.config.audio_encoder.sampling_rate
        tokens_per_second = 50  # approximate
        max_new_tokens = int(duration * tokens_per_second)

        audio_values = None
        for attempt in range(2):
            try:
                with torch.no_grad():
                    audio_values = self._model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                    )
                break  # success
            except RuntimeError as exc:
                if attempt == 0 and "CUDA" in str(exc):
                    console.print(
                        "[yellow]⚠ CUDA error during MusicGen — "
                        "falling back to CPU...[/yellow]"
                    )
                    # CUDA context is poisoned.  Do NOT touch the GPU.
                    self._force_unload()
                    # Reload entirely on CPU (MusicGen-small is ~300 MB,
                    # runs fine on CPU).
                    self._load_model(device="cpu")
                    # Re-tokenize for CPU
                    inputs = self._processor(
                        text=[prompt],
                        padding=True,
                        return_tensors="pt",
                    )
                    # inputs stay on CPU — no .to(device) needed
                    continue
                raise  # non-CUDA error or second attempt failed

        # Save as WAV
        audio_data = audio_values[0, 0].cpu().float().numpy()
        scipy.io.wavfile.write(
            output_path,
            rate=sample_rate,
            data=audio_data,
        )

        console.print(f"[green]✓ Music saved to {output_path}[/green]")

        if self.config.auto_unload:
            self.unload()

        return output_path

    def generate_single(
        self,
        prompt: str,
        output_path: str,
        duration: Optional[float] = None,
    ) -> Optional[str]:
        """Synchronous convenience wrapper."""
        import asyncio

        return asyncio.run(self.generate_music(prompt, output_path, duration))
