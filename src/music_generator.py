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

    def _load_model(self) -> None:
        """Load MusicGen model onto GPU."""
        if self._model is not None:
            return

        if self.engine == "none":
            console.print("[dim]Music generation disabled.[/dim]")
            return

        console.print("[cyan]Loading MusicGen model...[/cyan]")
        log_vram("before MusicGen load")

        from transformers import AutoProcessor, MusicgenForConditionalGeneration  # type: ignore[import-untyped]

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = MusicgenForConditionalGeneration.from_pretrained(self.model_id)

        dtype = torch.float16 if self.config.half_precision else torch.float32
        self._model = self._model.to(dtype).to(self.config.device)

        log_vram("after MusicGen load")
        console.print("[green]✓ MusicGen model loaded[/green]")

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
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        # Calculate max_new_tokens from duration
        # MusicGen generates at ~50 tokens/second (depends on model)
        sample_rate = self._model.config.audio_encoder.sampling_rate
        tokens_per_second = 50  # approximate
        max_new_tokens = int(duration * tokens_per_second)

        with torch.no_grad():
            audio_values = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        # Save as WAV
        audio_data = audio_values[0, 0].cpu().numpy()
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
