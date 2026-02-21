"""
ContentCreator - Music Generator

Generates background music using Meta's MusicGen (local, free).
Supports text-to-music with configurable duration and model sizes.
For long videos (>30s) generates in chunks and cross-fades them.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import torch
from rich.console import Console

from src.config import Config
from src.gpu_utils import free_vram, log_vram, unload_model

console = Console()

# Maximum seconds per single MusicGen generation call.
# Keeps KV-cache small and avoids quality degradation / OOM on long videos.
_MAX_CHUNK_SECS = 30.0
_CROSSFADE_SECS = 1.0  # overlap between adjacent chunks


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

        For durations > 30 s the audio is generated in chunks and
        cross-faded together so the model stays within its comfortable
        context window, avoiding OOM and quality degradation.

        Args:
            prompt: Music style description
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
        dur: float = float(duration or self.duration)

        console.print(
            f"[cyan]Generating {dur}s of music: '{prompt}'[/cyan]"
        )

        import scipy.io.wavfile  # type: ignore[import-untyped]

        sample_rate = self._model.config.audio_encoder.sampling_rate
        tokens_per_second = 50  # approximate

        # Decide whether to use single-shot or chunked generation
        if dur <= _MAX_CHUNK_SECS:
            audio_data = self._generate_audio_chunk(
                prompt, dur, tokens_per_second
            )
        else:
            audio_data = self._generate_chunked(
                prompt, dur, tokens_per_second, sample_rate
            )

        if audio_data is None:
            console.print("[red]✗ Music generation failed[/red]")
            return None

        # Save as WAV
        scipy.io.wavfile.write(output_path, rate=sample_rate, data=audio_data)

        console.print(f"[green]✓ Music saved to {output_path}[/green]")

        if self.config.auto_unload:
            self.unload()

        return output_path

    # -----------------------------------------------------------------
    # Internal generation helpers
    # -----------------------------------------------------------------

    def _generate_audio_chunk(
        self,
        prompt: str,
        duration: float,
        tokens_per_second: int,
    ) -> Optional[np.ndarray]:
        """Generate a single chunk of audio, with CUDA-fallback retry."""
        if self._model is None or self._processor is None:
            return None

        dev = getattr(self, "_active_device", str(self.config.device))
        inputs = self._processor(
            text=[prompt], padding=True, return_tensors="pt",
        )
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        max_new_tokens = int(duration * tokens_per_second)

        audio_values = None
        for attempt in range(2):
            try:
                with torch.no_grad():
                    audio_values = self._model.generate(
                        **inputs, max_new_tokens=max_new_tokens,
                    )
                break
            except RuntimeError as exc:
                if attempt == 0 and "CUDA" in str(exc):
                    console.print(
                        "[yellow]⚠ CUDA error during MusicGen — "
                        "falling back to CPU...[/yellow]"
                    )
                    self._force_unload()
                    self._load_model(device="cpu")
                    dev = "cpu"
                    inputs = self._processor(
                        text=[prompt], padding=True, return_tensors="pt",
                    )
                    continue
                raise

        if audio_values is None:
            return None
        return audio_values[0, 0].cpu().float().numpy()

    def _generate_chunked(
        self,
        prompt: str,
        total_duration: float,
        tokens_per_second: int,
        sample_rate: int,
    ) -> Optional[np.ndarray]:
        """Generate long audio in overlapping chunks and cross-fade."""
        overlap_samples = int(_CROSSFADE_SECS * sample_rate)
        chunks: list[np.ndarray] = []
        remaining = total_duration

        chunk_idx = 0
        while remaining > 0:
            # Last chunk can be shorter
            chunk_dur = min(remaining + _CROSSFADE_SECS, _MAX_CHUNK_SECS)
            chunk_dur = max(chunk_dur, 2.0)  # minimum sensible length

            console.print(
                f"[dim]  Music chunk {chunk_idx + 1}: "
                f"{chunk_dur:.0f}s ({remaining:.0f}s remaining)[/dim]"
            )

            data = self._generate_audio_chunk(prompt, chunk_dur, tokens_per_second)
            if data is None:
                return None
            chunks.append(data)
            remaining -= (chunk_dur - _CROSSFADE_SECS)
            chunk_idx += 1

        # Cross-fade adjacent chunks
        if len(chunks) == 1:
            return chunks[0]

        result = chunks[0]
        for next_chunk in chunks[1:]:
            xf = min(overlap_samples, len(result), len(next_chunk))
            if xf > 0:
                fade_out = np.linspace(1.0, 0.0, xf, dtype=np.float32)
                fade_in = np.linspace(0.0, 1.0, xf, dtype=np.float32)
                result[-xf:] = result[-xf:] * fade_out + next_chunk[:xf] * fade_in
                result = np.concatenate([result, next_chunk[xf:]])
            else:
                result = np.concatenate([result, next_chunk])

        return result

    def generate_single(
        self,
        prompt: str,
        output_path: str,
        duration: Optional[float] = None,
    ) -> Optional[str]:
        """Synchronous convenience wrapper."""
        import asyncio

        return asyncio.run(self.generate_music(prompt, output_path, duration))
