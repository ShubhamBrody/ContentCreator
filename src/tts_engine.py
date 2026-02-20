"""
ContentCreator - Text-to-Speech Engine

Generates voiceover audio from scene narration text.
Supports Coqui XTTS v2 (local, free, GPU-accelerated).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config import Config
from src.gpu_utils import free_vram, log_vram, unload_model
from src.models.schemas import ParsedScript

console = Console()


class TTSEngine:
    """Text-to-Speech engine using Coqui XTTS v2."""

    def __init__(self, config: Config):
        self.config = config
        self.engine = config.tts.get("engine", "coqui")
        self.model_name = config.tts.get(
            "model", "tts_models/multilingual/multi-dataset/xtts_v2"
        )
        self.language = config.tts.get("language", "en")
        self.speaker_wav = config.tts.get("speaker_wav")
        self.speed = config.tts.get("speed", 1.0)
        self._tts_model = None

    def _load_model(self) -> None:
        """Load the TTS model onto GPU."""
        if self._tts_model is not None:
            return

        console.print("[cyan]Loading TTS model (Coqui XTTS v2)...[/cyan]")
        log_vram("before TTS load")

        from TTS.api import TTS  # type: ignore[import-untyped]

        self._tts_model = TTS(model_name=self.model_name)

        # Move to GPU if available
        device = self.config.device
        if device == "cuda":
            self._tts_model = self._tts_model.to(device)

        log_vram("after TTS load")
        console.print("[green]✓ TTS model loaded[/green]")

    def unload(self) -> None:
        """Unload the TTS model and free VRAM."""
        if self._tts_model is not None:
            console.print("[dim]Unloading TTS model...[/dim]")
            unload_model(self._tts_model)
            self._tts_model = None
            free_vram()
            log_vram("after TTS unload")

    async def generate_scene_audio(
        self,
        parsed_script: ParsedScript,
        output_dir: str,
    ) -> List[str]:
        """
        Generate voiceover audio for each scene.

        Args:
            parsed_script: The parsed script with scenes
            output_dir: Directory to save audio files

        Returns:
            List of paths to generated audio files
        """
        self._load_model()
        os.makedirs(output_dir, exist_ok=True)

        audio_files: List[str] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Generating voiceover...",
                total=len(parsed_script.scenes),
            )

            for scene in parsed_script.scenes:
                filename = f"scene_{scene.scene_number:03d}_audio.wav"
                filepath = str(Path(output_dir) / filename)

                progress.update(
                    task,
                    description=f"TTS: Scene {scene.scene_number}/{parsed_script.scene_count}",
                )

                self._synthesize(scene.narration, filepath)
                audio_files.append(filepath)
                scene.audio_path = filepath

                progress.advance(task)

        console.print(f"[green]✓ Generated {len(audio_files)} audio files[/green]")

        # Unload if configured
        if self.config.auto_unload:
            self.unload()

        return audio_files

    def _synthesize(self, text: str, output_path: str) -> None:
        """Synthesize a single text to audio file."""
        if self._tts_model is None:
            raise RuntimeError("TTS model not loaded. Call _load_model() first.")

        if self.speaker_wav and os.path.exists(self.speaker_wav):
            # Voice cloning mode
            self._tts_model.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=self.speaker_wav,
                language=self.language,
                speed=self.speed,
            )
        else:
            # Default speaker
            self._tts_model.tts_to_file(
                text=text,
                file_path=output_path,
                language=self.language,
                speed=self.speed,
            )

    def generate_single(
        self,
        text: str,
        output_path: str,
        speaker_wav: Optional[str] = None,
    ) -> str:
        """
        Generate audio for a single text (utility method).

        Args:
            text: Text to synthesize
            output_path: Where to save the audio
            speaker_wav: Optional reference audio for voice cloning

        Returns:
            Path to generated audio file
        """
        self._load_model()

        if speaker_wav:
            self.speaker_wav = speaker_wav

        self._synthesize(text, output_path)

        if self.config.auto_unload:
            self.unload()

        return output_path
