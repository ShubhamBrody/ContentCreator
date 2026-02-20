"""
ContentCreator - Text-to-Speech Engine

Generates voiceover audio from scene narration text.
Supports:
  - edge-tts: Microsoft Edge TTS (free, high quality, many voices, no GPU needed)
  - coqui: Coqui XTTS v2 (local, GPU, voice cloning — requires Python <3.12)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config import Config
from src.models.schemas import ParsedScript

console = Console()


# =============================================================================
# Default voice map for edge-tts
# =============================================================================
EDGE_VOICES = {
    "en-male": "en-US-GuyNeural",
    "en-female": "en-US-JennyNeural",
    "en-narrator": "en-US-DavisNeural",
    "en-cheerful": "en-US-AriaNeural",
    "en-british-male": "en-GB-RyanNeural",
    "en-british-female": "en-GB-SoniaNeural",
    "en-australian-female": "en-AU-NatashaNeural",
    "hi-male": "hi-IN-MadhurNeural",
    "hi-female": "hi-IN-SwaraNeural",
}


class TTSEngine:
    """Text-to-Speech engine using Microsoft Edge TTS (free, high quality)."""

    def __init__(self, config: Config):
        self.config = config
        self.engine = config.tts.get("engine", "edge")
        self.language = config.tts.get("language", "en")
        self.voice = config.tts.get("voice", "en-US-GuyNeural")
        self.rate = config.tts.get("rate", "+0%")
        self.volume = config.tts.get("volume", "+0%")
        self.pitch = config.tts.get("pitch", "+0Hz")

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
                filename = f"scene_{scene.scene_number:03d}_audio.mp3"
                filepath = str(Path(output_dir) / filename)

                progress.update(
                    task,
                    description=f"TTS: Scene {scene.scene_number}/{parsed_script.scene_count}",
                )

                await self._synthesize(scene.narration, filepath)
                audio_files.append(filepath)
                scene.audio_path = filepath

                progress.advance(task)

        console.print(f"[green]✓ Generated {len(audio_files)} audio files[/green]")
        return audio_files

    async def _synthesize(self, text: str, output_path: str) -> None:
        """Synthesize text to audio using edge-tts."""
        import edge_tts

        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            volume=self.volume,
            pitch=self.pitch,
        )
        await communicate.save(output_path)

    async def generate_single(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
    ) -> str:
        """
        Generate audio for a single text (utility method).

        Args:
            text: Text to synthesize
            output_path: Where to save the audio
            voice: Optional voice override

        Returns:
            Path to generated audio file
        """
        import edge_tts

        v = voice or self.voice
        communicate = edge_tts.Communicate(
            text=text,
            voice=v,
            rate=self.rate,
            volume=self.volume,
            pitch=self.pitch,
        )
        await communicate.save(output_path)
        return output_path

    @staticmethod
    async def list_voices(language: str = "en") -> List[str]:
        """List available voices for a language."""
        import edge_tts

        voices = await edge_tts.list_voices()
        filtered = [
            f"{v['ShortName']} ({v['Gender']}) - {v['Locale']}"
            for v in voices
            if v.get("Locale", "").startswith(language)
        ]
        return filtered
