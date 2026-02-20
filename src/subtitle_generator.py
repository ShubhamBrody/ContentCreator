"""
ContentCreator - Subtitle Generator

Generates word-level timed subtitles using OpenAI Whisper (local, free).
Creates styled subtitle overlays for the final video.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

from src.config import Config
from src.gpu_utils import free_vram, log_vram

console = Console()


# =============================================================================
# Subtitle Data
# =============================================================================

@dataclass
class SubtitleWord:
    """A single word with timing information."""
    word: str
    start: float  # seconds
    end: float    # seconds


@dataclass
class SubtitleSegment:
    """A group of words displayed together as one subtitle line."""
    text: str
    start: float
    end: float
    words: List[SubtitleWord] = field(default_factory=list)


# =============================================================================
# Subtitle Generator
# =============================================================================

class SubtitleGenerator:
    """Generates timed subtitles from audio using Whisper."""

    def __init__(self, config: Config):
        self.config = config
        self.enabled = config.subtitles.get("enabled", True)
        self.model_size = config.subtitles.get("model_size", "base")
        self.font = config.subtitles.get("font", "Arial-Bold")
        self.font_size = config.subtitles.get("font_size", 48)
        self.color = config.subtitles.get("color", "white")
        self.stroke_color = config.subtitles.get("stroke_color", "black")
        self.stroke_width = config.subtitles.get("stroke_width", 2)
        self.position = config.subtitles.get("position", "bottom")
        self.max_words_per_line = config.subtitles.get("max_words_per_line", 5)
        self._model: Any = None

    def _load_model(self) -> None:
        """Load Whisper model."""
        if self._model is not None:
            return

        console.print(f"[cyan]Loading Whisper ({self.model_size}) for subtitles...[/cyan]")
        log_vram("before Whisper load")

        import whisper  # type: ignore[import-untyped]

        self._model = whisper.load_model(
            self.model_size,
            device=self.config.device,
        )

        log_vram("after Whisper load")
        console.print("[green]✓ Whisper model loaded[/green]")

    def unload(self) -> None:
        """Unload Whisper model."""
        if self._model is not None:
            console.print("[dim]Unloading Whisper model...[/dim]")
            del self._model
            self._model = None
            free_vram()
            log_vram("after Whisper unload")

    async def generate_subtitles(
        self,
        audio_files: List[str],
        output_dir: str,
    ) -> List[List[SubtitleSegment]]:
        """
        Generate timed subtitles for each audio file.

        Args:
            audio_files: List of audio file paths (one per scene)
            output_dir: Directory to save SRT files

        Returns:
            List of subtitle segments per scene
        """
        if not self.enabled:
            console.print("[dim]Subtitles disabled.[/dim]")
            return []

        self._load_model()
        os.makedirs(output_dir, exist_ok=True)

        all_segments: List[List[SubtitleSegment]] = []

        for i, audio_path in enumerate(audio_files):
            console.print(f"[dim]Transcribing scene {i + 1}/{len(audio_files)}...[/dim]")

            segments = self._transcribe(audio_path)
            all_segments.append(segments)

            # Save SRT file
            srt_path = str(Path(output_dir) / f"scene_{i + 1:03d}.srt")
            self._write_srt(segments, srt_path)

        console.print(f"[green]✓ Generated subtitles for {len(audio_files)} scenes[/green]")

        if self.config.auto_unload:
            self.unload()

        return all_segments

    def _transcribe(self, audio_path: str) -> List[SubtitleSegment]:
        """Transcribe audio file and return timed segments."""
        if self._model is None:
            raise RuntimeError("Whisper model not loaded.")

        result = self._model.transcribe(
            audio_path,
            word_timestamps=True,
            language="en",
        )

        segments: List[SubtitleSegment] = []

        for segment_data in result.get("segments", []):
            words_data = segment_data.get("words", [])

            # Group words into chunks of max_words_per_line
            words = [
                SubtitleWord(
                    word=w["word"].strip(),
                    start=w["start"],
                    end=w["end"],
                )
                for w in words_data
                if w.get("word", "").strip()
            ]

            # Chunk words into subtitle lines
            for j in range(0, len(words), self.max_words_per_line):
                chunk = words[j : j + self.max_words_per_line]
                if not chunk:
                    continue

                segments.append(
                    SubtitleSegment(
                        text=" ".join(w.word for w in chunk),
                        start=chunk[0].start,
                        end=chunk[-1].end,
                        words=chunk,
                    )
                )

        return segments

    def _write_srt(
        self,
        segments: List[SubtitleSegment],
        output_path: str,
    ) -> None:
        """Write subtitle segments to SRT file format."""
        with open(output_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                start_ts = self._format_timestamp(seg.start)
                end_ts = self._format_timestamp(seg.end)
                f.write(f"{i}\n")
                f.write(f"{start_ts} --> {end_ts}\n")
                f.write(f"{seg.text}\n\n")

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

    def get_subtitle_style(self) -> Dict[str, Any]:
        """Return subtitle styling config for the video assembler."""
        return {
            "font": self.font,
            "font_size": self.font_size,
            "color": self.color,
            "stroke_color": self.stroke_color,
            "stroke_width": self.stroke_width,
            "position": self.position,
        }
