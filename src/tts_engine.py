"""
ContentCreator - Text-to-Speech Engine  (Expressive Narration)

Generates voiceover audio from scene narration text using Microsoft Edge TTS.
Each scene's ``narration_tone`` drives an ``<mstts:express-as>`` SSML style
plus dynamic prosody (rate / pitch) so the narrator sounds like a real
storyteller with emotional range — not a flat robot.

Supported tones:
  excited, dramatic, calm, sad, angry, hopeful, cheerful, serious,
  fearful, inspiring, mysterious, epic, gentle, friendly, tense,
  triumphant, neutral

Recommended expressive voices (set in config.yaml → tts.voice):
  en-US-AriaNeural   — Female, 30+ styles, best for storytelling
  en-US-JennyNeural  — Female, warm & professional
  en-US-DavisNeural  — Male, conversational storyteller
  en-US-GuyNeural    — Male, authoritative narrator
  en-US-JasonNeural  — Male, natural & expressive
  en-US-SaraNeural   — Female, friendly narrator
"""

from __future__ import annotations

import contextvars
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config import Config
from src.models.schemas import ParsedScript

console = Console()


# =============================================================================
# Tone → Express-As Style  +  Prosody overrides
# =============================================================================
# Each entry:  "tone" → { style, rate, pitch }
#   style:  edge-tts express-as style name (None = no style tag)
#   rate:   speech speed relative to config (+/-  percentage)
#   pitch:  pitch shift   relative to config (+/-  Hz)
# Prosody values are *overrides*, not deltas, so they replace the config value
# for that scene only — the base config values are used for "neutral".

_TONE_MAP: Dict[str, Dict[str, Any]] = {
    "excited":     {"style": "excited",                "rate": "+12%",  "pitch": "+5Hz"},
    "dramatic":    {"style": "narration-professional",  "rate": "-8%",   "pitch": "-3Hz"},
    "calm":        {"style": "calm",                   "rate": "-12%",  "pitch": "-3Hz"},
    "sad":         {"style": "sad",                    "rate": "-18%",  "pitch": "-5Hz"},
    "angry":       {"style": "angry",                  "rate": "+8%",   "pitch": "+3Hz"},
    "hopeful":     {"style": "hopeful",                "rate": "+0%",   "pitch": "+2Hz"},
    "cheerful":    {"style": "cheerful",               "rate": "+8%",   "pitch": "+3Hz"},
    "serious":     {"style": "serious",                "rate": "-5%",   "pitch": "-2Hz"},
    "fearful":     {"style": "terrified",              "rate": "+10%",  "pitch": "+5Hz"},
    "inspiring":   {"style": "hopeful",                "rate": "+5%",   "pitch": "+3Hz"},
    "mysterious":  {"style": "whispering",             "rate": "-15%",  "pitch": "-5Hz"},
    "epic":        {"style": "narration-professional",  "rate": "-5%",   "pitch": "-3Hz"},
    "gentle":      {"style": "gentle",                 "rate": "-10%",  "pitch": "-3Hz"},
    "friendly":    {"style": "friendly",               "rate": "+0%",   "pitch": "+0Hz"},
    "tense":       {"style": "serious",                "rate": "+5%",   "pitch": "+2Hz"},
    "triumphant":  {"style": "excited",                "rate": "+10%",  "pitch": "+5Hz"},
    "neutral":     {"style": None,                     "rate": "+0%",   "pitch": "+0Hz"},
}

# Fallback styles when the primary isn't supported by the chosen voice.
# Maps unsupported style → safe alternative (every Neural voice supports these).
_STYLE_FALLBACK: Dict[str, str] = {
    "narration-professional": "serious",
    "calm":                   "friendly",
    "gentle":                 "friendly",
    "whispering":             "serious",
}


# =============================================================================
# Default voice map for edge-tts
# =============================================================================
EDGE_VOICES = {
    "en-male":              "en-US-GuyNeural",
    "en-female":            "en-US-JennyNeural",
    "en-narrator":          "en-US-DavisNeural",
    "en-storyteller":       "en-US-AriaNeural",
    "en-cheerful":          "en-US-AriaNeural",
    "en-british-male":      "en-GB-RyanNeural",
    "en-british-female":    "en-GB-SoniaNeural",
    "en-australian-female": "en-AU-NatashaNeural",
    "hi-male":              "hi-IN-MadhurNeural",
    "hi-female":            "hi-IN-SwaraNeural",
}


# =============================================================================
# Monkey-patch:  edge-tts  mkssml  →  express-as  SSML
# =============================================================================
# edge-tts 7.x builds SSML with only <prosody> tags.  We inject the Azure
# <mstts:express-as> element supported by Neural voices so the speech engine
# applies emotional colouring.  A context-var carries the current style to
# the patched function without changing the edge-tts call surface.

_current_style: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "edge_tts_express_style", default=None
)

_patch_applied = False


def _apply_ssml_patch() -> None:
    """One-time monkey-patch of ``edge_tts.communicate.mkssml``."""
    global _patch_applied
    if _patch_applied:
        return

    import edge_tts.communicate as _mod

    _original_mkssml = _mod.mkssml

    def _expressive_mkssml(
        tc: Any, escaped_text: Union[str, bytes]
    ) -> str:
        style = _current_style.get()
        if not style:
            return _original_mkssml(tc, escaped_text)

        if isinstance(escaped_text, bytes):
            escaped_text = escaped_text.decode("utf-8")

        return (
            "<speak version='1.0' "
            "xmlns='http://www.w3.org/2001/10/synthesis' "
            "xmlns:mstts='http://www.w3.org/2001/mstts' "
            "xml:lang='en-US'>"
            f"<voice name='{tc.voice}'>"
            f"<mstts:express-as style='{style}'>"
            f"<prosody pitch='{tc.pitch}' rate='{tc.rate}' volume='{tc.volume}'>"
            f"{escaped_text}"
            "</prosody>"
            "</mstts:express-as>"
            "</voice>"
            "</speak>"
        )

    _mod.mkssml = _expressive_mkssml  # type: ignore[assignment]
    _patch_applied = True


# =============================================================================
# TTS Engine
# =============================================================================

class TTSEngine:
    """Expressive Text-to-Speech engine using Microsoft Edge TTS.

    Applies per-scene emotional styles via ``<mstts:express-as>`` SSML and
    dynamic prosody adjustments driven by ``Scene.narration_tone``.
    """

    def __init__(self, config: Config):
        self.config = config
        self.engine = config.tts.get("engine", "edge")
        self.language = config.tts.get("language", "en")
        self.voice = config.tts.get("voice", "en-US-AriaNeural")
        self.rate = config.tts.get("rate", "+0%")
        self.volume = config.tts.get("volume", "+0%")
        self.pitch = config.tts.get("pitch", "+0Hz")

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    async def generate_scene_audio(
        self,
        parsed_script: ParsedScript,
        output_dir: str,
    ) -> List[str]:
        """Generate expressive voiceover audio for each scene.

        Reads ``scene.narration_tone`` and applies the matching express-as
        style plus prosody adjustments for every scene individually.
        """
        os.makedirs(output_dir, exist_ok=True)
        audio_files: List[str] = []

        # Make sure the SSML patch is in place before first synthesis
        _apply_ssml_patch()

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

                tone = getattr(scene, "narration_tone", "neutral") or "neutral"
                progress.update(
                    task,
                    description=(
                        f"TTS: Scene {scene.scene_number}/{parsed_script.scene_count} "
                        f"[{tone}]"
                    ),
                )

                await self._synthesize_expressive(
                    scene.narration, filepath, tone
                )
                audio_files.append(filepath)
                scene.audio_path = filepath

                progress.advance(task)

        console.print(f"[green]✓ Generated {len(audio_files)} expressive audio files[/green]")
        return audio_files

    async def generate_single(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        tone: Optional[str] = None,
    ) -> str:
        """Generate audio for a single text (utility method).

        Args:
            text: Text to synthesize
            output_path: Where to save the audio
            voice: Optional voice override
            tone: Optional narration tone override

        Returns:
            Path to generated audio file
        """
        _apply_ssml_patch()
        v = voice or self.voice
        t = tone or "neutral"
        await self._synthesize_expressive(text, output_path, t, voice_override=v)
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

    # -----------------------------------------------------------------
    # Internal: expressive synthesis
    # -----------------------------------------------------------------

    async def _synthesize_expressive(
        self,
        text: str,
        output_path: str,
        tone: str,
        voice_override: Optional[str] = None,
    ) -> None:
        """Synthesize *text* with emotional *tone* via Express-As SSML.

        Attempt order:
        1. Express-as style + tone prosody
        2. Fallback style (if defined) + tone prosody
        3. No express-as, tone prosody only  (still sounds different per scene)
        """
        import asyncio

        import edge_tts

        mapping = _TONE_MAP.get(tone, _TONE_MAP["neutral"])
        style: Optional[str] = mapping["style"]
        rate: str = mapping["rate"]
        pitch: str = mapping["pitch"]
        voice = voice_override or self.voice

        # Build ordered list of styles to try
        attempts: list[Optional[str]] = [style]
        if style and style in _STYLE_FALLBACK:
            attempts.append(_STYLE_FALLBACK[style])
        attempts.append(None)  # final fallback: plain prosody only

        for attempt_style in attempts:
            try:
                _current_style.set(attempt_style)
                communicate = edge_tts.Communicate(
                    text=text,
                    voice=voice,
                    rate=rate,
                    volume=self.volume,
                    pitch=pitch,
                )
                await communicate.save(output_path)
                return  # success
            except Exception:
                if attempt_style is None:
                    raise  # give up — even plain synthesis failed
                # Only log at debug level to avoid noisy output
                console.print(
                    f"[dim]express-as '{attempt_style}' unavailable, "
                    f"falling back to prosody[/dim]"
                )
                # Brief delay before retry to avoid rate-limiting
                await asyncio.sleep(0.5)
            finally:
                _current_style.set(None)
