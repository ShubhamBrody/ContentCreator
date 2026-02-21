"""
ContentCreator - Video Assembler

Stitches scene video clips, voiceover audio, background music,
and subtitles into a final video using MoviePy + FFmpeg.

For heavy scene loads (>12 clips), assembles in batches to avoid
exhausting file handles and RAM.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console

from src.config import Config
from src.models.schemas import ParsedScript, Platform, SceneTransition
from src.subtitle_generator import SubtitleSegment

console = Console()


def _resolve_font(font_name: str) -> str:
    """
    Resolve a font name to a full .ttf file path that Pillow / MoviePy can use.

    MoviePy v2 delegates text rendering to Pillow which requires an actual
    file path on Windows (font family names like 'Arial-Bold' don't work).
    This function tries several strategies:
      1. If font_name is already a valid file path, return it.
      2. Look in the Windows Fonts directory for common bold variants.
      3. Fall back to a guaranteed-present font (segoeui / tahoma / arial).
    """
    if os.path.isfile(font_name):
        return font_name

    fonts_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")

    # Map friendly names to actual .ttf filenames
    _FONT_MAP: dict[str, list[str]] = {
        "arial-bold":    ["arialbd.ttf", "Arial-Bold.ttf"],
        "arial":         ["arial.ttf"],
        "impact":        ["impact.ttf"],
        "calibri-bold":  ["calibrib.ttf"],
        "segoe-ui-bold": ["segoeuib.ttf"],
        "tahoma-bold":   ["tahomabd.ttf"],
    }

    # Try the requested font
    key = font_name.lower().replace(" ", "-")
    for candidates in [_FONT_MAP.get(key, []), [font_name, f"{font_name}.ttf"]]:
        for cand in candidates:
            full = os.path.join(fonts_dir, cand)
            if os.path.isfile(full):
                return full

    # Fallback chain: try common bold fonts that ship with every Windows install
    for fallback in ["arialbd.ttf", "impact.ttf", "segoeuib.ttf",
                     "tahomabd.ttf", "calibrib.ttf", "arial.ttf"]:
        full = os.path.join(fonts_dir, fallback)
        if os.path.isfile(full):
            console.print(
                f"[yellow]Font '{font_name}' not found, falling back "
                f"to {fallback}[/yellow]"
            )
            return full

    # Last resort — return original name and let Pillow try
    return font_name


class VideoAssembler:
    """Assembles all generated assets into the final video."""

    def __init__(self, config: Config):
        self.config = config

    # Maximum clips to keep open simultaneously. Beyond this, the assembler
    # concatenates in batches and writes intermediate files to avoid file-handle
    # and RAM exhaustion on heavy scene loads (20+ scenes).
    _BATCH_SIZE = 10

    def assemble(
        self,
        parsed_script: ParsedScript,
        video_clips: List[str],
        audio_files: List[str],
        music_path: Optional[str],
        subtitles: Optional[List[List[SubtitleSegment]]],
        output_path: str,
    ) -> str:
        """
        Assemble all assets into the final video.

        For >10 scenes, concatenation happens in batches so we never hold
        more than ``_BATCH_SIZE`` MoviePy clip objects simultaneously.

        Args:
            parsed_script: The parsed script (for metadata & timing)
            video_clips: Paths to scene video clips
            audio_files: Paths to scene voiceover audio files
            music_path: Path to background music (or None)
            subtitles: Subtitle segments per scene (or None)
            output_path: Where to save the final video

        Returns:
            Path to the final video
        """
        from moviepy import (  # type: ignore[import-untyped]
            AudioFileClip,
            CompositeAudioClip,
            CompositeVideoClip,
            TextClip,
            VideoFileClip,
            concatenate_videoclips,
        )

        console.print("[cyan]Assembling final video...[/cyan]")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        preset = self.config.get_output_preset(parsed_script.platform.value)
        target_w = preset["width"]
        target_h = preset["height"]
        target_fps = preset["fps"]
        bitrate = preset.get("bitrate", "8M")

        # =====================================================================
        # 1. Build scene clips (load, resize, attach audio & subtitles)
        # =====================================================================
        scene_count = len(parsed_script.scenes)
        use_batching = scene_count > self._BATCH_SIZE
        if use_batching:
            console.print(
                f"[dim]Heavy load: {scene_count} scenes — using batched assembly[/dim]"
            )

        scene_clips: list = []
        temp_dir = os.path.dirname(output_path) or "."
        batch_files: list[str] = []  # intermediate mp4 paths
        batch_idx = 0

        for i, scene in enumerate(parsed_script.scenes):
            clip_path = video_clips[i] if i < len(video_clips) else None
            audio_path = audio_files[i] if i < len(audio_files) else None

            if clip_path is None or not os.path.exists(clip_path):
                console.print(
                    f"[yellow]Warning: Missing video for scene {scene.scene_number}[/yellow]"
                )
                continue

            # Load video clip
            clip = VideoFileClip(clip_path)

            # Resize to target dimensions
            clip = clip.resized(new_size=(target_w, target_h))

            # Set duration to match voiceover audio if available
            if audio_path and os.path.exists(audio_path):
                audio_clip = AudioFileClip(audio_path)
                # Extend or trim video to match audio length
                if clip.duration < audio_clip.duration:
                    gap = audio_clip.duration - clip.duration
                    ratio = audio_clip.duration / max(clip.duration, 0.01)

                    if ratio <= 2.0:
                        # Small gap: slow the clip down to fill the duration
                        from moviepy.video.fx import MultiplySpeed
                        slow_factor = clip.duration / audio_clip.duration
                        clip = clip.with_effects(
                            [MultiplySpeed(factor=slow_factor)]
                        )
                    else:
                        # Large gap: loop
                        console.print(
                            f"[yellow]Warning: Scene {i+1} video "
                            f"({clip.duration:.1f}s) much shorter than "
                            f"audio ({audio_clip.duration:.1f}s) — looping "
                            f"(consider using image_motion engine)[/yellow]"
                        )
                        from moviepy.video.fx import Loop
                        clip = clip.with_effects(
                            [Loop(duration=audio_clip.duration)]
                        )
                else:
                    clip = clip.subclipped(0, audio_clip.duration)
                clip = clip.with_audio(audio_clip)
            else:
                clip = clip.with_duration(scene.duration_seconds)

            # Add subtitles for this scene
            if subtitles and i < len(subtitles) and subtitles[i]:
                clip = self._add_subtitles_to_clip(
                    clip, subtitles[i], target_w, target_h
                )

            scene_clips.append(clip)

            # --- Flush batch to disk if we hit the limit ---
            if use_batching and len(scene_clips) >= self._BATCH_SIZE:
                batch_path = os.path.join(
                    temp_dir, f"_batch_{batch_idx:03d}.mp4"
                )
                console.print(
                    f"[dim]  Writing batch {batch_idx + 1} "
                    f"({len(scene_clips)} clips)...[/dim]"
                )
                batch_clip = concatenate_videoclips(scene_clips, method="chain")
                batch_clip.write_videofile(
                    batch_path, fps=target_fps, codec="libx264",
                    audio_codec="aac", bitrate=bitrate, logger=None,
                )
                batch_clip.close()
                for c in scene_clips:
                    c.close()
                scene_clips.clear()
                batch_files.append(batch_path)
                batch_idx += 1
                gc.collect()

        if not scene_clips and not batch_files:
            raise RuntimeError("No video clips were loaded. Cannot assemble.")

        # =====================================================================
        # 2. Apply transitions and concatenate
        # =====================================================================
        if batch_files:
            # Flush remaining clips as the last batch
            if scene_clips:
                batch_path = os.path.join(
                    temp_dir, f"_batch_{batch_idx:03d}.mp4"
                )
                console.print(
                    f"[dim]  Writing batch {batch_idx + 1} "
                    f"({len(scene_clips)} clips)...[/dim]"
                )
                batch_clip = concatenate_videoclips(scene_clips, method="chain")
                batch_clip.write_videofile(
                    batch_path, fps=target_fps, codec="libx264",
                    audio_codec="aac", bitrate=bitrate, logger=None,
                )
                batch_clip.close()
                for c in scene_clips:
                    c.close()
                scene_clips.clear()
                batch_files.append(batch_path)
                gc.collect()

            # Re-load batches (now just N/10 files) and concatenate
            console.print(
                f"[dim]  Joining {len(batch_files)} batches...[/dim]"
            )
            batch_clips = [VideoFileClip(bf) for bf in batch_files]
            final_clip = concatenate_videoclips(batch_clips, method="chain")
        else:
            final_clip = self._concatenate_with_transitions(
                scene_clips, parsed_script
            )

        # =====================================================================
        # 3. Add background music
        # =====================================================================
        if music_path and os.path.exists(music_path):
            music_volume = self.config.music.get("volume", 0.3)
            music_clip = AudioFileClip(music_path)

            # Loop music if shorter than video
            if music_clip.duration < final_clip.duration:
                from moviepy.audio.fx import AudioLoop
                music_clip = music_clip.with_effects([AudioLoop(duration=final_clip.duration)])
            else:
                music_clip = music_clip.subclipped(0, final_clip.duration)

            # Reduce music volume
            music_clip = music_clip.with_volume_scaled(music_volume)

            # Mix with existing audio
            if final_clip.audio is not None:
                final_audio = CompositeAudioClip(
                    [final_clip.audio, music_clip]
                )
            else:
                final_audio = music_clip

            final_clip = final_clip.with_audio(final_audio)

        # =====================================================================
        # 4. Export final video
        # =====================================================================
        console.print(
            f"[cyan]Exporting video ({target_w}x{target_h} @ {target_fps}fps)...[/cyan]"
        )

        final_clip.write_videofile(
            output_path,
            fps=target_fps,
            codec=self.config.output.get("codec", "libx264"),
            audio_codec=self.config.output.get("audio_codec", "aac"),
            bitrate=bitrate,
            logger=None,
        )

        # Cleanup
        final_clip.close()
        for clip in scene_clips:
            clip.close()
        if batch_files:
            for bf_clip_path in batch_files:
                try:
                    os.remove(bf_clip_path)
                except OSError:
                    pass

        console.print(f"[green bold]✓ Final video saved: {output_path}[/green bold]")
        return output_path

    # =========================================================================
    # Transitions
    # =========================================================================

    def _concatenate_with_transitions(
        self,
        clips: List[Any],
        parsed_script: ParsedScript,
    ) -> Any:
        """Concatenate clips with transitions between scenes."""
        from moviepy import concatenate_videoclips  # type: ignore[import-untyped]

        transition_clips = []
        for i, clip in enumerate(clips):
            if i < len(parsed_script.scenes):
                transition = parsed_script.scenes[i].transition
            else:
                transition = SceneTransition.CUT

            if transition == SceneTransition.FADE and i > 0:
                from moviepy.video.fx import FadeIn, FadeOut
                clip = clip.with_effects([FadeIn(0.5)])
                if i < len(clips) - 1:
                    clip = clip.with_effects([FadeOut(0.5)])
            elif transition == SceneTransition.CROSSFADE and i > 0:
                from moviepy.video.fx import CrossFadeIn
                clip = clip.with_effects([CrossFadeIn(0.5)])

            transition_clips.append(clip)

        # Use method="compose" for crossfade support
        method = "compose" if any(
            s.transition == SceneTransition.CROSSFADE
            for s in parsed_script.scenes
        ) else "chain"

        return concatenate_videoclips(transition_clips, method=method)

    # =========================================================================
    # Subtitle Rendering
    # =========================================================================

    def _add_subtitles_to_clip(
        self,
        clip: Any,
        segments: List[SubtitleSegment],
        width: int,
        height: int,
    ) -> Any:
        """Overlay subtitle text onto a video clip."""
        from moviepy import CompositeVideoClip, TextClip  # type: ignore[import-untyped]

        sub_config = self.config.subtitles
        font_size = sub_config.get("font_size", 48)
        color = sub_config.get("color", "white")
        stroke_color = sub_config.get("stroke_color", "black")
        stroke_width = sub_config.get("stroke_width", 2)
        position = sub_config.get("position", "bottom")
        font_path = _resolve_font(sub_config.get("font", "Arial-Bold"))

        # Calculate Y position
        if position == "bottom":
            y_pos = height - font_size * 2 - 40
        elif position == "center":
            y_pos = height // 2
        else:  # top
            y_pos = 40

        subtitle_clips = [clip]

        for seg in segments:
            try:
                txt_clip = (
                    TextClip(
                        text=seg.text,
                        font_size=font_size,
                        color=color,
                        stroke_color=stroke_color,
                        stroke_width=stroke_width,
                        method="caption",
                        size=(width - 80, None),
                        font=font_path,
                        duration=seg.end - seg.start,
                    )
                    .with_position(("center", y_pos))
                    .with_start(seg.start)
                )
                subtitle_clips.append(txt_clip)
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Could not render subtitle '{seg.text}': {e}[/yellow]"
                )

        return CompositeVideoClip(subtitle_clips)
