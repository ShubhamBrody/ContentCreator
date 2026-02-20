"""
ContentCreator - Data Models (Pydantic Schemas)

Defines the structured data flowing through the pipeline:
  Script → Scenes → Assets → Final Video
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

class Platform(str, Enum):
    """Target platform / aspect ratio."""
    YOUTUBE = "youtube"      # 16:9 landscape
    REELS = "reels"          # 9:16 portrait
    SHORTS = "shorts"        # 9:16 portrait


class SceneTransition(str, Enum):
    """Transition between scenes."""
    CUT = "cut"
    FADE = "fade"
    CROSSFADE = "crossfade"
    SLIDE_LEFT = "slide_left"
    SLIDE_RIGHT = "slide_right"
    ZOOM_IN = "zoom_in"


# =============================================================================
# Scene Model — One segment of the video
# =============================================================================

class Scene(BaseModel):
    """A single scene in the video."""

    scene_number: int = Field(..., description="Sequential scene number (1-based)")
    title: str = Field(..., description="Short title for the scene")
    narration: str = Field(..., description="Voiceover text for this scene")
    image_prompt: str = Field(
        ...,
        description="Detailed prompt for AI image generation "
                    "(visual description of what should appear)"
    )
    duration_seconds: float = Field(
        default=5.0,
        description="Estimated duration in seconds"
    )
    transition: SceneTransition = Field(
        default=SceneTransition.FADE,
        description="Transition to use AFTER this scene"
    )
    music_mood: Optional[str] = Field(
        default=None,
        description="Music mood/style for this scene (e.g., 'upbeat', 'dramatic')"
    )

    # --- Paths populated during pipeline ---
    audio_path: Optional[str] = Field(default=None, description="Path to generated TTS audio")
    image_path: Optional[str] = Field(default=None, description="Path to generated image")
    video_clip_path: Optional[str] = Field(default=None, description="Path to generated video clip")


# =============================================================================
# Script Model — Full parsed script
# =============================================================================

class ParsedScript(BaseModel):
    """The full script broken down into scenes by the LLM."""

    title: str = Field(..., description="Video title")
    description: str = Field(default="", description="Short video description")
    platform: Platform = Field(default=Platform.REELS, description="Target platform")
    scenes: List[Scene] = Field(..., description="Ordered list of scenes")
    music_prompt: str = Field(
        default="background music",
        description="Overall music style/prompt for the video"
    )

    @property
    def total_duration(self) -> float:
        return sum(s.duration_seconds for s in self.scenes)

    @property
    def scene_count(self) -> int:
        return len(self.scenes)


# =============================================================================
# Pipeline Artifacts — Tracks generated files
# =============================================================================

class PipelineArtifacts(BaseModel):
    """Tracks all generated assets during the pipeline run."""

    project_dir: str = Field(..., description="Root output directory for this run")
    parsed_script: Optional[ParsedScript] = None
    scene_audio_files: List[str] = Field(default_factory=list)
    scene_image_files: List[str] = Field(default_factory=list)
    scene_video_files: List[str] = Field(default_factory=list)
    music_file: Optional[str] = None
    subtitle_file: Optional[str] = None
    final_video_path: Optional[str] = None

    def get_project_path(self, filename: str) -> str:
        return str(Path(self.project_dir) / filename)
