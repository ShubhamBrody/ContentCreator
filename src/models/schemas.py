"""
ContentCreator - Data Models (Pydantic Schemas)

Defines the structured data flowing through the pipeline:
  Script → Scenes → Assets → Final Video
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


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
# Character Model
# =============================================================================

class CharacterDef(BaseModel):
    """Definition of a character used across scenes."""

    name: str = Field(..., description="Character name/identifier")
    description: str = Field(
        ...,
        description="Visual description (e.g., 'tall man with glasses and a beard')"
    )
    photo_path: Optional[str] = Field(
        default=None, description="Path to a real photo to convert to sketch"
    )
    sketch_path: Optional[str] = Field(
        default=None, description="Path to generated sketch (populated by pipeline)"
    )
    style: str = Field(
        default="stick_figure",
        description="Sketch style: stick_figure, cartoon, manga, doodle, whiteboard"
    )
    traits: List[str] = Field(
        default_factory=list,
        description="Visual traits: ['wears red hat', 'has curly hair']"
    )


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
    characters_in_scene: List[str] = Field(
        default_factory=list,
        description="Names of characters that appear in this scene"
    )
    character_actions: Optional[str] = Field(
        default=None,
        description="What the characters are doing (e.g., 'talking to each other', 'running')"
    )
    character_emotions: Optional[str] = Field(
        default=None,
        description="Character emotions/expressions (e.g., 'happy', 'surprised, angry')"
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

    # --- Validators: handle common LLM output variations ---

    @field_validator("character_actions", "character_emotions", mode="before")
    @classmethod
    def _coerce_to_string(cls, v):
        """LLMs sometimes return a list instead of a string."""
        if isinstance(v, list):
            return ", ".join(str(i) for i in v)
        return v

    @field_validator("transition", mode="before")
    @classmethod
    def _normalize_transition(cls, v):
        """Fall back to 'fade' if the LLM returns an unsupported transition."""
        valid = {e.value for e in SceneTransition}
        if isinstance(v, str) and v.lower().strip() not in valid:
            return SceneTransition.FADE.value
        return v

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
    characters: List[CharacterDef] = Field(
        default_factory=list,
        description="Characters appearing in the video"
    )
    scenes: List[Scene] = Field(..., description="Ordered list of scenes")
    music_prompt: str = Field(
        default="background music",
        description="Overall music style/prompt for the video"
    )
    visual_style: str = Field(
        default="",
        description=(
            "Detected art / visual style for image generation. "
            "For known IPs this contains the franchise-specific style "
            "(e.g. 'anime art style, Attack on Titan aesthetic'). "
            "Empty when no specific style was detected."
        ),
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
    character_sketches: List[str] = Field(default_factory=list, description="Paths to character sketch images")
    scene_audio_files: List[str] = Field(default_factory=list)
    scene_image_files: List[str] = Field(default_factory=list)
    scene_video_files: List[str] = Field(default_factory=list)
    music_file: Optional[str] = None
    subtitle_file: Optional[str] = None
    final_video_path: Optional[str] = None

    def get_project_path(self, filename: str) -> str:
        return str(Path(self.project_dir) / filename)
