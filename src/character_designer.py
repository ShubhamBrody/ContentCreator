"""
ContentCreator - Character Designer

Creates sketch/cartoon-style characters for use in animated video scenes.

Two modes:
  1. Photo → Sketch: Takes a real photo and converts it into a hand-drawn
     sketch/stick-figure cartoon character (like the "miguelito queriendo" style).
     Uses Canny edge detection + SDXL ControlNet.

  2. Description → Sketch: Takes a text description of a person/animal/character
     and generates a sketch-style character using SDXL with sketch style prompts.

Characters are stored as reference sheets and injected into scene image prompts
for visual consistency across the video.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from rich.console import Console

from src.config import Config
from src.gpu_utils import free_vram, log_vram, unload_model

console = Console()


# =============================================================================
# Sketch Style Presets
# =============================================================================

SKETCH_STYLES: Dict[str, Dict[str, str]] = {
    "stick_figure": {
        "prompt_suffix": (
            "simple stick figure character, black ink on white background, "
            "bold outlines, minimalist cartoon style, hand-drawn sketch, "
            "expressive face, like a newspaper comic strip character, "
            "clean lines, no shading, white background"
        ),
        "negative": (
            "realistic, photographic, 3d render, detailed shading, "
            "complex background, gradient, watercolor, oil painting"
        ),
    },
    "cartoon": {
        "prompt_suffix": (
            "cartoon character illustration, bold black outlines, "
            "flat colors, animated style, expressive cartoon face, "
            "simple shapes, like an animated TV show character, "
            "clean vector art style, solid color fills"
        ),
        "negative": (
            "realistic, photographic, 3d render, complex shading, "
            "noise, textured, watercolor, oil painting, blurry"
        ),
    },
    "manga": {
        "prompt_suffix": (
            "manga style character, black and white ink drawing, "
            "clean lineart, expressive anime eyes, manga panel style, "
            "screen tones, Japanese comic art style"
        ),
        "negative": (
            "realistic, photographic, 3d render, western comic style, "
            "watercolor, oil painting, blurry, low quality"
        ),
    },
    "doodle": {
        "prompt_suffix": (
            "hand-drawn doodle character, rough sketchy lines, "
            "notebook doodle style, black pen on white paper, "
            "whimsical and playful, imperfect hand-drawn look, "
            "casual sketch style with personality"
        ),
        "negative": (
            "realistic, photographic, 3d render, clean lines, "
            "professional, polished, digital art, watercolor"
        ),
    },
    "whiteboard": {
        "prompt_suffix": (
            "whiteboard animation style character, simple black marker "
            "drawing on white background, educational illustration style, "
            "like a whiteboard explainer video, clean simple shapes"
        ),
        "negative": (
            "realistic, photographic, 3d render, colorful, complex, "
            "detailed background, gradient, textured"
        ),
    },
}


# =============================================================================
# Character Data
# =============================================================================

class CharacterProfile:
    """Represents a character with visual reference and description."""

    def __init__(
        self,
        name: str,
        description: str,
        reference_image_path: Optional[str] = None,
        sketch_path: Optional[str] = None,
        style: str = "stick_figure",
        traits: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.reference_image_path = reference_image_path
        self.sketch_path = sketch_path
        self.style = style
        self.traits = traits or []

    def to_prompt_description(self) -> str:
        """Generate a text description for embedding in image prompts."""
        parts = [f"character named {self.name}"]
        if self.description:
            parts.append(self.description)
        if self.traits:
            parts.append(f"traits: {', '.join(self.traits)}")
        return ", ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "reference_image_path": self.reference_image_path,
            "sketch_path": self.sketch_path,
            "style": self.style,
            "traits": self.traits,
        }


# =============================================================================
# Character Designer
# =============================================================================

class CharacterDesigner:
    """
    Creates sketch-style character art from photos or descriptions.

    Pipeline:
      Photo mode:  photo → edge detection → ControlNet SDXL → sketch character
      Text mode:   description → SDXL with sketch style prompt → sketch character
    """

    def __init__(self, config: Config):
        self.config = config
        self.default_style = config.image.get("character_style", "stick_figure")
        self._pipe: Any = None
        self._controlnet: Any = None

    # =========================================================================
    # Photo → Sketch (Edge Detection + ControlNet)
    # =========================================================================

    def photo_to_sketch(
        self,
        photo_path: str,
        output_path: str,
        character_name: str = "Character",
        style: str = "stick_figure",
        description: str = "",
    ) -> CharacterProfile:
        """
        Convert a real photo into a sketch-style character.

        Args:
            photo_path: Path to the input photo
            output_path: Where to save the sketch
            character_name: Name for this character
            style: Sketch style preset (stick_figure, cartoon, manga, doodle, whiteboard)
            description: Optional extra description

        Returns:
            CharacterProfile with the generated sketch
        """
        console.print(f"[cyan]Converting photo to sketch: {character_name}[/cyan]")

        if not os.path.exists(photo_path):
            raise FileNotFoundError(f"Photo not found: {photo_path}")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Step 1: Extract edges from photo
        edge_image = self._extract_edges(photo_path)

        # Step 2: Generate sketch using SDXL with ControlNet (or img2img if ControlNet unavailable)
        style_config = SKETCH_STYLES.get(style, SKETCH_STYLES["stick_figure"])

        prompt = self._build_character_prompt(
            description or f"a person based on the reference photo",
            character_name,
            style_config,
        )

        self._generate_sketch_from_edges(
            edge_image=edge_image,
            prompt=prompt,
            negative_prompt=style_config["negative"],
            output_path=output_path,
        )

        profile = CharacterProfile(
            name=character_name,
            description=description,
            reference_image_path=photo_path,
            sketch_path=output_path,
            style=style,
        )

        console.print(f"[green]✓ Character sketch saved: {output_path}[/green]")
        return profile

    # =========================================================================
    # Description → Sketch (Text-to-Image with sketch style)
    # =========================================================================

    def description_to_sketch(
        self,
        description: str,
        output_path: str,
        character_name: str = "Character",
        style: str = "stick_figure",
        poses: Optional[List[str]] = None,
    ) -> CharacterProfile:
        """
        Generate a sketch character from a text description.

        Args:
            description: Text description of the character
            output_path: Where to save the sketch
            character_name: Name for this character
            style: Sketch style preset
            poses: Optional list of poses to generate (generates multiple sheets)

        Returns:
            CharacterProfile with the generated sketch
        """
        console.print(
            f"[cyan]Generating character from description: {character_name}[/cyan]"
        )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        style_config = SKETCH_STYLES.get(style, SKETCH_STYLES["stick_figure"])

        # Generate main character sheet
        prompt = self._build_character_prompt(description, character_name, style_config)
        self._generate_sketch_from_text(prompt, style_config["negative"], output_path)

        # Generate additional poses if requested
        pose_paths: List[str] = []
        if poses:
            base_dir = os.path.dirname(output_path)
            base_name = Path(output_path).stem
            for i, pose in enumerate(poses):
                pose_path = str(Path(base_dir) / f"{base_name}_pose_{i + 1}.png")
                pose_prompt = self._build_character_prompt(
                    f"{description}, {pose}",
                    character_name,
                    style_config,
                )
                self._generate_sketch_from_text(
                    pose_prompt, style_config["negative"], pose_path
                )
                pose_paths.append(pose_path)

        profile = CharacterProfile(
            name=character_name,
            description=description,
            sketch_path=output_path,
            style=style,
        )

        console.print(f"[green]✓ Character sketch saved: {output_path}[/green]")
        if pose_paths:
            console.print(f"[green]  + {len(pose_paths)} pose variations[/green]")

        return profile

    # =========================================================================
    # Edge Detection (for photo-to-sketch pipeline)
    # =========================================================================

    def _extract_edges(self, image_path: str) -> Image.Image:
        """Extract edges from a photo using Canny-like edge detection."""
        console.print("[dim]Extracting edges from photo...[/dim]")

        img = Image.open(image_path).convert("L")  # Grayscale

        # Resize to standard size
        img = img.resize((512, 512), Image.LANCZOS)

        # Apply edge detection using PIL filters
        # Step 1: Gaussian blur to reduce noise
        img_blur = img.filter(ImageFilter.GaussianBlur(radius=1))

        # Step 2: Find edges
        edges = img_blur.filter(ImageFilter.FIND_EDGES)

        # Step 3: Enhance and threshold to get clean black/white lines
        edges = ImageOps.autocontrast(edges)

        # Step 4: Invert (black lines on white background)
        edges = ImageOps.invert(edges)

        # Step 5: Threshold to get clean binary image
        threshold = 180
        edges = edges.point(lambda p: 255 if p > threshold else 0)

        return edges.convert("RGB")

    def _extract_edges_canny(self, image_path: str) -> Image.Image:
        """Extract edges using OpenCV Canny (better quality, requires cv2)."""
        try:
            import cv2  # type: ignore[import-untyped]

            img = cv2.imread(image_path)
            img = cv2.resize(img, (512, 512))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # Canny edge detection
            edges = cv2.Canny(gray, 50, 150)

            # Invert: black lines on white
            edges = 255 - edges

            return Image.fromarray(edges).convert("RGB")
        except ImportError:
            console.print("[yellow]OpenCV not available, using PIL edge detection[/yellow]")
            return self._extract_edges(image_path)

    # =========================================================================
    # Image Generation Backends
    # =========================================================================

    def _load_pipeline(self) -> None:
        """Load SDXL pipeline for character generation."""
        if self._pipe is not None:
            return

        console.print("[cyan]Loading SDXL for character design...[/cyan]")
        log_vram("before character SDXL load")

        from diffusers import StableDiffusionXLPipeline  # type: ignore[import-untyped]

        dtype = torch.float16 if self.config.half_precision else torch.float32
        model_id = self.config.image.get(
            "model", "stabilityai/stable-diffusion-xl-base-1.0"
        )

        self._pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if self.config.half_precision else None,
        )
        self._pipe.to(self.config.device)
        self._pipe.enable_attention_slicing()

        log_vram("after character SDXL load")
        console.print("[green]✓ SDXL loaded for character design[/green]")

    def _load_controlnet_pipeline(self) -> None:
        """Load SDXL + ControlNet Canny pipeline for photo-to-sketch."""
        if self._controlnet is not None:
            return

        console.print("[cyan]Loading ControlNet for photo-to-sketch...[/cyan]")
        log_vram("before ControlNet load")

        try:
            from diffusers import (  # type: ignore[import-untyped]
                ControlNetModel,
                StableDiffusionXLControlNetPipeline,
            )

            dtype = torch.float16 if self.config.half_precision else torch.float32
            model_id = self.config.image.get(
                "model", "stabilityai/stable-diffusion-xl-base-1.0"
            )
            controlnet_id = self.config.image.get(
                "controlnet_model", "diffusers/controlnet-canny-sdxl-1.0"
            )

            controlnet = ControlNetModel.from_pretrained(
                controlnet_id,
                torch_dtype=dtype,
            )

            self._controlnet = StableDiffusionXLControlNetPipeline.from_pretrained(
                model_id,
                controlnet=controlnet,
                torch_dtype=dtype,
                use_safetensors=True,
                variant="fp16" if self.config.half_precision else None,
            )
            self._controlnet.to(self.config.device)
            self._controlnet.enable_attention_slicing()

            log_vram("after ControlNet load")
            console.print("[green]✓ ControlNet loaded[/green]")

        except Exception as e:
            console.print(
                f"[yellow]ControlNet load failed ({e}), "
                f"falling back to img2img sketch style[/yellow]"
            )
            self._controlnet = None

    def unload(self) -> None:
        """Unload all models and free VRAM."""
        if self._pipe is not None:
            unload_model(self._pipe)
            self._pipe = None
        if self._controlnet is not None:
            unload_model(self._controlnet)
            self._controlnet = None
        free_vram()
        log_vram("after character designer unload")

    # =========================================================================
    # Generation Methods
    # =========================================================================

    def _generate_sketch_from_edges(
        self,
        edge_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        output_path: str,
    ) -> None:
        """Generate a sketch character guided by edge-detected reference."""
        # Try ControlNet first (better quality)
        self._load_controlnet_pipeline()

        if self._controlnet is not None:
            console.print("[dim]Generating sketch via ControlNet...[/dim]")
            with torch.no_grad():
                result = self._controlnet(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=edge_image,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    controlnet_conditioning_scale=0.8,
                    width=1024,
                    height=1024,
                ).images[0]
            result.save(output_path)
        else:
            # Fallback: use the edge image as a strong style guide via img2img
            self._load_pipeline()
            console.print("[dim]Generating sketch via SDXL (no ControlNet)...[/dim]")

            from diffusers import StableDiffusionXLImg2ImgPipeline  # type: ignore[import-untyped]

            img2img = StableDiffusionXLImg2ImgPipeline(**self._pipe.components)

            # Use edge image as init image with high strength
            init_image = edge_image.resize((1024, 1024), Image.LANCZOS)
            with torch.no_grad():
                result = img2img(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=init_image,
                    strength=0.75,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                ).images[0]
            result.save(output_path)

        if self.config.auto_unload:
            self.unload()

    def _generate_sketch_from_text(
        self,
        prompt: str,
        negative_prompt: str,
        output_path: str,
    ) -> None:
        """Generate a sketch character from text description only."""
        self._load_pipeline()

        console.print(f"[dim]Generating sketch from text...[/dim]")

        with torch.no_grad():
            result = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=1024,
                height=1024,
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]

        result.save(output_path)

        if self.config.auto_unload:
            self.unload()

    # =========================================================================
    # Prompt Building
    # =========================================================================

    @staticmethod
    def _build_character_prompt(
        description: str,
        name: str,
        style_config: Dict[str, str],
    ) -> str:
        """Build a full prompt for character generation."""
        return (
            f"character design sheet for '{name}', "
            f"{description}, "
            f"{style_config['prompt_suffix']}, "
            f"full body, front view, character reference sheet, "
            f"multiple expressions showing happy, sad, angry, surprised"
        )

    # =========================================================================
    # Utility: list available styles
    # =========================================================================

    @staticmethod
    def list_styles() -> List[str]:
        """Return available sketch style presets."""
        return list(SKETCH_STYLES.keys())

    @staticmethod
    def get_style_info(style: str) -> Dict[str, str]:
        """Get style prompt info."""
        return SKETCH_STYLES.get(style, SKETCH_STYLES["stick_figure"])
