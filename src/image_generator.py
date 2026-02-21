"""
ContentCreator - Image Generator

Generates images for each scene using Stable Diffusion XL (local, free).
Supports SDXL base + optional refiner for higher quality.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional

import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config import Config
from src.gpu_utils import free_vram, log_vram, unload_model
from src.models.schemas import CharacterDef, ParsedScript, Platform

console = Console()


class ImageGenerator:
    """Generates scene images using Stable Diffusion XL (diffusers)."""

    def __init__(self, config: Config):
        self.config = config
        self.model_id = config.image.get(
            "model", "stabilityai/stable-diffusion-xl-base-1.0"
        )
        self.refiner_id = config.image.get(
            "refiner", "stabilityai/stable-diffusion-xl-refiner-1.0"
        )
        self.use_refiner = config.image.get("use_refiner", False)
        self.num_inference_steps = config.image.get("num_inference_steps", 30)
        self.guidance_scale = config.image.get("guidance_scale", 7.5)
        self.negative_prompt = config.image.get(
            "negative_prompt",
            "blurry, low quality, distorted, deformed, ugly, bad anatomy",
        )
        # Character sketch-style overrides
        self._character_style: Optional[str] = None
        self._characters: dict[str, CharacterDef] = {}
        # Visual style detected from the script (e.g. "anime, Attack on Titan")
        self._visual_style: str = ""
        self._pipe: Any = None
        self._refiner: Any = None

    # -----------------------------------------------------------------
    # Character / style integration
    # -----------------------------------------------------------------

    def set_character_context(
        self,
        characters: list[CharacterDef],
        style: Optional[str] = None,
    ) -> None:
        """
        Configure characters that may appear in scenes.

        When a scene references characters, their visual descriptions and
        the chosen sketch style will be injected into the image prompt.
        """
        self._characters = {c.name.lower(): c for c in characters}
        self._character_style = style

    def set_visual_style(self, visual_style: str) -> None:
        """
        Set the visual / art style detected from the script.

        When set, this overrides the default cinematic style suffix
        so that all generated images match the source material
        (e.g. anime style for Attack on Titan, game art for Far Cry).
        """
        self._visual_style = visual_style.strip()

    def _load_model(self) -> None:
        """Load SDXL pipeline onto GPU."""
        if self._pipe is not None:
            return

        console.print("[cyan]Loading SDXL image generation model...[/cyan]")
        log_vram("before SDXL load")

        from diffusers import StableDiffusionXLPipeline  # type: ignore[import-untyped]

        dtype = torch.float16 if self.config.half_precision else torch.float32

        self._pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if self.config.half_precision else None,
        )
        self._pipe.to(self.config.device)

        # Enable memory optimizations
        self._pipe.enable_attention_slicing()
        try:
            self._pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass  # xformers not available, skip

        if self.use_refiner:
            from diffusers import StableDiffusionXLImg2ImgPipeline  # type: ignore[import-untyped]

            self._refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                self.refiner_id,
                torch_dtype=dtype,
                use_safetensors=True,
                variant="fp16" if self.config.half_precision else None,
            )
            self._refiner.to(self.config.device)

        log_vram("after SDXL load")
        console.print("[green]✓ SDXL model loaded[/green]")

    def unload(self) -> None:
        """Unload SDXL models and free VRAM."""
        if self._pipe is not None:
            console.print("[dim]Unloading SDXL model...[/dim]")
            unload_model(self._pipe)
            self._pipe = None
        if self._refiner is not None:
            unload_model(self._refiner)
            self._refiner = None
        free_vram()
        log_vram("after SDXL unload")

    def _get_dimensions(self, platform: Platform) -> tuple[int, int]:
        """Get image dimensions based on platform."""
        presets = self.config.output.get("presets", {})
        preset_name = platform.value
        if preset_name in presets:
            w = presets[preset_name].get("width", 1024)
            h = presets[preset_name].get("height", 1024)
            # SDXL works best with dimensions divisible by 8
            # Scale down for generation, upscale later in video assembly
            max_dim = 1024
            aspect = w / h
            if w > h:
                gen_w = max_dim
                gen_h = int(max_dim / aspect)
            else:
                gen_h = max_dim
                gen_w = int(max_dim * aspect)
            # Ensure divisible by 8
            gen_w = (gen_w // 8) * 8
            gen_h = (gen_h // 8) * 8
            return gen_w, gen_h
        return 1024, 1024

    async def generate_scene_images(
        self,
        parsed_script: ParsedScript,
        output_dir: str,
    ) -> List[str]:
        """
        Generate an image for each scene.

        Includes per-image retry logic and periodic VRAM cleanup to handle
        heavy scene loads (20+ scenes) without CUDA OOM or fragmentation.

        Args:
            parsed_script: The parsed script with image prompts
            output_dir: Directory to save images

        Returns:
            List of paths to generated images
        """
        self._load_model()
        os.makedirs(output_dir, exist_ok=True)

        width, height = self._get_dimensions(parsed_script.platform)
        image_files: List[str] = []

        max_retries_per_image = 3

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Generating images...",
                total=len(parsed_script.scenes),
            )

            for scene in parsed_script.scenes:
                filename = f"scene_{scene.scene_number:03d}_image.png"
                filepath = str(Path(output_dir) / filename)

                progress.update(
                    task,
                    description=(
                        f"Image: Scene {scene.scene_number}"
                        f"/{parsed_script.scene_count}"
                    ),
                )

                # Build character-aware prompt
                prompt = self._build_scene_prompt(scene)

                # --- Per-image retry with VRAM recovery ---
                last_err: Optional[Exception] = None
                for attempt in range(1, max_retries_per_image + 1):
                    try:
                        self._generate_single_image(
                            prompt=prompt,
                            output_path=filepath,
                            width=width,
                            height=height,
                        )
                        break  # success
                    except RuntimeError as exc:
                        last_err = exc
                        err_msg = str(exc).lower()
                        console.print(
                            f"[yellow]⚠ Image gen scene {scene.scene_number} "
                            f"failed (attempt {attempt}/{max_retries_per_image}): "
                            f"{exc}[/yellow]"
                        )
                        if "cuda" in err_msg or "out of memory" in err_msg:
                            # VRAM fragmentation — reload model for clean state
                            console.print("[yellow]Reloading SDXL for clean VRAM...[/yellow]")
                            self.unload()
                            self._load_model()
                        else:
                            free_vram()
                else:
                    raise RuntimeError(
                        f"Image generation failed for scene {scene.scene_number} "
                        f"after {max_retries_per_image} attempts: {last_err}"
                    )

                image_files.append(filepath)
                scene.image_path = filepath

                # Periodic VRAM cleanup to prevent fragmentation on long runs
                free_vram()
                progress.advance(task)

        console.print(f"[green]✓ Generated {len(image_files)} images[/green]")

        if self.config.auto_unload:
            self.unload()

        return image_files

    def _generate_single_image(
        self,
        prompt: str,
        output_path: str,
        width: int = 1024,
        height: int = 1024,
    ) -> None:
        """Generate a single image from a prompt."""
        if self._pipe is None:
            raise RuntimeError("SDXL model not loaded.")

        # Enhance prompt — use sketch style when characters are configured
        enhanced_prompt = self._enhance_prompt(prompt)

        with torch.no_grad():
            if self.use_refiner and self._refiner is not None:
                # Two-stage: base + refiner
                image = self._pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=self.negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    output_type="latent",
                    denoising_end=0.8,
                ).images[0]

                image = self._refiner(
                    prompt=enhanced_prompt,
                    negative_prompt=self.negative_prompt,
                    image=image,
                    num_inference_steps=self.num_inference_steps,
                    denoising_start=0.8,
                ).images[0]
            else:
                # Single-stage base only
                image = self._pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=self.negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                ).images[0]

        image.save(output_path)

    # -----------------------------------------------------------------
    # Prompt helpers
    # -----------------------------------------------------------------

    # Known sketch styles (mirrors character_designer.py SKETCH_STYLES)
    SKETCH_STYLE_SUFFIXES: dict[str, str] = {
        "stick_figure": (
            "stick figure drawing, simple line art, black lines on white background, "
            "minimalist, hand-drawn sketch, thin clean lines"
        ),
        "cartoon": (
            "cartoon style illustration, bold outlines, flat colors, "
            "comic book style, expressive, vibrant, animated look"
        ),
        "manga": (
            "manga style illustration, Japanese comic art, screentone shading, "
            "expressive eyes, dynamic poses, black and white ink"
        ),
        "doodle": (
            "doodle art style, hand-drawn, casual sketch, pen drawing, "
            "playful, loose lines, notebook doodle"
        ),
        "whiteboard": (
            "whiteboard drawing, marker sketch, clean lines, simple shapes, "
            "educational illustration, black marker on white"
        ),
    }

    def _build_scene_prompt(self, scene: Any) -> str:
        """
        Build a complete image prompt for a scene, injecting character
        descriptions and actions when characters are present.
        """
        parts: list[str] = [scene.image_prompt]

        # Inject referenced character descriptions
        char_names = getattr(scene, "characters_in_scene", [])
        if char_names:
            descs = []
            for name in char_names:
                cdef = self._characters.get(name.lower())
                if cdef:
                    traits = ", ".join(cdef.traits) if cdef.traits else ""
                    desc_text = cdef.description
                    if traits:
                        desc_text += f" ({traits})"
                    descs.append(f"{cdef.name}: {desc_text}")
            if descs:
                parts.append("Characters: " + "; ".join(descs))

        # Inject character actions / emotions
        actions = getattr(scene, "character_actions", None)
        emotions = getattr(scene, "character_emotions", None)
        if actions:
            parts.append(f"Action: {actions}")
        if emotions:
            parts.append(f"Emotion: {emotions}")

        return ", ".join(parts)

    def _enhance_prompt(self, prompt: str) -> str:
        """
        Add quality / style suffix to a prompt.

        Priority order:
          1. Explicit character sketch style (stick_figure, cartoon, manga, etc.)
          2. Visual style detected from the script (anime, game art, etc.)
          3. Default cinematic style
        """
        # 1. Explicit sketch style from character designer
        if self._character_style and self._character_style in self.SKETCH_STYLE_SUFFIXES:
            style_suffix = self.SKETCH_STYLE_SUFFIXES[self._character_style]
            return f"{prompt}, {style_suffix}"

        # 2. Visual style detected from script context (known franchise/IP)
        if self._visual_style:
            return (
                f"{prompt}, {self._visual_style}, "
                f"high quality, detailed, masterpiece"
            )

        # 3. Default cinematic style (generic / original content)
        return (
            f"{prompt}, cinematic lighting, high quality, detailed, "
            f"8k resolution, professional photography"
        )

    def generate_single(
        self,
        prompt: str,
        output_path: str,
        width: int = 1024,
        height: int = 1024,
    ) -> str:
        """Utility: generate a single image (loads/unloads model)."""
        self._load_model()
        self._generate_single_image(prompt, output_path, width, height)
        if self.config.auto_unload:
            self.unload()
        return output_path
