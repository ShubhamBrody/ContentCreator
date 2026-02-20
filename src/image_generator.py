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
from src.models.schemas import ParsedScript, Platform

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
        self._pipe: Any = None
        self._refiner: Any = None

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

                self._generate_single_image(
                    prompt=scene.image_prompt,
                    output_path=filepath,
                    width=width,
                    height=height,
                )

                image_files.append(filepath)
                scene.image_path = filepath
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

        # Enhance prompt for cinematic quality
        enhanced_prompt = (
            f"{prompt}, cinematic lighting, high quality, detailed, "
            f"8k resolution, professional photography"
        )

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
