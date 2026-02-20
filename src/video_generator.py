"""
ContentCreator - Video Generator

Converts scene images / prompts into video clips. Supports:
  1. AnimateDiff  (AI text-to-video, 24fps smooth motion, ~8-10 GB VRAM)
  2. Stable Video Diffusion  (AI image-to-video, ~12 GB VRAM)
  3. Image Motion  (Ken Burns zoom/pan, CPU-friendly fallback)
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config import Config
from src.gpu_utils import free_vram, log_vram, unload_model
from src.models.schemas import ParsedScript

console = Console()


class VideoGenerator:
    """Generates video clips from scene images or prompts."""

    def __init__(self, config: Config):
        self.config = config
        self.engine = config.video.get("engine", "image_motion")

        # --- SVD settings ---
        self.model_id = config.video.get(
            "model", "stabilityai/stable-video-diffusion-img2vid-xt"
        )
        self.num_frames = config.video.get("num_frames", 25)
        self.fps = config.video.get("fps", 8)
        self.motion_bucket_id = config.video.get("motion_bucket_id", 127)
        self.noise_aug_strength = config.video.get("noise_aug_strength", 0.02)
        self.decode_chunk_size = config.video.get("decode_chunk_size", 8)

        # --- AnimateDiff settings ---
        ad_config = config.video.get("animatediff", {})
        self.ad_base_model = ad_config.get(
            "base_model", "SG161222/Realistic_Vision_V5.1_noVAE"
        )
        self.ad_motion_adapter = ad_config.get(
            "motion_adapter", "guanzhi/animatediff-motion-adapter-v1-5-3"
        )
        self.ad_num_frames = ad_config.get("num_frames", 16)
        self.ad_fps = ad_config.get("fps", 24)
        self.ad_guidance_scale = ad_config.get("guidance_scale", 7.5)
        self.ad_num_inference_steps = ad_config.get("num_inference_steps", 25)
        self.ad_width = ad_config.get("width", 512)
        self.ad_height = ad_config.get("height", 768)

        # --- Image motion config (fallback) ---
        im_config = config.video.get("image_motion", {})
        self.zoom_range: list[float] = im_config.get("zoom_range", [1.0, 1.15])
        self.pan_range: list[float] = im_config.get("pan_range", [-0.05, 0.05])
        self.duration_per_scene: float = im_config.get("duration_per_scene", 5)

        self._pipe: Any = None
        self._motion_adapter: Any = None

    # =========================================================================
    # Model management
    # =========================================================================

    def _load_animatediff(self) -> None:
        """Load AnimateDiff pipeline (SD 1.5 base + motion adapter)."""
        if self._pipe is not None:
            return

        console.print("[cyan]Loading AnimateDiff motion adapter...[/cyan]")
        log_vram("before AnimateDiff load")

        from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler

        dtype = torch.float16 if self.config.half_precision else torch.float32

        # 1. Load motion adapter weights
        self._motion_adapter = MotionAdapter.from_pretrained(
            self.ad_motion_adapter, torch_dtype=dtype
        )

        # 2. Build pipeline with SD 1.5 base + adapter
        console.print(f"[cyan]Loading base model: {self.ad_base_model}[/cyan]")
        self._pipe = AnimateDiffPipeline.from_pretrained(
            self.ad_base_model,
            motion_adapter=self._motion_adapter,
            torch_dtype=dtype,
        )

        # 3. Configure scheduler for stable generation
        self._pipe.scheduler = DDIMScheduler.from_pretrained(
            self.ad_base_model,
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )

        # 4. VRAM optimizations
        self._pipe.enable_vae_slicing()
        self._pipe.enable_vae_tiling()
        try:
            self._pipe.enable_model_cpu_offload()
            console.print("[dim]Using CPU offloading for AnimateDiff[/dim]")
        except Exception:
            self._pipe.to(self.config.device)

        log_vram("after AnimateDiff load")
        console.print("[green]✓ AnimateDiff model loaded[/green]")

    def _load_svd(self) -> None:
        """Load Stable Video Diffusion pipeline."""
        if self._pipe is not None:
            return

        console.print("[cyan]Loading Stable Video Diffusion model...[/cyan]")
        log_vram("before SVD load")

        from diffusers import StableVideoDiffusionPipeline  # type: ignore[import-untyped]

        dtype = torch.float16 if self.config.half_precision else torch.float32

        self._pipe = StableVideoDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            variant="fp16" if self.config.half_precision else None,
        )
        self._pipe.to(self.config.device)

        # Memory optimizations
        self._pipe.enable_attention_slicing()
        try:
            self._pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        log_vram("after SVD load")
        console.print("[green]✓ SVD model loaded[/green]")

    def unload(self) -> None:
        """Unload video model and free VRAM."""
        if self._pipe is not None:
            console.print("[dim]Unloading video model...[/dim]")
            unload_model(self._pipe)
            self._pipe = None
        if self._motion_adapter is not None:
            unload_model(self._motion_adapter)
            self._motion_adapter = None
        free_vram()
        log_vram("after video model unload")

    # =========================================================================
    # Public API
    # =========================================================================

    async def generate_scene_videos(
        self,
        parsed_script: ParsedScript,
        output_dir: str,
    ) -> List[str]:
        """
        Generate video clips for each scene.

        Args:
            parsed_script: Parsed script with scenes
            output_dir: Directory to save video clips

        Returns:
            List of paths to generated video clips
        """
        os.makedirs(output_dir, exist_ok=True)
        video_files: List[str] = []

        # Choose engine and load model
        if self.engine == "animatediff":
            self._load_animatediff()
        elif self.engine == "svd":
            self._load_svd()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Generating video clips...",
                total=len(parsed_script.scenes),
            )

            for scene in parsed_script.scenes:
                filename = f"scene_{scene.scene_number:03d}_video.mp4"
                filepath = str(Path(output_dir) / filename)

                progress.update(
                    task,
                    description=(
                        f"Video: Scene {scene.scene_number}"
                        f"/{parsed_script.scene_count} [{self.engine}]"
                    ),
                )

                duration = scene.duration_seconds

                if self.engine == "animatediff":
                    # AnimateDiff: text → animated video
                    prompt = getattr(scene, "image_prompt", None) or scene.narration
                    self._generate_animatediff_clip(prompt, filepath, duration)
                elif self.engine == "svd":
                    # SVD: image → video (requires image_path)
                    if scene.image_path is None:
                        raise ValueError(
                            f"Scene {scene.scene_number} has no image. "
                            "Run image generation first."
                        )
                    self._generate_svd_clip(scene.image_path, filepath)
                else:
                    # Ken Burns fallback
                    if scene.image_path is None:
                        raise ValueError(
                            f"Scene {scene.scene_number} has no image. "
                            "Run image generation first."
                        )
                    self._generate_motion_clip(
                        scene.image_path, filepath, duration
                    )

                video_files.append(filepath)
                scene.video_clip_path = filepath
                progress.advance(task)

        console.print(f"[green]✓ Generated {len(video_files)} video clips[/green]")

        if self.engine in ("svd", "animatediff") and self.config.auto_unload:
            self.unload()

        return video_files

    # =========================================================================
    # AnimateDiff — AI text-to-animated-video (24fps smooth motion)
    # =========================================================================

    def _generate_animatediff_clip(
        self,
        prompt: str,
        output_path: str,
        target_duration: float,
    ) -> None:
        """Generate an animated video clip from a text prompt using AnimateDiff."""
        if self._pipe is None:
            raise RuntimeError("AnimateDiff model not loaded.")

        # Enhance the prompt for cinematic motion
        enhanced_prompt = (
            f"{prompt}, cinematic, smooth camera motion, "
            "high quality, detailed, 4k, professional lighting"
        )
        negative_prompt = (
            "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
            "static, frozen, jittery, noise, watermark, text"
        )

        console.print(f"[dim]AnimateDiff: {self.ad_num_frames} frames "
                       f"@ {self.ad_width}x{self.ad_height}[/dim]")

        with torch.no_grad():
            output = self._pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                num_frames=self.ad_num_frames,
                width=self.ad_width,
                height=self.ad_height,
                guidance_scale=self.ad_guidance_scale,
                num_inference_steps=self.ad_num_inference_steps,
            )
            frames = output.frames[0]

        # Calculate fps so the raw clip length ≈ target_duration
        # AnimateDiff gives us ad_num_frames frames.
        # E.g. 16 frames / 5s ≈ 3.2 fps is too slow. We write at ad_fps and
        # let the assembler loop / trim to match scene duration.
        self._frames_to_video(frames, output_path, self.ad_fps)

    # =========================================================================
    # SVD — AI-generated motion from image
    # =========================================================================

    def _generate_svd_clip(self, image_path: str, output_path: str) -> None:
        """Generate a video clip from an image using Stable Video Diffusion."""
        if self._pipe is None:
            raise RuntimeError("SVD model not loaded.")

        image = Image.open(image_path).convert("RGB")
        image = image.resize((1024, 576))  # SVD expects this resolution

        with torch.no_grad():
            frames = self._pipe(
                image,
                num_frames=self.num_frames,
                motion_bucket_id=self.motion_bucket_id,
                noise_aug_strength=self.noise_aug_strength,
                decode_chunk_size=self.decode_chunk_size,
            ).frames[0]

        self._frames_to_video(frames, output_path, self.fps)

    # =========================================================================
    # Image Motion — Ken Burns effect (CPU-friendly fallback)
    # =========================================================================

    def _generate_motion_clip(
        self,
        image_path: str,
        output_path: str,
        duration: float,
    ) -> None:
        """Generate a video clip with Ken Burns (zoom/pan) effect."""
        from moviepy.editor import ImageClip  # type: ignore[import-untyped]

        image = Image.open(image_path).convert("RGB")
        img_array = np.array(image)

        # Random motion parameters
        start_zoom = self.zoom_range[0]
        end_zoom = random.uniform(self.zoom_range[0], self.zoom_range[1])
        pan_x = random.uniform(self.pan_range[0], self.pan_range[1])
        pan_y = random.uniform(self.pan_range[0], self.pan_range[1])

        h, w = img_array.shape[:2]

        def make_frame(t: float) -> np.ndarray:
            """Create a frame at time t with zoom/pan effect."""
            progress_ratio = t / max(duration, 0.01)

            # Interpolate zoom
            zoom = start_zoom + (end_zoom - start_zoom) * progress_ratio

            # Calculate crop region
            crop_w = int(w / zoom)
            crop_h = int(h / zoom)

            # Center + pan offset
            cx = w // 2 + int(pan_x * w * progress_ratio)
            cy = h // 2 + int(pan_y * h * progress_ratio)

            # Clamp to image bounds
            x1 = max(0, cx - crop_w // 2)
            y1 = max(0, cy - crop_h // 2)
            x2 = min(w, x1 + crop_w)
            y2 = min(h, y1 + crop_h)

            # Adjust if we hit the edge
            if x2 - x1 < crop_w:
                x1 = max(0, x2 - crop_w)
            if y2 - y1 < crop_h:
                y1 = max(0, y2 - crop_h)

            cropped = img_array[y1:y2, x1:x2]

            # Resize back to original dimensions
            from PIL import Image as PILImage

            frame = PILImage.fromarray(cropped).resize((w, h), PILImage.LANCZOS)
            return np.array(frame)

        clip = ImageClip(img_array).set_duration(duration).set_fps(30)
        clip = clip.fl(lambda gf, t: make_frame(t), apply_to=["mask"])

        # Override get_frame directly for Ken Burns
        clip = clip.set_make_frame(make_frame)

        clip.write_videofile(
            output_path,
            fps=30,
            codec="libx264",
            audio=False,
            logger=None,
        )
        clip.close()

    # =========================================================================
    # Utility
    # =========================================================================

    @staticmethod
    def _frames_to_video(
        frames: list[Any],
        output_path: str,
        fps: int,
    ) -> None:
        """Save a list of PIL images as a video file."""
        from moviepy.editor import ImageSequenceClip  # type: ignore[import-untyped]

        frame_arrays = [np.array(f) for f in frames]
        clip = ImageSequenceClip(frame_arrays, fps=fps)
        clip.write_videofile(
            output_path,
            codec="libx264",
            audio=False,
            logger=None,
        )
        clip.close()
