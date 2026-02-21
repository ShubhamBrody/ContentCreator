"""
ContentCreator - Video Generator

Converts scene images / prompts into video clips. Supports:
  1. Image Motion  (Advanced Ken Burns — cinematic zoom/pan/drift on AI images,
                    full-duration, CPU + GPU-accelerated resize, DEFAULT)
  2. AnimateDiff  (AI text-to-video, 24fps smooth motion, ~8-10 GB VRAM)
  3. Stable Video Diffusion  (AI image-to-video, ~12 GB VRAM)

The default engine is "image_motion" because it produces full-duration clips
that exactly match the narration audio, using the AI-generated scene images.
AnimateDiff generates only a limited number of frames (typically ~1 second)
and must be looped to fill longer narrations, causing visible repetition.
"""

from __future__ import annotations

import gc
import os
import random
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config import Config
from src.gpu_utils import free_vram, log_vram, unload_model
from src.models.schemas import ParsedScript

console = Console()


# ---------------------------------------------------------------------------
# Motion presets for the advanced Ken Burns engine
# ---------------------------------------------------------------------------

class MotionPreset(Enum):
    """Cinematic camera motion types for Ken Burns effect."""
    ZOOM_IN = "zoom_in"               # Slow push-in (dramatic focus)
    ZOOM_OUT = "zoom_out"             # Slow pull-out (reveal)
    PAN_LEFT = "pan_left"             # Horizontal drift left
    PAN_RIGHT = "pan_right"           # Horizontal drift right
    PAN_UP = "pan_up"                 # Tilt up (establishing shot)
    PAN_DOWN = "pan_down"             # Tilt down (reveal)
    DRIFT_ZOOM_IN = "drift_zoom_in"   # Pan + slow zoom in
    DRIFT_ZOOM_OUT = "drift_zoom_out" # Pan + slow zoom out


# Cycle through these so consecutive scenes get different motions
_MOTION_CYCLE: list[MotionPreset] = [
    MotionPreset.ZOOM_IN,
    MotionPreset.PAN_RIGHT,
    MotionPreset.DRIFT_ZOOM_OUT,
    MotionPreset.PAN_LEFT,
    MotionPreset.ZOOM_OUT,
    MotionPreset.DRIFT_ZOOM_IN,
    MotionPreset.PAN_UP,
    MotionPreset.PAN_DOWN,
]


def _ease_in_out(t: float) -> float:
    """Smooth ease-in-out curve (cubic Hermite)."""
    return t * t * (3.0 - 2.0 * t)


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
            "motion_adapter", "guoyww/animatediff-motion-adapter-v1-5-2"
        )
        self.ad_num_frames = ad_config.get("num_frames", 16)
        self.ad_fps = ad_config.get("fps", 24)
        self.ad_guidance_scale = ad_config.get("guidance_scale", 7.5)
        self.ad_num_inference_steps = ad_config.get("num_inference_steps", 25)
        self.ad_width = ad_config.get("width", 512)
        self.ad_height = ad_config.get("height", 768)

        # --- Advanced Image Motion config (primary) ---
        im_config = config.video.get("image_motion", {})
        self.zoom_range: list[float] = im_config.get("zoom_range", [1.0, 1.20])
        self.pan_range: list[float] = im_config.get("pan_range", [-0.08, 0.08])
        self.duration_per_scene: float = im_config.get("duration_per_scene", 5)
        self.motion_fps: int = im_config.get("fps", 24)

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
        audio_files: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate video clips for each scene.

        Args:
            parsed_script: Parsed script with scenes
            output_dir: Directory to save video clips
            audio_files: Optional list of per-scene audio paths. When
                         provided, each video clip's duration is matched
                         exactly to the corresponding audio duration, so
                         the assembler never needs to loop or trim.

        Returns:
            List of paths to generated video clips
        """
        os.makedirs(output_dir, exist_ok=True)
        video_files: List[str] = []

        # Pre-compute audio durations so video clips match narration exactly
        audio_durations: list[Optional[float]] = []
        if audio_files:
            from moviepy import AudioFileClip  # type: ignore[import-untyped]
            for af in audio_files:
                if af and os.path.exists(af):
                    try:
                        aclip = AudioFileClip(af)
                        audio_durations.append(aclip.duration)
                        aclip.close()
                    except Exception:
                        audio_durations.append(None)
                else:
                    audio_durations.append(None)

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

            for idx, scene in enumerate(parsed_script.scenes):
                filename = f"scene_{scene.scene_number:03d}_video.mp4"
                filepath = str(Path(output_dir) / filename)

                progress.update(
                    task,
                    description=(
                        f"Video: Scene {scene.scene_number}"
                        f"/{parsed_script.scene_count} [{self.engine}]"
                    ),
                )

                # Use audio duration if available, otherwise script duration
                if idx < len(audio_durations) and audio_durations[idx] is not None:
                    duration = float(audio_durations[idx])  # type: ignore[arg-type]
                else:
                    duration = float(scene.duration_seconds)

                # Minimum 1.5 s to avoid degenerate clips
                duration = max(duration, 1.5)

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
                    # Advanced Ken Burns — cinematic motion on AI image
                    if scene.image_path is None:
                        raise ValueError(
                            f"Scene {scene.scene_number} has no image. "
                            "Run image generation first."
                        )
                    motion = _MOTION_CYCLE[idx % len(_MOTION_CYCLE)]
                    self._generate_motion_clip(
                        scene.image_path, filepath, duration, motion
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
    # Image Motion — Advanced Ken Burns (cinematic zoom / pan / drift)
    # =========================================================================

    def _generate_motion_clip(
        self,
        image_path: str,
        output_path: str,
        duration: float,
        motion: MotionPreset = MotionPreset.ZOOM_IN,
    ) -> None:
        """
        Generate a high-quality Ken Burns video clip from an image.

        Uses smooth ease-in-out curves and configurable motion presets
        so consecutive scenes feel visually distinct. The clip duration
        matches the narration audio exactly.
        """
        from moviepy import ImageClip  # type: ignore[import-untyped]
        from PIL import Image as PILImage

        image = PILImage.open(image_path).convert("RGB")

        # Up-scale the source image so crop regions stay sharp after resize
        # We work at 2× internal resolution then down-scale when writing frames
        base_w, base_h = image.size
        work_scale = 2.0
        work_w = int(base_w * work_scale)
        work_h = int(base_h * work_scale)
        image = image.resize((work_w, work_h), PILImage.LANCZOS)
        img_array = np.array(image)

        # ----- Motion parameters per preset -----
        zoom_start = 1.0
        zoom_end = 1.0
        pan_x_start = 0.0
        pan_x_end = 0.0
        pan_y_start = 0.0
        pan_y_end = 0.0

        # Zoom range (fraction of image)
        z_lo, z_hi = self.zoom_range  # e.g. [1.0, 1.20]
        p_lo, p_hi = self.pan_range   # e.g. [-0.08, 0.08]

        if motion == MotionPreset.ZOOM_IN:
            zoom_start, zoom_end = 1.0, random.uniform(z_hi * 0.8, z_hi)
        elif motion == MotionPreset.ZOOM_OUT:
            zoom_start, zoom_end = random.uniform(z_hi * 0.8, z_hi), 1.0
        elif motion == MotionPreset.PAN_LEFT:
            pan_x_start = random.uniform(0.02, abs(p_hi))
            pan_x_end = -pan_x_start
        elif motion == MotionPreset.PAN_RIGHT:
            pan_x_start = -random.uniform(0.02, abs(p_hi))
            pan_x_end = -pan_x_start
        elif motion == MotionPreset.PAN_UP:
            pan_y_start = random.uniform(0.02, abs(p_hi))
            pan_y_end = -pan_y_start
        elif motion == MotionPreset.PAN_DOWN:
            pan_y_start = -random.uniform(0.02, abs(p_hi))
            pan_y_end = -pan_y_start
        elif motion == MotionPreset.DRIFT_ZOOM_IN:
            zoom_start, zoom_end = 1.0, random.uniform(z_hi * 0.7, z_hi)
            pan_x_start = random.uniform(p_lo * 0.5, p_hi * 0.5)
            pan_x_end = -pan_x_start
            pan_y_start = random.uniform(p_lo * 0.3, p_hi * 0.3)
            pan_y_end = -pan_y_start
        elif motion == MotionPreset.DRIFT_ZOOM_OUT:
            zoom_start, zoom_end = random.uniform(z_hi * 0.7, z_hi), 1.0
            pan_x_start = random.uniform(p_lo * 0.5, p_hi * 0.5)
            pan_x_end = -pan_x_start
            pan_y_start = random.uniform(p_lo * 0.3, p_hi * 0.3)
            pan_y_end = -pan_y_start

        h, w = img_array.shape[:2]
        out_w, out_h = base_w, base_h  # final frame dimensions

        def make_frame(t: float) -> np.ndarray:
            """Produce one frame with smooth easing at time t."""
            raw_t = t / max(duration, 0.01)
            ease_t = _ease_in_out(raw_t)

            # Interpolate zoom & pan
            zoom = zoom_start + (zoom_end - zoom_start) * ease_t
            px = pan_x_start + (pan_x_end - pan_x_start) * ease_t
            py = pan_y_start + (pan_y_end - pan_y_start) * ease_t

            # Crop region (smaller crop = more zoom)
            crop_w = int(w / max(zoom, 1.001))
            crop_h = int(h / max(zoom, 1.001))

            # Center with pan offset
            cx = w // 2 + int(px * w)
            cy = h // 2 + int(py * h)

            x1 = max(0, cx - crop_w // 2)
            y1 = max(0, cy - crop_h // 2)
            x2 = min(w, x1 + crop_w)
            y2 = min(h, y1 + crop_h)

            # Clamp to avoid zero-size crops
            if x2 - x1 < 2:
                x1, x2 = 0, w
            if y2 - y1 < 2:
                y1, y2 = 0, h

            # Re-adjust if we hit the border
            if x2 - x1 < crop_w:
                x1 = max(0, x2 - crop_w)
            if y2 - y1 < crop_h:
                y1 = max(0, y2 - crop_h)

            cropped = img_array[y1:y2, x1:x2]

            # Resize to output dimensions using high-quality resampling
            frame_pil = PILImage.fromarray(cropped).resize(
                (out_w, out_h), PILImage.LANCZOS
            )
            return np.array(frame_pil)

        # Build the clip at the exact narration duration using VideoClip
        from moviepy import VideoClip  # type: ignore[import-untyped]

        motion_clip = VideoClip(make_frame, duration=duration)

        console.print(
            f"[dim]Ken Burns ({motion.value}): {duration:.1f}s "
            f"@ {self.motion_fps}fps, {out_w}×{out_h}[/dim]"
        )

        motion_clip.write_videofile(
            output_path,
            fps=self.motion_fps,
            codec="libx264",
            audio=False,
            logger=None,
        )
        motion_clip.close()

        # Eagerly free the 2× upscaled image array to prevent RAM creep
        # across many scenes (the make_frame closure holds a reference).
        del img_array, image
        gc.collect()

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
        from moviepy import ImageSequenceClip  # type: ignore[import-untyped]

        frame_arrays = [np.array(f) for f in frames]
        clip = ImageSequenceClip(frame_arrays, fps=fps)
        clip.write_videofile(
            output_path,
            codec="libx264",
            audio=False,
            logger=None,
        )
        clip.close()
