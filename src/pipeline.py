"""
ContentCreator - Pipeline Orchestrator

Orchestrates the full video creation pipeline:
  Script → Parse → [TTS ‖ Images] → Video → Music → Subtitles → Assemble

Runs TTS and Image generation concurrently (TTS = network I/O,
Images = GPU, no resource conflict). Manages VRAM by loading/unloading
models sequentially for GPU-heavy stages.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.character_designer import CharacterDesigner, CharacterProfile
from src.checkpoint import CheckpointManager
from src.config import Config
from src.image_generator import ImageGenerator
from src.models.schemas import CharacterDef, ParsedScript, PipelineArtifacts, Platform
from src.music_generator import MusicGenerator
from src.script_parser import ScriptParser
from src.subtitle_generator import SubtitleGenerator, SubtitleSegment
from src.tts_engine import TTSEngine
from src.video_assembler import VideoAssembler
from src.video_generator import VideoGenerator

console = Console()


class Pipeline:
    """Orchestrates the full content creation pipeline."""

    def __init__(self, config: Config):
        self.config = config
        self.script_parser = ScriptParser(config)
        self.tts_engine = TTSEngine(config)
        self.image_generator = ImageGenerator(config)
        self.video_generator = VideoGenerator(config)
        self.music_generator = MusicGenerator(config)
        self.subtitle_generator = SubtitleGenerator(config)
        self.video_assembler = VideoAssembler(config)
        self.character_designer = CharacterDesigner(config)

    async def run(
        self,
        script: str,
        platform: Platform = Platform.REELS,
        output_name: Optional[str] = None,
        num_scenes: Optional[int] = None,
        characters: Optional[List[dict]] = None,
        character_style: Optional[str] = None,
        progress_callback: Optional[Callable[..., Coroutine[Any, Any, None]]] = None,
        resume_dir: Optional[str] = None,
    ) -> str:
        """
        Run the full pipeline: script → final video.

        Args:
            script: Raw script text or video idea
            platform: Target platform (youtube, reels, shorts)
            output_name: Optional custom name for the output
            num_scenes: Optional override for number of scenes
            characters: Optional list of character dicts with keys:
                        name, description, photo_path (optional), traits (optional)
            character_style: Sketch style (stick_figure, cartoon, manga, doodle, whiteboard)
            progress_callback: Optional async callback(stage, status, message)
                               for reporting progress to a frontend.
            resume_dir: If set, resume from the checkpoint in this project directory.

        Returns:
            Path to the final video file
        """

        async def _report(stage: str, status: str, message: str = "") -> None:
            if progress_callback:
                await progress_callback(stage, status, message)
        start_time = time.time()
        stages = self.config.pipeline.get("stages", [])

        # ---- Checkpoint / resume bookkeeping ----
        resumed_stages: set = set()
        cached_arts: dict = {}
        parsed_script: Optional[ParsedScript] = None

        if resume_dir and os.path.isdir(resume_dir):
            project_dir = resume_dir
            project_name = os.path.basename(project_dir)
            ckpt_mgr = CheckpointManager(project_dir)
            ckpt = ckpt_mgr.load()
            if ckpt and ckpt_mgr.validate(ckpt):
                resumed_stages = set(ckpt.get("completed_stages", []))
                cached_arts = ckpt.get("artifacts", {})
                console.print(
                    f"[yellow]Resuming — skipping {len(resumed_stages)} completed "
                    f"stages: {', '.join(resumed_stages)}[/yellow]"
                )
        else:
            # Fresh run — create project directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_name = output_name or f"video_{timestamp}"
            output_dir = self.config.output.get("directory", "output")
            project_dir = str(Path(output_dir) / project_name)
            ckpt_mgr = CheckpointManager(project_dir)

        os.makedirs(project_dir, exist_ok=True)
        temp_dir = str(Path(project_dir) / "temp")
        os.makedirs(temp_dir, exist_ok=True)

        artifacts = PipelineArtifacts(project_dir=project_dir)

        # Params dict to persist in checkpoint
        _ckpt_params = {
            "script": script,
            "platform": platform.value,
            "num_scenes": num_scenes,
            "characters": characters,
            "character_style": character_style,
        }

        def _save_checkpoint(completed: List[str]) -> None:
            """Persist current progress to disk."""
            ckpt_mgr.save(
                params=_ckpt_params,
                completed_stages=completed,
                artifacts=artifacts.model_dump(),
                active_stages=stages,
            )

        completed_so_far: List[str] = list(resumed_stages)

        self._print_header(script, platform, project_dir)

        # =================================================================
        # Stage 1: Parse Script
        # =================================================================
        if "script_parse" in stages:
            if "script_parse" in resumed_stages:
                console.print("[dim]Stage 1/8: Script Parsing — using cache[/dim]")
                await _report("script_parse", "completed", "Restored from checkpoint")
                # Restore parsed_script from cached JSON file
                script_json_path = str(Path(project_dir) / "parsed_script.json")
                if os.path.isfile(script_json_path):
                    with open(script_json_path, "r", encoding="utf-8") as f:
                        parsed_script = ParsedScript.model_validate_json(f.read())
                    artifacts.parsed_script = parsed_script
                else:
                    raise RuntimeError("Checkpoint references parsed_script.json but file is missing.")
            else:
                console.print(Panel("[bold]Stage 1/8: Parsing Script[/bold]"))
                await _report("script_parse", "running", "Parsing script with LLM...")
                parsed_script = await self.script_parser.parse(
                    script, platform, num_scenes, characters=characters
                )
                artifacts.parsed_script = parsed_script

                # Save parsed script as JSON
                script_json_path = str(Path(project_dir) / "parsed_script.json")
                with open(script_json_path, "w", encoding="utf-8") as f:
                    f.write(parsed_script.model_dump_json(indent=2))

                self._print_scene_summary(parsed_script)
                completed_so_far.append("script_parse")
                _save_checkpoint(completed_so_far)
                await _report("script_parse", "completed", f"Parsed {parsed_script.scene_count} scenes")
        else:
            raise RuntimeError("script_parse stage is required.")

        # =================================================================
        # Stage 1.5: Character Design (sketches from photos / descriptions)
        # =================================================================
        char_style = character_style or self.config.characters.get(
            "default_style", "stick_figure"
        )

        if characters and "character_design" in stages:
            console.print(Panel("[bold]Stage 1.5/8: Designing Characters[/bold]"))
            await _report("character_design", "running", "Designing character sketches...")
            sketch_dir = str(Path(temp_dir) / "characters")
            os.makedirs(sketch_dir, exist_ok=True)

            char_defs: List[CharacterDef] = []
            sketch_paths: List[str] = []

            for char_info in characters:
                name = char_info.get("name", "character")
                description = char_info.get("description", "")
                photo_path = char_info.get("photo_path")
                traits = char_info.get("traits", [])

                sketch_filename = f"{name.lower().replace(' ', '_')}_sketch.png"
                sketch_out = str(Path(sketch_dir) / sketch_filename)

                if photo_path and os.path.isfile(photo_path):
                    # Photo → sketch conversion
                    console.print(
                        f"  [cyan]Converting photo to sketch: {name}[/cyan]"
                    )
                    profile = self.character_designer.photo_to_sketch(
                        photo_path=photo_path,
                        output_path=sketch_out,
                        character_name=name,
                        style=char_style,
                        description=description,
                    )
                else:
                    # Description → sketch generation
                    console.print(
                        f"  [cyan]Generating sketch from description: {name}[/cyan]"
                    )
                    profile = self.character_designer.description_to_sketch(
                        description=description,
                        output_path=sketch_out,
                        character_name=name,
                        style=char_style,
                    )

                char_def = CharacterDef(
                    name=name,
                    description=description,
                    photo_path=photo_path,
                    sketch_path=profile.sketch_path,
                    style=char_style,
                    traits=traits,
                )
                char_defs.append(char_def)
                if profile.sketch_path:
                    sketch_paths.append(profile.sketch_path)

            # Store character defs in parsed_script
            artifacts.parsed_script.characters = char_defs  # type: ignore[union-attr]
            artifacts.character_sketches = sketch_paths

            # Pass character context to image generator
            self.image_generator.set_character_context(char_defs, char_style)

            console.print(
                f"[green]✓ Created {len(sketch_paths)} character sketches[/green]"
            )
            completed_so_far.append("character_design")
            _save_checkpoint(completed_so_far)
            await _report("character_design", "completed", f"Created {len(sketch_paths)} character sketches")
        elif characters:
            # Characters defined but stage disabled — still pass style context
            char_defs_basic = [
                CharacterDef(
                    name=c.get("name", "character"),
                    description=c.get("description", ""),
                    traits=c.get("traits", []),
                    style=char_style,
                )
                for c in characters
            ]
            if artifacts.parsed_script:
                artifacts.parsed_script.characters = char_defs_basic
            self.image_generator.set_character_context(char_defs_basic, char_style)

        # Pass the visual style detected by the LLM to the image generator
        # so all images match the source material's art style
        if parsed_script and parsed_script.visual_style:
            self.image_generator.set_visual_style(parsed_script.visual_style)
            console.print(
                f"[cyan]Visual style: {parsed_script.visual_style}[/cyan]"
            )

        # =================================================================
        # Stages 2+3: TTS ‖ Image Generation (concurrent)
        # TTS uses network/CPU (edge-tts), Image Gen uses GPU (SDXL).
        # Running in parallel saves significant wall-clock time.
        # =================================================================
        audio_files: List[str] = []
        image_files: List[str] = []

        tts_needed = "tts" in stages
        image_gen_needed = "image_gen" in stages
        tts_cached = "tts" in resumed_stages
        img_cached = "image_gen" in resumed_stages

        # Restore cached file lists if resuming
        if tts_cached:
            audio_files = cached_arts.get("scene_audio_files", [])
            artifacts.scene_audio_files = audio_files
            console.print("[dim]Stage 2/8: Voice Generation — using cache[/dim]")
            await _report("tts", "completed", f"Restored {len(audio_files)} audio files from cache")
        if img_cached:
            image_files = cached_arts.get("scene_image_files", [])
            artifacts.scene_image_files = image_files
            console.print("[dim]Stage 3/8: Image Generation — using cache[/dim]")
            await _report("image_gen", "completed", f"Restored {len(image_files)} images from cache")

        tts_run = tts_needed and not tts_cached
        img_run = image_gen_needed and not img_cached

        if tts_run and img_run:
            console.print(
                Panel("[bold]Stages 2-3/8: Voice + Images (parallel)[/bold]")
            )
            audio_dir = str(Path(temp_dir) / "audio")
            image_dir = str(Path(temp_dir) / "images")

            await _report("tts", "running", "Generating voiceover audio...")
            await _report("image_gen", "running", "Generating scene images with SDXL...")

            # TTS is truly async (network I/O) — runs in current event loop
            async def _do_tts() -> List[str]:
                result = await self.tts_engine.generate_scene_audio(
                    parsed_script, audio_dir
                )
                await _report(
                    "tts", "completed", f"Generated {len(result)} audio files"
                )
                return result

            # Image gen does blocking GPU work — run in a thread so TTS
            # can proceed concurrently on the event loop.
            def _do_images_sync() -> List[str]:
                _loop = asyncio.new_event_loop()
                try:
                    return _loop.run_until_complete(
                        self.image_generator.generate_scene_images(
                            parsed_script, image_dir
                        )
                    )
                finally:
                    _loop.close()

            loop = asyncio.get_event_loop()
            tts_task = asyncio.ensure_future(_do_tts())
            image_future = loop.run_in_executor(None, _do_images_sync)

            audio_files, image_files = await asyncio.gather(
                tts_task, image_future
            )

            artifacts.scene_audio_files = audio_files
            artifacts.scene_image_files = image_files
            await _report(
                "image_gen", "completed",
                f"Generated {len(image_files)} images",
            )
            console.print(
                f"[green]✓ Parallel complete — "
                f"{len(audio_files)} audio, {len(image_files)} images[/green]"
            )
            completed_so_far.extend(["tts", "image_gen"])
            _save_checkpoint(completed_so_far)

        elif tts_run:
            console.print(Panel("[bold]Stage 2/8: Generating Voiceover[/bold]"))
            await _report("tts", "running", "Generating voiceover audio...")
            audio_dir = str(Path(temp_dir) / "audio")
            audio_files = await self.tts_engine.generate_scene_audio(
                parsed_script, audio_dir
            )
            artifacts.scene_audio_files = audio_files
            completed_so_far.append("tts")
            _save_checkpoint(completed_so_far)
            await _report(
                "tts", "completed", f"Generated {len(audio_files)} audio files"
            )

        elif img_run:
            console.print(Panel("[bold]Stage 3/8: Generating Images[/bold]"))
            await _report(
                "image_gen", "running", "Generating scene images with SDXL..."
            )
            image_dir = str(Path(temp_dir) / "images")
            image_files = await self.image_generator.generate_scene_images(
                parsed_script, image_dir
            )
            artifacts.scene_image_files = image_files
            completed_so_far.append("image_gen")
            _save_checkpoint(completed_so_far)
            await _report(
                "image_gen", "completed",
                f"Generated {len(image_files)} images",
            )

        # =================================================================
        # Stage 4: Video Generation
        # =================================================================
        video_files: List[str] = []
        if "video_gen" in stages:
            if "video_gen" in resumed_stages:
                video_files = cached_arts.get("scene_video_files", [])
                artifacts.scene_video_files = video_files
                console.print("[dim]Stage 4/8: Video Generation — using cache[/dim]")
                await _report("video_gen", "completed", f"Restored {len(video_files)} clips from cache")
            else:
                engine = self.config.video.get("engine", "image_motion")
                console.print(
                    Panel(f"[bold]Stage 4/8: Generating Video Clips [{engine}][/bold]")
                )
                await _report(
                    "video_gen", "running",
                    f"Generating video clips with {engine}...",
                )
                video_dir = str(Path(temp_dir) / "videos")
                video_files = await self.video_generator.generate_scene_videos(
                    parsed_script, video_dir, audio_files=audio_files
                )
                artifacts.scene_video_files = video_files
                completed_so_far.append("video_gen")
                _save_checkpoint(completed_so_far)
                await _report(
                    "video_gen", "completed",
                    f"Generated {len(video_files)} video clips",
                )

        # =================================================================
        # Stage 5: Music Generation
        # =================================================================
        music_path: Optional[str] = None
        if "music_gen" in stages:
            if "music_gen" in resumed_stages:
                music_path = cached_arts.get("music_file")
                if music_path:
                    artifacts.music_file = music_path
                console.print("[dim]Stage 5/8: Music Generation — using cache[/dim]")
                await _report("music_gen", "completed", "Restored music from cache")
            else:
                console.print(Panel("[bold]Stage 5/8: Generating Music[/bold]"))
                await _report("music_gen", "running", "Generating background music...")
                music_file = str(Path(temp_dir) / "background_music.wav")
                music_path = await self.music_generator.generate_music(
                    prompt=parsed_script.music_prompt,
                    output_path=music_file,
                    duration=parsed_script.total_duration + 5,  # extra buffer
                )
                if music_path:
                    artifacts.music_file = music_path
                completed_so_far.append("music_gen")
                _save_checkpoint(completed_so_far)
                await _report("music_gen", "completed", "Background music generated")

        # =================================================================
        # Stage 6: Subtitles
        # =================================================================
        subtitles: Optional[List[List[SubtitleSegment]]] = None
        if "subtitles" in stages and audio_files:
            if "subtitles" in resumed_stages:
                console.print("[dim]Stage 6/8: Subtitles — using cache[/dim]")
                await _report("subtitles", "completed", "Restored subtitles from cache")
                # Subtitles aren't easily serialised; regenerate from SRT dir
                # if files exist, but don't block the pipeline.
                subtitles = None  # assembler handles missing subs gracefully
            else:
                console.print(Panel("[bold]Stage 6/8: Generating Subtitles[/bold]"))
                await _report("subtitles", "running", "Generating word-level subtitles...")
                srt_dir = str(Path(temp_dir) / "subtitles")
                subtitles = await self.subtitle_generator.generate_subtitles(
                    audio_files, srt_dir
                )
                completed_so_far.append("subtitles")
                _save_checkpoint(completed_so_far)
                await _report("subtitles", "completed", "Subtitles generated")

        # =================================================================
        # Stage 7: Assembly  (never cached — always re-run)
        # =================================================================
        if "assemble" in stages:
            console.print(Panel("[bold]Stage 7/8: Assembling Final Video[/bold]"))
            await _report("assemble", "running", "Assembling final video...")
            final_path = str(
                Path(project_dir) / f"{project_name}.{self.config.output.get('format', 'mp4')}"
            )
            final_video = self.video_assembler.assemble(
                parsed_script=parsed_script,
                video_clips=video_files,
                audio_files=audio_files,
                music_path=music_path,
                subtitles=subtitles,
                output_path=final_path,
            )
            artifacts.final_video_path = final_video
            completed_so_far.append("assemble")
            await _report("assemble", "completed", "Final video assembled!")

        # =================================================================
        # Done — remove checkpoint (full run succeeded)
        # =================================================================
        ckpt_mgr.delete()
        elapsed = time.time() - start_time
        self._print_summary(artifacts, elapsed)

        return artifacts.final_video_path or project_dir

    # =====================================================================
    # Display helpers
    # =====================================================================

    def _print_header(
        self, script: str, platform: Platform, project_dir: str
    ) -> None:
        console.print()
        console.print(
            Panel(
                f"[bold magenta]ContentCreator AI Video Engine[/bold magenta]\n\n"
                f"Platform: [cyan]{platform.value}[/cyan]\n"
                f"Output: [cyan]{project_dir}[/cyan]\n"
                f"Script: [dim]{script[:100]}{'...' if len(script) > 100 else ''}[/dim]",
                title="🎬 Starting Pipeline",
            )
        )

    def _print_scene_summary(self, parsed_script: ParsedScript) -> None:
        table = Table(title=f"Scenes: {parsed_script.title}")
        table.add_column("#", style="cyan", width=4)
        table.add_column("Title", style="green")
        table.add_column("Duration", style="yellow", width=8)
        table.add_column("Transition", style="dim", width=12)
        table.add_column("Narration", style="white", max_width=50)

        for scene in parsed_script.scenes:
            table.add_row(
                str(scene.scene_number),
                scene.title,
                f"{scene.duration_seconds:.0f}s",
                scene.transition.value,
                scene.narration[:50] + ("..." if len(scene.narration) > 50 else ""),
            )

        console.print(table)
        console.print(
            f"[dim]Total duration: {parsed_script.total_duration:.0f}s | "
            f"Music: {parsed_script.music_prompt}[/dim]\n"
        )

    def _print_summary(
        self, artifacts: PipelineArtifacts, elapsed: float
    ) -> None:
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)

        console.print()
        console.print(
            Panel(
                f"[bold green]Pipeline Complete![/bold green]\n\n"
                f"Time: [cyan]{mins}m {secs}s[/cyan]\n"
                f"Scenes: [cyan]{len(artifacts.scene_video_files)}[/cyan]\n"
                f"Audio files: [cyan]{len(artifacts.scene_audio_files)}[/cyan]\n"
                f"Video clips: [cyan]{len(artifacts.scene_video_files)}[/cyan]\n"
                f"Music: [cyan]{'Yes' if artifacts.music_file else 'No'}[/cyan]\n"
                f"Subtitles: [cyan]{'Yes' if artifacts.subtitle_file else 'No'}[/cyan]\n"
                f"Final video: [bold cyan]{artifacts.final_video_path or 'N/A'}[/bold cyan]",
                title="✅ Done",
            )
        )
