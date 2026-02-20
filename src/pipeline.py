"""
ContentCreator - Pipeline Orchestrator

Orchestrates the full video creation pipeline:
  Script → Parse → TTS → Images → Video → Music → Subtitles → Assemble

Manages VRAM by loading/unloading models sequentially.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import Config
from src.image_generator import ImageGenerator
from src.models.schemas import ParsedScript, PipelineArtifacts, Platform
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

    async def run(
        self,
        script: str,
        platform: Platform = Platform.REELS,
        output_name: Optional[str] = None,
        num_scenes: Optional[int] = None,
    ) -> str:
        """
        Run the full pipeline: script → final video.

        Args:
            script: Raw script text or video idea
            platform: Target platform (youtube, reels, shorts)
            output_name: Optional custom name for the output
            num_scenes: Optional override for number of scenes

        Returns:
            Path to the final video file
        """
        start_time = time.time()
        stages = self.config.pipeline.get("stages", [])

        # Create project directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = output_name or f"video_{timestamp}"
        output_dir = self.config.output.get("directory", "output")
        project_dir = str(Path(output_dir) / project_name)
        os.makedirs(project_dir, exist_ok=True)

        temp_dir = str(Path(project_dir) / "temp")
        os.makedirs(temp_dir, exist_ok=True)

        artifacts = PipelineArtifacts(project_dir=project_dir)

        self._print_header(script, platform, project_dir)

        # =================================================================
        # Stage 1: Parse Script
        # =================================================================
        if "script_parse" in stages:
            console.print(Panel("[bold]Stage 1/7: Parsing Script[/bold]"))
            parsed_script = await self.script_parser.parse(
                script, platform, num_scenes
            )
            artifacts.parsed_script = parsed_script

            # Save parsed script as JSON
            script_json_path = str(Path(project_dir) / "parsed_script.json")
            with open(script_json_path, "w", encoding="utf-8") as f:
                f.write(parsed_script.model_dump_json(indent=2))

            self._print_scene_summary(parsed_script)
        else:
            raise RuntimeError("script_parse stage is required.")

        # =================================================================
        # Stage 2: Text-to-Speech
        # =================================================================
        audio_files: List[str] = []
        if "tts" in stages:
            console.print(Panel("[bold]Stage 2/7: Generating Voiceover[/bold]"))
            audio_dir = str(Path(temp_dir) / "audio")
            audio_files = await self.tts_engine.generate_scene_audio(
                parsed_script, audio_dir
            )
            artifacts.scene_audio_files = audio_files

        # =================================================================
        # Stage 3: Image Generation
        # =================================================================
        image_files: List[str] = []
        if "image_gen" in stages:
            console.print(Panel("[bold]Stage 3/7: Generating Images[/bold]"))
            image_dir = str(Path(temp_dir) / "images")
            image_files = await self.image_generator.generate_scene_images(
                parsed_script, image_dir
            )
            artifacts.scene_image_files = image_files

        # =================================================================
        # Stage 4: Video Generation
        # =================================================================
        video_files: List[str] = []
        if "video_gen" in stages:
            console.print(Panel("[bold]Stage 4/7: Generating Video Clips[/bold]"))
            video_dir = str(Path(temp_dir) / "videos")
            video_files = await self.video_generator.generate_scene_videos(
                parsed_script, video_dir
            )
            artifacts.scene_video_files = video_files

        # =================================================================
        # Stage 5: Music Generation
        # =================================================================
        music_path: Optional[str] = None
        if "music_gen" in stages:
            console.print(Panel("[bold]Stage 5/7: Generating Music[/bold]"))
            music_file = str(Path(temp_dir) / "background_music.wav")
            music_path = await self.music_generator.generate_music(
                prompt=parsed_script.music_prompt,
                output_path=music_file,
                duration=parsed_script.total_duration + 5,  # extra buffer
            )
            if music_path:
                artifacts.music_file = music_path

        # =================================================================
        # Stage 6: Subtitles
        # =================================================================
        subtitles: Optional[List[List[SubtitleSegment]]] = None
        if "subtitles" in stages and audio_files:
            console.print(Panel("[bold]Stage 6/7: Generating Subtitles[/bold]"))
            srt_dir = str(Path(temp_dir) / "subtitles")
            subtitles = await self.subtitle_generator.generate_subtitles(
                audio_files, srt_dir
            )

        # =================================================================
        # Stage 7: Assembly
        # =================================================================
        if "assemble" in stages:
            console.print(Panel("[bold]Stage 7/7: Assembling Final Video[/bold]"))
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

        # =================================================================
        # Done
        # =================================================================
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
