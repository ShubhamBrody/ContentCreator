"""
ContentCreator - CLI Entry Point

Usage:
    python main.py --script "Your video idea here" --platform reels
    python main.py --script-file script.txt --platform youtube
    python main.py --script "5 tips for productivity" --platform shorts --scenes 5
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from src.config import Config
from src.models.schemas import Platform
from src.pipeline import Pipeline

console = Console()


@click.command()
@click.option(
    "--script", "-s",
    type=str,
    default=None,
    help="Script text or video idea (inline).",
)
@click.option(
    "--script-file", "-f",
    type=click.Path(exists=True),
    default=None,
    help="Path to a text file containing the script.",
)
@click.option(
    "--platform", "-p",
    type=click.Choice(["youtube", "reels", "shorts"], case_sensitive=False),
    default="reels",
    help="Target platform (affects aspect ratio and pacing).",
)
@click.option(
    "--output", "-o",
    type=str,
    default=None,
    help="Custom name for the output folder/video.",
)
@click.option(
    "--scenes", "-n",
    type=int,
    default=None,
    help="Override number of scenes to generate.",
)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    default=None,
    help="Path to custom config.yaml.",
)
def main(
    script: str | None,
    script_file: str | None,
    platform: str,
    output: str | None,
    scenes: int | None,
    config: str | None,
) -> None:
    """
    ContentCreator - AI Video Generation Engine

    Generate reels, shorts, or YouTube videos from a script using
    local AI models. No paid APIs required.

    Examples:

        python main.py -s "5 tips for better sleep" -p reels

        python main.py -f my_script.txt -p youtube -n 15

        python main.py -s "The history of AI in 60 seconds" -p shorts
    """
    # Resolve script text
    if script is None and script_file is None:
        console.print("[red]Error: Provide --script or --script-file[/red]")
        sys.exit(1)

    if script_file:
        script_text = Path(script_file).read_text(encoding="utf-8")
    else:
        script_text = script  # type: ignore[assignment]

    if not script_text or not script_text.strip():
        console.print("[red]Error: Script is empty.[/red]")
        sys.exit(1)

    # Map platform string to enum
    platform_map = {
        "youtube": Platform.YOUTUBE,
        "reels": Platform.REELS,
        "shorts": Platform.SHORTS,
    }
    target_platform = platform_map[platform.lower()]

    # Load config
    try:
        cfg = Config(config)
    except FileNotFoundError as e:
        console.print(f"[red]Config error: {e}[/red]")
        sys.exit(1)

    # Run pipeline
    pipeline = Pipeline(cfg)

    try:
        result = asyncio.run(
            pipeline.run(
                script=script_text,
                platform=target_platform,
                output_name=output,
                num_scenes=scenes,
            )
        )
        console.print(f"\n[bold green]Output: {result}[/bold green]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red bold]Pipeline failed: {e}[/red bold]")
        console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
