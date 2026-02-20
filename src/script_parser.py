"""
ContentCreator - Script Parser

Uses a local LLM (via Ollama) to parse a raw script into structured scenes.
Each scene gets: narration text, image prompt, duration, transition, music mood.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

import httpx
from rich.console import Console
from rich.panel import Panel

from src.config import Config
from src.models.schemas import ParsedScript, Platform

console = Console()

# =============================================================================
# System prompt that instructs the LLM how to decompose a script
# =============================================================================

SYSTEM_PROMPT = """You are an expert video producer AI. Your job is to take a raw script or idea and break it into structured scenes for an AI video generation pipeline.

For EACH scene you must provide:
1. scene_number: Sequential number starting from 1
2. title: Short scene title (2-5 words)
3. narration: The voiceover text (what the narrator says)
4. image_prompt: A detailed, vivid description for an AI image generator. Include style, composition, lighting, colors. Be specific and cinematic.
5. characters_in_scene: A list of character names that appear in this scene (use the exact names provided). Leave empty [] if no characters.
6. character_actions: What the characters are doing in this scene (e.g., "talking to a friend", "running", "sitting at a desk"). null if no characters.
7. character_emotions: The emotions/expressions of characters (e.g., "happy, excited", "sad, thoughtful"). null if no characters.
8. duration_seconds: How long this scene should last (typically 3-8 seconds)
9. transition: One of: "cut", "fade", "crossfade", "slide_left", "slide_right", "zoom_in"
10. music_mood: The mood/energy for background music (e.g., "upbeat electronic", "calm piano", "dramatic orchestral")

Also provide:
- title: Overall video title
- description: One-line video description
- music_prompt: Overall background music style description for the full video

RULES:
- Keep narration natural and engaging
- Image prompts should be highly detailed and visual (describe what the VIEWER sees)
- When characters are provided, include them naturally in your image_prompt descriptions. Describe what they look like and what they are doing.
- Each scene should be 3-8 seconds of narration
- For short-form (reels/shorts): 5-10 scenes total, punchy and fast
- For YouTube: 10-30 scenes, more detailed
- Always respond with ONLY valid JSON, no markdown, no extra text

Respond with this exact JSON structure:
{
  "title": "Video Title",
  "description": "Short description",
  "music_prompt": "overall music style",
  "scenes": [
    {
      "scene_number": 1,
      "title": "Scene Title",
      "narration": "What the narrator says...",
      "image_prompt": "Detailed visual description for AI image generation...",
      "characters_in_scene": ["CharacterName"],
      "character_actions": "what characters are doing",
      "character_emotions": "happy, curious",
      "duration_seconds": 5.0,
      "transition": "fade",
      "music_mood": "upbeat"
    }
  ]
}"""


# =============================================================================
# Script Parser Class
# =============================================================================

class ScriptParser:
    """Parses raw scripts into structured scenes using a local LLM via Ollama."""

    def __init__(self, config: Config):
        self.config = config
        self.base_url = config.llm.get("base_url", "http://localhost:11434")
        self.model = config.llm.get("model", "glm-4.7-flash:latest")
        self.temperature = config.llm.get("temperature", 0.7)
        self.max_tokens = config.llm.get("max_tokens", 4096)

    async def parse(
        self,
        script: str,
        platform: Platform = Platform.REELS,
        num_scenes: Optional[int] = None,
        characters: Optional[list[dict[str, Any]]] = None,
    ) -> ParsedScript:
        """
        Parse a raw script/idea into structured scenes.

        Args:
            script: Raw script text or video idea
            platform: Target platform (affects scene count and pacing)
            num_scenes: Optional override for number of scenes

        Returns:
            ParsedScript with structured scenes
        """
        console.print(Panel(f"[bold cyan]Parsing script with {self.model}[/bold cyan]"))

        user_prompt = self._build_user_prompt(script, platform, num_scenes, characters)

        # Call Ollama API
        response_text = await self._call_ollama(user_prompt)

        # Parse JSON from response
        parsed_data = self._extract_json(response_text)

        # Add platform info
        parsed_data["platform"] = platform.value

        # Validate with Pydantic
        parsed_script = ParsedScript(**parsed_data)

        console.print(
            f"[green]✓ Parsed {parsed_script.scene_count} scenes "
            f"({parsed_script.total_duration:.0f}s total)[/green]"
        )

        return parsed_script

    def _build_user_prompt(
        self,
        script: str,
        platform: Platform,
        num_scenes: Optional[int],
        characters: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        """Build the user prompt with context."""
        platform_guide = {
            Platform.REELS: "Instagram Reels / TikTok (9:16 vertical, 15-60 seconds, 5-10 scenes, fast-paced)",
            Platform.SHORTS: "YouTube Shorts (9:16 vertical, 15-60 seconds, 5-10 scenes, fast-paced)",
            Platform.YOUTUBE: "YouTube video (16:9 landscape, 1-10 minutes, 10-30 scenes, detailed)",
        }

        prompt = f"""Target Platform: {platform_guide[platform]}
"""
        if num_scenes:
            prompt += f"Number of scenes: exactly {num_scenes}\n"

        # Inject character definitions so the LLM knows who to place in scenes
        if characters:
            prompt += "\nCharacters in this video:\n"
            for char in characters:
                name = char.get("name", "Unknown")
                desc = char.get("description", "")
                traits = char.get("traits", [])
                traits_str = f" (Traits: {', '.join(traits)})" if traits else ""
                prompt += f"- {name}: {desc}{traits_str}\n"
            prompt += (
                "\nWhen writing scenes, naturally include these characters "
                "by adding their names to characters_in_scene, describe their "
                "actions in character_actions, and their emotions in character_emotions.\n"
            )

        prompt += f"""
Script / Idea:
\"\"\"
{script}
\"\"\"

Break this into scenes. Respond with ONLY the JSON."""

        return prompt

    async def _call_ollama(self, user_prompt: str) -> str:
        """Call the Ollama generate API."""
        url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        console.print("[dim]Calling Ollama API...[/dim]")

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        content: str = data.get("message", {}).get("content", "")
        if not content:
            raise ValueError("Empty response from Ollama. Is the model loaded?")

        return content

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response, handling markdown code blocks."""
        # Try to find JSON in code blocks first
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        # Try to find raw JSON object
        text = text.strip()
        if not text.startswith("{"):
            # Find the first { and last }
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start : end + 1]

        try:
            data: Dict[str, Any] = json.loads(text)
            return data
        except json.JSONDecodeError as e:
            console.print(f"[red]Failed to parse LLM response as JSON: {e}[/red]")
            console.print(f"[dim]Raw response:\n{text[:500]}[/dim]")
            raise ValueError(
                "LLM did not return valid JSON. "
                "Try again or use a different model."
            ) from e
