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

SYSTEM_PROMPT = """You are an expert video producer AI AND a master storyteller. Your job is to take a raw script or idea and break it into structured scenes for an AI video generation pipeline.

═══════════════════════════════════════════════════════════════
NARRATION STYLE — THIS IS THE MOST IMPORTANT SECTION
═══════════════════════════════════════════════════════════════

You MUST write the narration as ONE CONTINUOUS, FLOWING STORY told by a single captivating narrator.
The viewer should feel like they are being told a story by a skilled storyteller, NOT reading bullet points or a Wikipedia article.

RULES FOR NARRATION:
• Every scene's narration MUST naturally connect to the previous and next scene.
  Use transitional phrases: "But then...", "What no one expected was...",
  "And just when all hope seemed lost...", "Meanwhile, half a world away...",
  "Little did they know...", "But this was only the beginning..."
• VARY your sentence structure — mix short, punchy sentences with longer, flowing ones.
  Short for impact: "Everything changed." / "He was gone."
  Long for atmosphere: "Under a canopy of ten thousand stars, ancient eyes looked upward and dared to wonder what mysteries lay hidden in the heavens above."
• Build EMOTIONAL MOMENTUM throughout the video. Start with a hook, build tension or curiosity, hit climactic peaks, and resolve with a powerful conclusion.
• Write with SENSORY LANGUAGE — make the viewer SEE, FEEL, HEAR, and experience the story.
  Bad: "The rocket launched."
  Good: "With a deafening roar that shook the earth beneath their feet, the rocket tore through the clouds and vanished into the infinite black."
• Use RHETORICAL QUESTIONS to pull the viewer in: "But what if everything we thought we knew... was wrong?"
• The OPENING scene should HOOK the viewer in the first 3 seconds — start with drama, mystery, or a bold statement.
• The CLOSING scene should leave a LASTING EMOTIONAL IMPRESSION — end with an inspiring call to action, a thought-provoking question, or a powerful image.
• NEVER write flat, list-like narration. Each scene should feel like a chapter in an epic story.

BAD narration example (DO NOT write like this):
  Scene 1: "Ancient people looked at stars. They were curious about the sky."
  Scene 2: "Galileo made a telescope. He discovered many things."
  Scene 3: "Rockets were invented. They could go to space."

GOOD narration example (WRITE like this):
  Scene 1: "Long before cities lit up the night sky, our ancestors gazed upward — into an ocean of stars — and whispered the first questions that would define humanity forever."
  Scene 2: "Centuries passed. And then, in a small workshop in Italy, a man named Galileo pressed his eye to a strange new instrument... and the universe revealed its first secret."
  Scene 3: "But seeing the stars wasn't enough. Humanity wanted to TOUCH them. And so began the wildest dream in history — a dream written in fire and rocket fuel."

CRITICAL — SOURCE MATERIAL & VISUAL STYLE:
If the script references a KNOWN franchise, game, anime, movie, TV show, book, or any recognizable IP:
- You MUST set "visual_style" to describe the EXACT art style of that source material.
  Examples:
  • "Attack on Titan" → "anime art style, Attack on Titan anime aesthetic, cel-shaded, dramatic shading, Japanese animation style"
  • "Far Cry 3" → "Far Cry 3 video game art style, tropical first-person shooter aesthetic, realistic game rendering, lush jungle environment"
  • "Spider-Man" → "Marvel comic book art style, bold outlines, dynamic comic panel aesthetic"
  • "The Witcher" → "dark fantasy oil painting style, The Witcher game aesthetic, gritty medieval atmosphere"
  • "Naruto" → "Naruto anime art style, vibrant Japanese animation, cel-shaded, manga-inspired"
  • "GTA V" → "GTA V art style, satirical semi-realistic, bright colors, Rockstar Games aesthetic"
- Characters in image_prompt MUST be described as they appear in the source material:
  • For "Attack on Titan": describe Eren Yeager with his brown hair, green eyes, Survey Corps uniform with ODM gear, etc.
  • For "Far Cry 3": describe Jason Brody as the young American tourist with dark hair, or Vaas Montenegro with his mohawk, red tank top, facial scar, etc.
  • Use the character's CANONICAL appearance from the source material.
- ALL image_prompt fields must consistently use the same art style throughout.

If the script is generic / original content (not referencing any known IP):
- Set "visual_style" to a fitting cinematic style, e.g. "cinematic digital art, photorealistic, dramatic lighting".
- Describe characters with consistent visual details you invent.

For EACH scene you must provide:
1. scene_number: Sequential number starting from 1
2. title: Short scene title (2-5 words)
3. narration: The voiceover text — MUST follow the storytelling rules above. Write as a flowing, emotional narrative.
4. image_prompt: A detailed, vivid description for an AI image generator. MUST include:
   - The art style matching the source material
   - Specific character appearances (canonical for known IPs)
   - Composition, lighting, colors, camera angle
   - What the viewer SEES (not hears)
5. characters_in_scene: A list of character names that appear in this scene. Leave empty [] if no characters.
6. character_actions: What the characters are doing. null if no characters.
7. character_emotions: The emotions/expressions. null if no characters.
8. duration_seconds: How long this scene should last (typically 3-8 seconds)
9. transition: One of: "cut", "fade", "crossfade", "slide_left", "slide_right", "zoom_in"
10. music_mood: The mood/energy for background music
11. narration_tone: The emotional delivery tone for the voiceover. MUST be one of:
    "excited", "dramatic", "calm", "sad", "angry", "hopeful", "cheerful",
    "serious", "fearful", "inspiring", "mysterious", "epic", "gentle",
    "friendly", "tense", "triumphant", "neutral"
    Choose the tone that MATCHES the emotional beat of this specific scene.
    Vary tones across scenes to create an emotional journey — don't use the same tone for every scene.

Also provide:
- title: Overall video title
- description: One-line video description
- music_prompt: Overall background music style description for the full video
- visual_style: The art style for ALL images (see rules above). This MUST be consistent across all scenes.

RULES:
- Narration MUST read as flowing, emotional storytelling (see NARRATION STYLE above)
- Image prompts MUST be highly detailed and visual
- ALL scenes must use the SAME consistent visual_style
- Characters must look the same across ALL scenes (consistent appearance)
- For known franchises: characters MUST match their canonical appearance
- Each scene should be 3-8 seconds of narration
- For short-form (reels/shorts): 5-10 scenes total, punchy and fast
- For YouTube: 10-30 scenes, more detailed
- Always respond with ONLY valid JSON, no markdown, no extra text

Respond with this exact JSON structure:
{
  "title": "Video Title",
  "description": "Short description",
  "music_prompt": "overall music style",
  "visual_style": "art style description for image generation",
  "scenes": [
    {
      "scene_number": 1,
      "title": "Scene Title",
      "narration": "What the narrator says...",
      "image_prompt": "Detailed visual description including art style and character appearances...",
      "characters_in_scene": ["CharacterName"],
      "character_actions": "what characters are doing",
      "character_emotions": "happy, curious",
      "duration_seconds": 5.0,
      "transition": "fade",
      "music_mood": "upbeat",
      "narration_tone": "dramatic"
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

    # Approximate LLM output tokens per scene (narration + image_prompt + JSON keys)
    # Raised from 350 to 420 to accommodate richer storytelling narration
    _TOKENS_PER_SCENE = 420
    _TOKEN_OVERHEAD = 512  # title, description, music_prompt, visual_style, JSON wrapper

    def _estimate_max_tokens(
        self, platform: Platform, num_scenes: Optional[int]
    ) -> int:
        """Compute the minimum ``num_predict`` so the LLM won't truncate."""
        if num_scenes:
            needed = num_scenes * self._TOKENS_PER_SCENE + self._TOKEN_OVERHEAD
        else:
            # Platform-based defaults
            default_scenes = {
                Platform.REELS: 10,
                Platform.SHORTS: 10,
                Platform.YOUTUBE: 30,
            }
            needed = default_scenes.get(platform, 10) * self._TOKENS_PER_SCENE + self._TOKEN_OVERHEAD
        return max(self.max_tokens, needed)

    async def parse(
        self,
        script: str,
        platform: Platform = Platform.REELS,
        num_scenes: Optional[int] = None,
        characters: Optional[list[dict[str, Any]]] = None,
        max_retries: int = 3,
    ) -> ParsedScript:
        """
        Parse a raw script/idea into structured scenes.

        If the script already contains explicit Scene/Voiceover/Image-prompt
        blocks, it is parsed directly without calling the LLM.
        Otherwise the LLM is used to decompose free-form text.

        Args:
            script: Raw script text or video idea
            platform: Target platform (affects scene count and pacing)
            num_scenes: Optional override for number of scenes
            characters: Optional list of character dicts
            max_retries: Number of attempts if LLM returns invalid JSON

        Returns:
            ParsedScript with structured scenes
        """

        # ── Fast path: detect pre-structured scripts ────────────────────
        structured = self._try_structured_parse(script, platform)
        if structured is not None:
            console.print(Panel("[bold green]Detected pre-structured script — skipping LLM[/bold green]"))
            console.print(
                f"[green]✓ Parsed {structured.scene_count} scenes "
                f"({structured.total_duration:.0f}s total)[/green]"
            )
            return structured

        # ── Slow path: ask the LLM to decompose the script ─────────────
        console.print(Panel(f"[bold cyan]Parsing script with {self.model}[/bold cyan]"))

        user_prompt = self._build_user_prompt(script, platform, num_scenes, characters)
        required_tokens = self._estimate_max_tokens(platform, num_scenes)

        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1:
                    console.print(f"[yellow]Retry {attempt}/{max_retries}...[/yellow]")

                # Call Ollama API
                response_text = await self._call_ollama(user_prompt, required_tokens)

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

            except (ValueError, json.JSONDecodeError, Exception) as e:
                last_error = e
                console.print(f"[red]Attempt {attempt} failed: {e}[/red]")

        raise RuntimeError(
            f"Script parsing failed after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    # -----------------------------------------------------------------
    # Structured-script detector & parser
    # -----------------------------------------------------------------

    # Patterns that match common structured-script formats:
    #   "Scene 1:", "Scene 2 -", "SCENE 3.", etc.
    _SCENE_HEADER_RE = re.compile(
        r"^(?:scene|Scene|SCENE)\s*(\d+)\s*[:\-.]?\s*$", re.MULTILINE
    )
    # Label lines: "Voiceover:", "VO:", "Narration:", "Image-prompt:", "Image prompt:", "Visual:", etc.
    _LABEL_RE = re.compile(
        r"^\s*(?P<label>voiceover|vo|narration|narrator|text"
        r"|image[- ]?prompt|visual|image|img)"
        r"\s*[:\-]\s*(?P<value>.*)",
        re.IGNORECASE,
    )

    def _try_structured_parse(
        self, script: str, platform: Platform
    ) -> Optional[ParsedScript]:
        """
        Attempt to parse a script that already has explicit
        Scene / Voiceover / Image-prompt blocks.

        Returns a ParsedScript if at least 2 scenes were found, else None.
        """
        # Quick heuristic: need at least 2 "Scene N" headers
        headers = list(self._SCENE_HEADER_RE.finditer(script))
        if len(headers) < 2:
            return None

        # ── Extract the title (text before the first Scene header) ──────
        preamble = script[: headers[0].start()].strip()
        # First non-empty line of preamble is the title
        title_line = ""
        description = ""
        for line in preamble.splitlines():
            stripped = line.strip()
            if stripped and not title_line:
                title_line = stripped
            elif stripped and title_line:
                description = stripped
                break
        title = title_line or "Untitled Video"

        # ── Split into scene chunks ─────────────────────────────────────
        scene_chunks: list[str] = []
        for i, hdr in enumerate(headers):
            start = hdr.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(script)
            scene_chunks.append(script[start:end])

        # ── Parse each chunk into narration + image_prompt ──────────────
        TRANSITIONS = ["cut", "fade", "crossfade", "zoom_in"]
        scenes = []
        for idx, chunk in enumerate(scene_chunks, start=1):
            narration = ""
            image_prompt = ""
            current_label: Optional[str] = None
            current_buf: list[str] = []

            def _flush():
                nonlocal narration, image_prompt
                if current_label is None:
                    return
                text = " ".join(current_buf).strip()
                # Strip markdown bold/italic markers for cleaner TTS
                text = re.sub(r"[*_]{1,2}", "", text)
                if current_label in ("voiceover", "vo", "narration", "narrator", "text"):
                    narration = text
                elif current_label in ("image-prompt", "image prompt", "imageprompt",
                                       "visual", "image", "img"):
                    image_prompt = text

            for line in chunk.splitlines():
                m = self._LABEL_RE.match(line)
                if m:
                    _flush()
                    current_label = m.group("label").lower().replace(" ", "-")
                    # Normalize common variants
                    if current_label in ("vo", "narrator", "text"):
                        current_label = "voiceover"
                    if current_label in ("visual", "img"):
                        current_label = "image-prompt"
                    current_buf = [m.group("value").strip()] if m.group("value").strip() else []
                else:
                    stripped = line.strip()
                    if stripped:
                        current_buf.append(stripped)
            _flush()

            if not narration and not image_prompt:
                continue

            # Estimate duration from word count (~3 words/sec for voiceover)
            word_count = len(narration.split()) if narration else 6
            duration = max(3.0, min(round(word_count / 2.8, 1), 10.0))

            scenes.append({
                "scene_number": idx,
                "title": f"Scene {idx}",
                "narration": narration or f"Scene {idx}",
                "image_prompt": image_prompt or narration or f"Scene {idx}",
                "characters_in_scene": [],
                "character_actions": None,
                "character_emotions": None,
                "duration_seconds": duration,
                "transition": TRANSITIONS[idx % len(TRANSITIONS)],
                "music_mood": "cinematic",
            })

        if len(scenes) < 2:
            return None

        return ParsedScript(
            title=title,
            description=description or f"{len(scenes)}-scene video",
            platform=platform,
            music_prompt="cinematic background score matching the mood of each scene",
            scenes=scenes,
        )

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

    async def _call_ollama(
        self, user_prompt: str, max_tokens: Optional[int] = None
    ) -> str:
        """Call the Ollama chat API with JSON format enforcement."""
        url = f"{self.base_url}/api/chat"
        num_predict = max_tokens or self.max_tokens
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "format": "json",          # Force Ollama to return valid JSON
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": num_predict,
            },
        }

        console.print("[dim]Calling Ollama API (JSON mode)...[/dim]")

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        content: str = data.get("message", {}).get("content", "")
        if not content:
            raise ValueError("Empty response from Ollama. Is the model loaded?")

        return content

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response with robust sanitization."""
        # Try to find JSON in code blocks first
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        # Strip leading/trailing whitespace
        text = text.strip()

        # Find the outermost JSON object
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start : end + 1]

        # --- Sanitize common LLM JSON issues ---
        # Remove single-line comments  (// ...)
        text = re.sub(r'//[^\n]*', '', text)
        # Remove trailing commas before } or ]
        text = re.sub(r',\s*([}\]])', r'\1', text)
        # Replace single quotes with double quotes (crude but useful)
        # Only if the text doesn't already parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Attempt fixing common issues
        fixed = text
        # Replace NaN / Infinity (not valid JSON)
        fixed = re.sub(r'\bNaN\b', 'null', fixed)
        fixed = re.sub(r'\bInfinity\b', '999999', fixed)
        # Remove control characters except newline/tab
        fixed = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', fixed)

        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # Last resort: try to find a JSON array in the "scenes" key
        # Sometimes LLMs wrap the object in extra text
        scenes_match = re.search(r'"scenes"\s*:\s*\[', fixed)
        if scenes_match:
            # Find the balanced object that contains this
            depth = 0
            obj_start = None
            for i, ch in enumerate(fixed):
                if ch == '{' and obj_start is None:
                    obj_start = i
                    depth = 1
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0 and obj_start is not None:
                        candidate = fixed[obj_start : i + 1]
                        candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            obj_start = None

        console.print(f"[red]Failed to parse LLM response as JSON[/red]")
        console.print(f"[dim]Raw response (first 800 chars):\n{text[:800]}[/dim]")
        raise ValueError(
            "LLM did not return valid JSON. "
            "Try again or use a different model."
        )
