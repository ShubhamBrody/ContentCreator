"""
ContentCreator — Per-stage smoke tests.

Each test exercises one pipeline stage in isolation so failures are
immediately attributed to the right module.

Run all:     python -m pytest tests/ -v
Run one:     python -m pytest tests/test_stages.py::test_tts -v
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from src.config import Config
from src.models.schemas import ParsedScript, Platform, Scene, SceneTransition

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_config() -> Config:
    """Load the project config.yaml."""
    return Config()


def _minimal_script(num_scenes: int = 1) -> ParsedScript:
    """Return a tiny ParsedScript for quick tests."""
    scenes = [
        Scene(
            scene_number=i + 1,
            title=f"Test Scene {i + 1}",
            narration="The quick brown fox jumps over the lazy dog.",
            image_prompt="A cinematic shot of a fox jumping over a sleeping dog in a sunny field, golden hour lighting, 4k",
            duration_seconds=3.0,
            transition=SceneTransition.FADE,
        )
        for i in range(num_scenes)
    ]
    return ParsedScript(
        title="Test Video",
        description="Automated test",
        platform=Platform.REELS,
        scenes=scenes,
        music_prompt="calm acoustic background music",
    )


@pytest.fixture
def config():
    return _get_config()


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="cc_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# =========================================================================
# 1. Script Parsing  (requires Ollama running with mistral:latest)
# =========================================================================


class TestScriptParsing:
    """Test that the LLM parses a raw script into structured scenes."""

    def test_script_parse(self, config, tmp_dir):
        from src.script_parser import ScriptParser

        parser = ScriptParser(config)
        script_text = (
            "A 15 second reel about why cats are better than dogs. "
            "Funny and sarcastic tone."
        )
        parsed = asyncio.run(
            parser.parse(script_text, Platform.REELS, num_scenes=2)
        )

        assert isinstance(parsed, ParsedScript)
        assert parsed.scene_count >= 1, "Should have at least 1 scene"
        assert parsed.title, "Should have a title"
        for scene in parsed.scenes:
            assert scene.narration, f"Scene {scene.scene_number} missing narration"
            assert scene.image_prompt, f"Scene {scene.scene_number} missing image_prompt"
            assert scene.duration_seconds > 0

    def test_structured_parse(self, config):
        """Pre-formatted scripts should be parsed without calling the LLM."""
        from src.script_parser import ScriptParser

        parser = ScriptParser(config)
        structured = (
            "Scene 1: Opening\n"
            "Voiceover: Welcome to the jungle.\n"
            "Image-prompt: Dense tropical jungle at sunrise\n\n"
            "Scene 2: Closing\n"
            "Voiceover: Goodbye world.\n"
            "Image-prompt: Sunset over the ocean\n"
        )
        parsed = asyncio.run(
            parser.parse(structured, Platform.REELS, num_scenes=2)
        )
        assert parsed.scene_count == 2
        assert "jungle" in parsed.scenes[0].narration.lower()


# =========================================================================
# 2. Text-to-Speech  (requires internet for edge-tts)
# =========================================================================


class TestTTS:
    """Test voice generation with edge-tts."""

    def test_tts_single_scene(self, config, tmp_dir):
        from src.tts_engine import TTSEngine

        engine = TTSEngine(config)
        script = _minimal_script(1)

        audio_files = asyncio.run(
            engine.generate_scene_audio(script, tmp_dir)
        )

        assert len(audio_files) == 1
        assert os.path.isfile(audio_files[0])
        assert os.path.getsize(audio_files[0]) > 1000, "Audio file too small"

    def test_tts_multiple_scenes(self, config, tmp_dir):
        from src.tts_engine import TTSEngine

        engine = TTSEngine(config)
        script = _minimal_script(3)

        audio_files = asyncio.run(
            engine.generate_scene_audio(script, tmp_dir)
        )

        assert len(audio_files) == 3
        for f in audio_files:
            assert os.path.isfile(f)


# =========================================================================
# 3. Image Generation  (requires GPU + SDXL download)
# =========================================================================


class TestImageGeneration:
    """Test SDXL image generation."""

    def test_generate_one_image(self, config, tmp_dir):
        from src.image_generator import ImageGenerator

        gen = ImageGenerator(config)
        script = _minimal_script(1)

        image_files = asyncio.run(
            gen.generate_scene_images(script, tmp_dir)
        )

        assert len(image_files) == 1
        assert os.path.isfile(image_files[0])
        assert os.path.getsize(image_files[0]) > 5000, "Image file too small"

        # Verify it's a valid image
        from PIL import Image
        img = Image.open(image_files[0])
        assert img.size[0] > 0 and img.size[1] > 0

        gen.unload()


# =========================================================================
# 4. Video Generation  (requires GPU or CPU fallback)
# =========================================================================


class TestVideoGeneration:
    """Test video clip generation from images."""

    def _make_dummy_image(self, path: str, w: int = 512, h: int = 768):
        """Create a solid-color test image."""
        from PIL import Image
        import numpy as np
        arr = np.random.randint(0, 255, (h, w, 3), dtype="uint8")
        Image.fromarray(arr).save(path)

    def test_ken_burns_fallback(self, config, tmp_dir):
        """Test the CPU-only Ken Burns engine (always available)."""
        from src.video_generator import VideoGenerator

        # Force image_motion engine
        config._data["video"]["engine"] = "image_motion"
        gen = VideoGenerator(config)
        script = _minimal_script(1)

        img_path = os.path.join(tmp_dir, "test_scene.png")
        self._make_dummy_image(img_path)
        script.scenes[0].image_path = img_path

        video_files = asyncio.run(
            gen.generate_scene_videos(script, tmp_dir)
        )

        assert len(video_files) == 1
        assert os.path.isfile(video_files[0])
        assert os.path.getsize(video_files[0]) > 1000, "Video file too small"


# =========================================================================
# 5. Music Generation  (requires GPU + MusicGen download)
# =========================================================================


class TestMusicGeneration:
    """Test MusicGen background music creation."""

    def test_generate_short_clip(self, config, tmp_dir):
        from src.music_generator import MusicGenerator

        gen = MusicGenerator(config)
        out_path = os.path.join(tmp_dir, "test_music.wav")

        result = asyncio.run(
            gen.generate_music(
                prompt="calm acoustic guitar",
                output_path=out_path,
                duration=5,  # short clip for speed
            )
        )

        assert result is not None
        assert os.path.isfile(result)
        assert os.path.getsize(result) > 5000, "Music file too small"

        # Verify it's valid WAV
        import scipy.io.wavfile
        rate, data = scipy.io.wavfile.read(result)
        assert rate > 0
        assert len(data) > 0
        assert data.dtype.name in ("float32", "float64", "int16", "int32"), \
            f"Unexpected audio dtype: {data.dtype}"


# =========================================================================
# 6. Subtitle Generation  (requires Whisper + an audio file)
# =========================================================================


class TestSubtitleGeneration:
    """Test Whisper-based subtitle generation."""

    def test_subtitles_from_audio(self, config, tmp_dir):
        import shutil
        if not shutil.which("ffmpeg"):
            pytest.skip("ffmpeg not on PATH – required by Whisper")

        from src.tts_engine import TTSEngine
        from src.subtitle_generator import SubtitleGenerator

        # First generate a real audio file to feed to Whisper
        tts = TTSEngine(config)
        audio_dir = os.path.join(tmp_dir, "audio")
        script = _minimal_script(1)
        audio_files = asyncio.run(tts.generate_scene_audio(script, audio_dir))

        assert len(audio_files) == 1, "Need audio to test subtitles"

        sub_gen = SubtitleGenerator(config)
        srt_dir = os.path.join(tmp_dir, "subs")
        subtitles = asyncio.run(
            sub_gen.generate_subtitles(audio_files, srt_dir)
        )

        assert subtitles is not None
        assert len(subtitles) == 1, "Should have subtitles for 1 audio file"
        assert len(subtitles[0]) > 0, "Should have at least 1 subtitle segment"

        # Check segment structure
        seg = subtitles[0][0]
        assert hasattr(seg, "text")
        assert hasattr(seg, "start")
        assert hasattr(seg, "end")
        assert seg.end > seg.start


# =========================================================================
# 7. Video Assembly  (CPU-only, requires video + audio files)
# =========================================================================


class TestVideoAssembly:
    """Test final video assembly with MoviePy."""

    def _make_dummy_video(self, path: str, duration: float = 2.0):
        """Create a minimal test video."""
        from moviepy import ImageSequenceClip
        import numpy as np

        frames = [
            np.random.randint(0, 255, (320, 180, 3), dtype="uint8")
            for _ in range(int(duration * 10))
        ]
        clip = ImageSequenceClip(frames, fps=10)
        clip.write_videofile(path, codec="libx264", audio=False, logger=None)
        clip.close()

    def test_assemble_without_music(self, config, tmp_dir):
        from src.video_assembler import VideoAssembler
        from src.tts_engine import TTSEngine

        script = _minimal_script(1)

        # Create dummy video
        vid_path = os.path.join(tmp_dir, "scene_001_video.mp4")
        self._make_dummy_video(vid_path)

        # Create real TTS audio
        tts = TTSEngine(config)
        audio_dir = os.path.join(tmp_dir, "audio")
        audio_files = asyncio.run(tts.generate_scene_audio(script, audio_dir))

        # Set paths on scenes
        script.scenes[0].video_clip_path = vid_path
        script.scenes[0].image_path = vid_path  # not used but present

        assembler = VideoAssembler(config)
        out_path = os.path.join(tmp_dir, "final.mp4")

        result = assembler.assemble(
            parsed_script=script,
            video_clips=[vid_path],
            audio_files=audio_files,
            music_path=None,
            subtitles=None,
            output_path=out_path,
        )

        assert os.path.isfile(result)
        assert os.path.getsize(result) > 1000, "Final video file too small"


# =========================================================================
# 8. API Server health check
# =========================================================================


class TestAPIServer:
    """Test that the FastAPI server starts and responds."""

    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from src.api_server import app

        client = TestClient(app)
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "gpu" in data

    def test_stages_endpoint(self):
        from fastapi.testclient import TestClient
        from src.api_server import app

        client = TestClient(app)
        resp = client.get("/api/stages")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["stages"]) == 8
