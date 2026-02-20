"""
ContentCreator - Configuration Loader

Loads config.yaml and provides typed access to all settings.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Central configuration loaded from config.yaml."""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            resolved: Path = Path(__file__).parent.parent / "config.yaml"
        else:
            resolved = Path(config_path)

        if not resolved.exists():
            raise FileNotFoundError(f"Config file not found: {resolved}")

        with open(resolved, "r", encoding="utf-8") as f:
            self._data: Dict[str, Any] = yaml.safe_load(f)

    # --- Accessors ---

    @property
    def llm(self) -> Dict[str, Any]:
        return self._data.get("llm", {})

    @property
    def tts(self) -> Dict[str, Any]:
        return self._data.get("tts", {})

    @property
    def image(self) -> Dict[str, Any]:
        return self._data.get("image", {})

    @property
    def video(self) -> Dict[str, Any]:
        return self._data.get("video", {})

    @property
    def music(self) -> Dict[str, Any]:
        return self._data.get("music", {})

    @property
    def subtitles(self) -> Dict[str, Any]:
        return self._data.get("subtitles", {})

    @property
    def output(self) -> Dict[str, Any]:
        return self._data.get("output", {})

    @property
    def pipeline(self) -> Dict[str, Any]:
        return self._data.get("pipeline", {})

    @property
    def device(self) -> str:
        return self._data.get("pipeline", {}).get("device", "cuda")

    @property
    def half_precision(self) -> bool:
        return self._data.get("pipeline", {}).get("half_precision", True)

    @property
    def auto_unload(self) -> bool:
        return self._data.get("pipeline", {}).get("auto_unload_models", True)

    def get_output_preset(self, preset_name: str) -> Dict[str, Any]:
        """Get output dimensions/settings for a platform preset."""
        presets = self.output.get("presets", {})
        if preset_name not in presets:
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Available: {list(presets.keys())}"
            )
        return presets[preset_name]

    def __repr__(self):
        return f"Config(llm={self.llm['model']}, device={self.device})"
