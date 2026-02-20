"""
ContentCreator - FastAPI Backend Server

Provides REST API + SSE progress streaming for the React frontend.
Bridges the web UI to the pipeline engine.

Run:
    uvicorn src.api_server:app --host 0.0.0.0 --port 8001 --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.config import Config
from src.models.schemas import Platform
from src.pipeline import Pipeline

# =========================================================================
# Logging — visible in uvicorn console
# =========================================================================

logger = logging.getLogger("contentcreator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(handler)

# =========================================================================
# App & CORS
# =========================================================================

app = FastAPI(title="ContentCreator AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================================
# In-memory job store
# =========================================================================

jobs: Dict[str, Dict[str, Any]] = {}

STAGES = [
    {"id": "script_parse", "label": "Script Parsing", "icon": "FileText"},
    {"id": "character_design", "label": "Character Design", "icon": "Palette"},
    {"id": "tts", "label": "Voice Generation", "icon": "Mic"},
    {"id": "image_gen", "label": "Image Generation", "icon": "Image"},
    {"id": "video_gen", "label": "Video Generation", "icon": "Film"},
    {"id": "music_gen", "label": "Music Generation", "icon": "Music"},
    {"id": "subtitles", "label": "Subtitle Generation", "icon": "MessageSquare"},
    {"id": "assemble", "label": "Final Assembly", "icon": "Clapperboard"},
]


def _init_job(job_id: str, active_stages: List[str]) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "status": "queued",
        "current_stage": None,
        "stages": {
            s["id"]: ("pending" if s["id"] in active_stages else "skipped")
            for s in STAGES
        },
        "active_stages": active_stages,
        "message": "Preparing pipeline...",
        "progress": 0,
        "video_path": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
    }


async def _progress_callback(
    job_id: str, stage: str, status: str, message: str = ""
) -> None:
    """Called by the pipeline to report stage progress."""
    if job_id not in jobs:
        return
    job = jobs[job_id]
    job["current_stage"] = stage
    job["stages"][stage] = status
    if message:
        job["message"] = message

    # Calculate overall progress
    active = job["active_stages"]
    completed = sum(1 for s in active if job["stages"].get(s) == "completed")
    running = sum(1 for s in active if job["stages"].get(s) == "running")
    total = len(active)
    # A running stage counts as half done
    job["progress"] = round(((completed + running * 0.5) / max(total, 1)) * 100, 1)
    job["status"] = "running"


async def _run_pipeline_job(
    job_id: str, config: Config, params: Dict[str, Any]
) -> None:
    """Run the pipeline in a background thread so the event loop stays free."""

    def _sync_run():
        """Execute pipeline in its own event loop (runs in a worker thread)."""
        # Ensure prints from Rich/pipeline are flushed to console
        sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_pipeline_work(job_id, config, params))
        finally:
            loop.close()

    thread = threading.Thread(target=_sync_run, daemon=True)
    thread.start()


async def _pipeline_work(
    job_id: str, config: Config, params: Dict[str, Any]
) -> None:
    """The actual pipeline execution (runs inside the worker thread's loop)."""
    try:
        logger.info("[%s] Pipeline starting — model=%s, platform=%s",
                    job_id, config.llm.get('model'), params['platform'])
        jobs[job_id]["status"] = "running"
        pipeline = Pipeline(config)

        async def cb(stage: str, status: str, message: str = "") -> None:
            logger.info("[%s] %s → %s  %s", job_id, stage, status, message)
            await _progress_callback(job_id, stage, status, message)

        result = await pipeline.run(
            script=params["script"],
            platform=Platform(params["platform"]),
            num_scenes=params.get("num_scenes"),
            characters=params.get("characters"),
            character_style=params.get("character_style"),
            progress_callback=cb,
        )

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"] = "Video ready!"
        jobs[job_id]["video_path"] = result
        logger.info("[%s] Pipeline COMPLETED — %s", job_id, result)

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["message"] = f"Failed: {str(e)}"
        logger.error("[%s] Pipeline FAILED — %s", job_id, e, exc_info=True)


# =========================================================================
# API endpoints
# =========================================================================


@app.post("/api/generate")
async def generate_video(
    script: str = Form(...),
    platform: str = Form("reels"),
    num_scenes: Optional[int] = Form(None),
    character_style: Optional[str] = Form(None),
    characters: Optional[str] = Form(None),
    voice: Optional[str] = Form(None),
    music_enabled: bool = Form(True),
    subtitles_enabled: bool = Form(True),
    quality: str = Form("standard"),
):
    """Start video generation and return a job_id for progress tracking."""
    job_id = str(uuid.uuid4())[:8]

    # Parse characters JSON
    char_list: Optional[List[Dict[str, Any]]] = None
    if characters:
        try:
            char_list = json.loads(characters)
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid characters JSON")

    # Load config and apply overrides
    config = Config()

    if voice:
        config._data["tts"]["voice"] = voice

    if quality == "high":
        config._data["image"]["num_inference_steps"] = 50
        config._data["image"]["use_refiner"] = True
    elif quality == "ultra":
        config._data["image"]["num_inference_steps"] = 50
        config._data["image"]["use_refiner"] = True
        config._data["image"]["guidance_scale"] = 9.0

    stages = list(config._data["pipeline"]["stages"])
    if not music_enabled and "music_gen" in stages:
        stages.remove("music_gen")
    if not subtitles_enabled and "subtitles" in stages:
        stages.remove("subtitles")
    if not char_list and "character_design" in stages:
        stages.remove("character_design")
    config._data["pipeline"]["stages"] = stages

    jobs[job_id] = _init_job(job_id, stages)

    params: Dict[str, Any] = {
        "script": script,
        "platform": platform,
        "num_scenes": num_scenes,
        "characters": char_list,
        "character_style": character_style,
    }

    asyncio.create_task(_run_pipeline_job(job_id, config, params))

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/progress/{job_id}")
async def progress_stream(job_id: str):
    """SSE stream — push progress updates to the frontend."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    async def stream():
        while True:
            if job_id in jobs:
                data = json.dumps(jobs[job_id], default=str)
                yield f"data: {data}\n\n"
                if jobs[job_id]["status"] in ("completed", "failed"):
                    break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/api/job/{job_id}")
async def get_job(job_id: str):
    """Non-streaming status check."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/api/video/{job_id}")
async def get_video(job_id: str):
    """Stream / download the final video file."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    video_path = jobs[job_id].get("video_path")
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(404, "Video not ready yet")
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=os.path.basename(video_path),
    )


@app.get("/api/stages")
async def get_stages():
    return {"stages": STAGES}


@app.get("/api/health")
async def health():
    return {"status": "ok", "gpu": _check_gpu()}


def _check_gpu() -> dict:
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "vram_gb": round(
                    torch.cuda.get_device_properties(0).total_memory / 1e9, 1
                ),
            }
    except Exception:
        pass
    return {"available": False, "name": "N/A", "vram_gb": 0}


# =========================================================================
# Serve built frontend in production
# =========================================================================
_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount(
        "/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend"
    )
