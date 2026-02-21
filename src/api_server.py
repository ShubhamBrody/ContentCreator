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
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.config import Config
from src.checkpoint import CheckpointManager
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

# Rough ETA estimates (seconds) per stage — used as initial guess before
# we have real timing data.  Updated with actual times after each run.
STAGE_ETA_ESTIMATES: Dict[str, float] = {
    "script_parse": 15,
    "character_design": 60,
    "tts": 20,
    "image_gen": 90,
    "video_gen": 180,
    "music_gen": 45,
    "subtitles": 20,
    "assemble": 15,
}


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
        # --- ETA tracking ---
        "stage_timings": {
            s["id"]: {
                "started_at": None,
                "completed_at": None,
                "elapsed": 0,
                "eta": STAGE_ETA_ESTIMATES.get(s["id"], 30),
            }
            for s in STAGES
            if s["id"] in active_stages
        },
        "total_elapsed": 0,
        "total_eta": sum(
            STAGE_ETA_ESTIMATES.get(s, 30) for s in active_stages
        ),
        "pipeline_started_at": None,
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

    now = time.time()
    timings = job.get("stage_timings", {})

    # Update timing data
    if stage in timings:
        st = timings[stage]
        if status == "running" and st["started_at"] is None:
            st["started_at"] = now
        elif status == "completed":
            st["completed_at"] = now
            if st["started_at"] is not None:
                actual = now - st["started_at"]
                st["elapsed"] = round(actual, 1)
                st["eta"] = 0
                # Update global estimate for this stage type
                STAGE_ETA_ESTIMATES[stage] = round(actual, 1)

    # Refresh elapsed for all running stages
    for sid, st in timings.items():
        if st["started_at"] is not None and st["completed_at"] is None:
            st["elapsed"] = round(now - st["started_at"], 1)
            original_eta = STAGE_ETA_ESTIMATES.get(sid, 30)
            remaining = max(0, original_eta - st["elapsed"])
            st["eta"] = round(remaining, 1)

    # Calculate overall progress
    active = job["active_stages"]
    completed = sum(1 for s in active if job["stages"].get(s) == "completed")
    running = sum(1 for s in active if job["stages"].get(s) == "running")
    total = len(active)
    # A running stage counts as half done
    job["progress"] = round(((completed + running * 0.5) / max(total, 1)) * 100, 1)
    job["status"] = "running"

    # Total elapsed and total ETA
    if job["pipeline_started_at"] is not None:
        job["total_elapsed"] = round(now - job["pipeline_started_at"], 1)
    _refresh_timings(job)


def _refresh_timings(job: Dict[str, Any]) -> None:
    """Recalculate elapsed/ETA for all running stages and overall progress.

    Called on every SSE tick so the frontend always sees live numbers,
    not just snapshots from the last _progress_callback call.
    """
    now = time.time()
    timings = job.get("stage_timings", {})
    active = job.get("active_stages", [])

    # Update elapsed for every running stage
    for sid, st in timings.items():
        if st["started_at"] is not None and st["completed_at"] is None:
            st["elapsed"] = round(now - st["started_at"], 1)
            original_eta = STAGE_ETA_ESTIMATES.get(sid, 30)
            remaining = max(0, original_eta - st["elapsed"])
            st["eta"] = round(remaining, 1)

    # Granular overall progress: completed stages contribute 100%,
    # running stages contribute proportional to elapsed / estimated time.
    total = max(len(active), 1)
    progress_sum = 0.0
    for s in active:
        status = job["stages"].get(s)
        if status == "completed":
            progress_sum += 1.0
        elif status == "running" and s in timings:
            st = timings[s]
            est = STAGE_ETA_ESTIMATES.get(s, 30)
            # Cap intra-stage progress at 95% to avoid showing 100% before done
            intra = min(st["elapsed"] / max(est, 1), 0.95)
            progress_sum += intra

    job["progress"] = round((progress_sum / total) * 100, 1)

    # Total elapsed
    if job.get("pipeline_started_at") is not None:
        job["total_elapsed"] = round(now - job["pipeline_started_at"], 1)

    # Total remaining ETA
    remaining_est = sum(
        st["eta"] for st in timings.values() if st["completed_at"] is None
    )
    job["total_eta"] = round(remaining_est, 1)


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
        jobs[job_id]["pipeline_started_at"] = time.time()
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
            resume_dir=params.get("resume_dir"),
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
                _refresh_timings(jobs[job_id])
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


@app.get("/api/resumable")
async def list_resumable():
    """Return a list of projects that can be resumed from a checkpoint."""
    config = Config()
    output_dir = config.output.get("directory", "output")
    items = CheckpointManager.find_resumable(output_dir)
    return {"resumable": items}


@app.post("/api/resume")
async def resume_project(
    project_dir: str = Form(...),
):
    """Resume a previously interrupted pipeline run from its checkpoint."""
    mgr = CheckpointManager(project_dir)
    ckpt = mgr.load()
    if ckpt is None or not mgr.validate(ckpt):
        raise HTTPException(400, "No valid checkpoint found for this project")

    params = ckpt.get("params", {})
    active_stages = ckpt.get("active_stages", [])
    completed = set(ckpt.get("completed_stages", []))

    # Build config with same overrides as original run
    config = Config()

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = _init_job(job_id, active_stages)

    # Mark already-completed stages instantly
    for s in completed:
        if s in jobs[job_id]["stages"]:
            jobs[job_id]["stages"][s] = "completed"

    params["resume_dir"] = project_dir

    asyncio.create_task(_run_pipeline_job(job_id, config, params))

    return {
        "job_id": job_id,
        "status": "queued",
        "resumed_from": project_dir,
        "skipped_stages": list(completed),
    }


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
