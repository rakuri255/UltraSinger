"""API routes for UltraSinger web application"""

import logging
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from api.models import (
    CreateJobRequest,
    JobResponse,
    JobListResponse,
    UploadResponse,
    JobStatus,
    JobProgress,
)
from services.queue_service import job_queue
from utils.config import settings
from utils.validators import is_valid_youtube_url, is_valid_audio_file, sanitize_filename

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload an audio file for processing"""
    # Validate file type
    if not is_valid_audio_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Supported formats: MP3, WAV, OGG, M4A, FLAC"
        )

    # Sanitize filename
    safe_filename = sanitize_filename(file.filename)

    # Check file size
    content = await file.read()
    if len(content) > settings.max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.max_file_size / 1024 / 1024}MB"
        )

    # Save file
    file_path = settings.upload_dir / safe_filename
    with open(file_path, "wb") as f:
        f.write(content)

    logger.info(f"Uploaded file: {safe_filename} ({len(content)} bytes)")

    return UploadResponse(
        filename=safe_filename,
        size=len(content),
        upload_id=safe_filename,
    )


@router.post("/jobs/create", response_model=JobResponse)
async def create_job(request: CreateJobRequest):
    """Create a new processing job"""
    # Validate YouTube URL if source is YouTube
    if request.source.value == "youtube":
        if not request.youtube_url or not is_valid_youtube_url(request.youtube_url):
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    # Validate upload file if source is upload
    if request.source.value == "upload":
        if not request.upload_filename:
            raise HTTPException(status_code=400, detail="upload_filename is required for upload source")
        upload_path = settings.upload_dir / request.upload_filename
        if not upload_path.exists():
            raise HTTPException(status_code=404, detail="Uploaded file not found")

    # Create job
    job_id = job_queue.create_job(
        source=request.source,
        language=request.language,
        quality=request.quality,
        youtube_url=request.youtube_url,
        upload_filename=request.upload_filename,
    )

    job = job_queue.get_job(job_id)
    return _job_to_response(job)


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Get job status and details"""
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return _job_to_response(job)


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(limit: int = 50, offset: int = 0):
    """List all jobs"""
    jobs = job_queue.list_jobs()
    total = len(jobs)

    # Apply pagination
    jobs = jobs[offset:offset + limit]

    return JobListResponse(
        jobs=[_job_to_response(job) for job in jobs],
        total=total,
    )


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Cancel and delete a job"""
    success = job_queue.delete_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")

    return {"message": "Job deleted successfully"}


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a processing job"""
    success = job_queue.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found or already finished")

    return {"message": "Job cancelled successfully"}


@router.get("/jobs/{job_id}/download")
async def download_result(job_id: str):
    """Download the generated UltraStar file"""
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")

    if not job.result_file or not job.result_file.exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(
        job.result_file,
        media_type="text/plain",
        filename=job.result_file.name,
    )


@router.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job progress updates"""
    await websocket.accept()

    # Check if job exists
    job = job_queue.get_job(job_id)
    if not job:
        await websocket.close(code=4004, reason="Job not found")
        return

    # Register websocket
    await job_queue.register_websocket(job_id, websocket)

    try:
        # Send initial status
        await websocket.send_json({
            "job_id": job_id,
            "status": job.status.value,
            "step": job.current_step.value if job.current_step else None,
            "percentage": job.progress_percentage,
            "message": job.progress_message,
            "elapsed_seconds": job.elapsed_seconds,
        })

        # Keep connection alive and listen for client messages
        while True:
            # Wait for any message (ping/pong, etc.)
            await websocket.receive_text()

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        # Unregister websocket
        await job_queue.unregister_websocket(job_id, websocket)


def _job_to_response(job) -> JobResponse:
    """Convert Job to JobResponse"""
    progress = None
    if job.current_step:
        progress = JobProgress(
            step=job.current_step,
            percentage=job.progress_percentage,
            message=job.progress_message,
            elapsed_seconds=job.elapsed_seconds,
        )

    return JobResponse(
        job_id=job.job_id,
        source=job.source,
        status=job.status,
        language=job.language,
        quality=job.quality,
        title=job.title,
        created_at=job.created_at,
        updated_at=job.updated_at,
        progress=progress,
        error_message=job.error_message,
        result_file_path=str(job.result_file) if job.result_file else None,
    )
