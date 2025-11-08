"""Job queue service for managing processing tasks"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Callable
from dataclasses import dataclass, field

from api.models import JobStatus, ProcessingStep, QualityPreset, JobSource
from services.youtube_service import download_youtube_audio
from services.ultrasinger_service import process_with_ultrasinger
from utils.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Job:
    """Represents a processing job"""
    job_id: str
    source: JobSource
    status: JobStatus
    language: str
    quality: QualityPreset
    youtube_url: Optional[str] = None
    upload_filename: Optional[str] = None
    title: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    input_file: Optional[Path] = None
    output_dir: Optional[Path] = None
    result_file: Optional[Path] = None
    error_message: Optional[str] = None
    current_step: Optional[ProcessingStep] = None
    progress_percentage: int = 0
    progress_message: str = ""
    elapsed_seconds: float = 0.0


class JobQueue:
    """Manages job queue and processing"""

    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.websocket_connections: Dict[str, set] = {}
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_jobs)
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the queue service"""
        logger.info("Starting job queue service")
        self._cleanup_task = asyncio.create_task(self._cleanup_old_jobs())

    async def stop(self):
        """Stop the queue service"""
        logger.info("Stopping job queue service")
        if self._cleanup_task:
            self._cleanup_task.cancel()

        # Cancel all processing tasks
        for task in self.processing_tasks.values():
            task.cancel()

    def create_job(
        self,
        source: JobSource,
        language: str,
        quality: QualityPreset,
        youtube_url: Optional[str] = None,
        upload_filename: Optional[str] = None,
    ) -> str:
        """Create a new job and add to queue"""
        job_id = str(uuid.uuid4())

        job = Job(
            job_id=job_id,
            source=source,
            status=JobStatus.QUEUED,
            language=language,
            quality=quality,
            youtube_url=youtube_url,
            upload_filename=upload_filename,
        )

        self.jobs[job_id] = job
        self.websocket_connections[job_id] = set()

        # Start processing
        task = asyncio.create_task(self._process_job(job_id))
        self.processing_tasks[job_id] = task

        logger.info(f"Created job {job_id}")
        return job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        return self.jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        """List all jobs"""
        return sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        job = self.jobs.get(job_id)
        if not job:
            return False

        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return False

        job.status = JobStatus.CANCELLED
        job.updated_at = datetime.now()

        # Cancel the processing task
        task = self.processing_tasks.get(job_id)
        if task:
            task.cancel()

        logger.info(f"Cancelled job {job_id}")
        return True

    def delete_job(self, job_id: str) -> bool:
        """Delete a job"""
        job = self.jobs.pop(job_id, None)
        if not job:
            return False

        # Cancel if still processing
        self.cancel_job(job_id)

        # Clean up files
        if job.output_dir and job.output_dir.exists():
            import shutil
            shutil.rmtree(job.output_dir, ignore_errors=True)

        logger.info(f"Deleted job {job_id}")
        return True

    async def register_websocket(self, job_id: str, websocket):
        """Register a websocket for job updates"""
        if job_id not in self.websocket_connections:
            self.websocket_connections[job_id] = set()
        self.websocket_connections[job_id].add(websocket)

    async def unregister_websocket(self, job_id: str, websocket):
        """Unregister a websocket"""
        if job_id in self.websocket_connections:
            self.websocket_connections[job_id].discard(websocket)

    async def _broadcast_progress(self, job_id: str, job: Job):
        """Broadcast progress to all connected websockets"""
        if job_id not in self.websocket_connections:
            return

        message = {
            "job_id": job_id,
            "status": job.status.value,
            "step": job.current_step.value if job.current_step else None,
            "percentage": job.progress_percentage,
            "message": job.progress_message,
            "elapsed_seconds": job.elapsed_seconds,
        }

        # Send to all connected websockets
        for ws in list(self.websocket_connections[job_id]):
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to websocket: {e}")
                self.websocket_connections[job_id].discard(ws)

    async def _process_job(self, job_id: str):
        """Process a job"""
        async with self._semaphore:
            job = self.jobs[job_id]
            start_time = datetime.now()

            try:
                job.status = JobStatus.PROCESSING
                job.updated_at = datetime.now()

                # Create output directory
                job.output_dir = settings.output_dir / job_id
                job.output_dir.mkdir(parents=True, exist_ok=True)

                # Progress callback
                def update_progress(step: str, message: str, percentage: float):
                    job.current_step = ProcessingStep(step)
                    job.progress_message = message
                    job.progress_percentage = int(percentage)
                    job.elapsed_seconds = (datetime.now() - start_time).total_seconds()
                    job.updated_at = datetime.now()

                    # Broadcast to websockets
                    asyncio.create_task(self._broadcast_progress(job_id, job))

                # Get quality settings
                whisper_model, crepe_model = self._get_quality_settings(job.quality)

                # Step 1: Get input file
                if job.source == JobSource.YOUTUBE:
                    update_progress("downloading", "Downloading from YouTube...", 0)

                    def yt_progress(msg: str, pct: float):
                        update_progress("downloading", msg, pct * 0.2)  # 0-20%

                    job.input_file, job.title = await download_youtube_audio(
                        job.youtube_url,
                        job.output_dir,
                        yt_progress
                    )
                else:
                    # Upload file is already in upload_dir
                    job.input_file = settings.upload_dir / job.upload_filename
                    job.title = job.upload_filename

                # Step 2: Process with UltraSinger
                update_progress("separating", "Starting UltraSinger processing...", 20)

                def us_progress(step: str, message: str, percentage: float):
                    # Map 20-100% to remaining steps
                    adjusted_pct = 20 + (percentage * 0.8)
                    update_progress(step, message, adjusted_pct)

                job.result_file = await process_with_ultrasinger(
                    input_file=job.input_file,
                    output_dir=job.output_dir,
                    language=job.language,
                    whisper_model=whisper_model,
                    crepe_model=crepe_model,
                    force_cpu=settings.force_cpu,
                    ultrasinger_src_path=settings.ultrasinger_src_path,
                    progress_callback=us_progress,
                )

                # Success!
                job.status = JobStatus.COMPLETED
                job.progress_percentage = 100
                job.progress_message = "Processing completed successfully!"
                job.elapsed_seconds = (datetime.now() - start_time).total_seconds()
                job.updated_at = datetime.now()

                logger.info(f"Job {job_id} completed successfully")

            except asyncio.CancelledError:
                job.status = JobStatus.CANCELLED
                job.error_message = "Job was cancelled"
                job.updated_at = datetime.now()
                logger.info(f"Job {job_id} was cancelled")

            except Exception as e:
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                job.updated_at = datetime.now()
                logger.error(f"Job {job_id} failed: {e}", exc_info=True)

            finally:
                # Final broadcast
                await self._broadcast_progress(job_id, job)

                # Clean up task
                self.processing_tasks.pop(job_id, None)

    def _get_quality_settings(self, quality: QualityPreset) -> tuple[str, str]:
        """Get whisper and crepe model for quality preset"""
        quality_map = {
            QualityPreset.FAST: ("tiny", "tiny"),
            QualityPreset.BALANCED: ("small", "medium"),
            QualityPreset.ACCURATE: ("medium", "full"),
        }
        return quality_map[quality]

    async def _cleanup_old_jobs(self):
        """Periodically clean up old jobs"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour

                cutoff = datetime.now() - timedelta(hours=settings.job_retention_hours)
                to_delete = [
                    job_id
                    for job_id, job in self.jobs.items()
                    if job.updated_at < cutoff and job.status in (
                        JobStatus.COMPLETED,
                        JobStatus.FAILED,
                        JobStatus.CANCELLED
                    )
                ]

                for job_id in to_delete:
                    self.delete_job(job_id)
                    logger.info(f"Cleaned up old job {job_id}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")


# Global queue instance
job_queue = JobQueue()
