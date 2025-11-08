"""Pydantic models for API requests and responses"""

from enum import Enum
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, validator


class JobSource(str, Enum):
    """Source type for processing job"""
    YOUTUBE = "youtube"
    UPLOAD = "upload"


class JobStatus(str, Enum):
    """Job processing status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingStep(str, Enum):
    """Current processing step"""
    DOWNLOADING = "downloading"
    SEPARATING = "separating"
    TRANSCRIBING = "transcribing"
    PITCHING = "pitching"
    GENERATING = "generating"


class QualityPreset(str, Enum):
    """Quality presets for processing"""
    FAST = "fast"  # tiny whisper, tiny crepe
    BALANCED = "balanced"  # small whisper, medium crepe
    ACCURATE = "accurate"  # medium whisper, full crepe


class CreateJobRequest(BaseModel):
    """Request to create a new processing job"""
    source: JobSource
    youtube_url: Optional[str] = None
    upload_filename: Optional[str] = None
    language: str = Field(..., description="Language code: it, en, or pl")
    quality: QualityPreset = QualityPreset.BALANCED
    manual_lyrics: Optional[str] = None

    @validator("youtube_url")
    def validate_youtube_url(cls, v, values):
        """Validate YouTube URL is provided if source is YouTube"""
        if values.get("source") == JobSource.YOUTUBE and not v:
            raise ValueError("youtube_url is required when source is youtube")
        return v

    @validator("upload_filename")
    def validate_upload_filename(cls, v, values):
        """Validate upload_filename is provided if source is upload"""
        if values.get("source") == JobSource.UPLOAD and not v:
            raise ValueError("upload_filename is required when source is upload")
        return v

    @validator("language")
    def validate_language(cls, v):
        """Validate language code"""
        if v not in ["it", "en", "pl"]:
            raise ValueError("language must be one of: it, en, pl")
        return v


class JobProgress(BaseModel):
    """Job progress information"""
    step: ProcessingStep
    percentage: int = Field(..., ge=0, le=100)
    message: str
    elapsed_seconds: float


class JobResponse(BaseModel):
    """Response with job information"""
    job_id: str
    source: JobSource
    status: JobStatus
    language: str
    quality: QualityPreset
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    progress: Optional[JobProgress] = None
    error_message: Optional[str] = None
    result_file_path: Optional[str] = None


class JobListResponse(BaseModel):
    """Response with list of jobs"""
    jobs: list[JobResponse]
    total: int


class UploadResponse(BaseModel):
    """Response after file upload"""
    filename: str
    size: int
    upload_id: str


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
