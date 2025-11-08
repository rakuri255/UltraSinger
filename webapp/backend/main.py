"""FastAPI application for UltraSinger web interface"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from services.queue_service import job_queue
from utils.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    logger.info("Starting UltraSinger Web Application")
    await job_queue.start()
    logger.info(f"Server running on {settings.backend_host}:{settings.backend_port}")

    yield

    # Shutdown
    logger.info("Shutting down UltraSinger Web Application")
    await job_queue.stop()


# Create FastAPI app
app = FastAPI(
    title="UltraSinger Web API",
    description="Web interface for UltraSinger - AI-powered UltraStar song file generator",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "UltraSinger Web API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "jobs_active": len([j for j in job_queue.jobs.values() if j.status.value == "processing"]),
        "jobs_queued": len([j for j in job_queue.jobs.values() if j.status.value == "queued"]),
        "jobs_total": len(job_queue.jobs),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
        log_level="info",
    )
