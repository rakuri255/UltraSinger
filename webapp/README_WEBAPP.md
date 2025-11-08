# UltraSinger Web Application

A modern, polished web interface for [UltraSinger](https://github.com/rakuri255/UltraSinger) - an AI-powered tool that automatically creates UltraStar karaoke files from YouTube videos or audio files.

## Features

- **Dual Input Sources**: Process songs from YouTube URLs or uploaded audio files (MP3, WAV, OGG, M4A, FLAC)
- **Multi-Language Support**: Italian, English, and Polish lyrics transcription
- **Quality Presets**: Choose between Fast, Balanced, or Accurate processing
- **Real-Time Progress**: WebSocket-based live updates with detailed step tracking
- **Job Queue System**: Process multiple songs concurrently with automatic queuing
- **Beautiful UI**: Modern dark-themed interface with smooth animations
- **Docker Ready**: One-command deployment with Docker Compose

## Architecture

### Backend (FastAPI + Python)
- RESTful API for job management
- WebSocket support for real-time progress updates
- Background job queue with asyncio
- Integration with UltraSinger's AI pipeline
- Automatic cleanup of old jobs

### Frontend (Vue 3 + Vite + Tailwind CSS)
- Responsive, mobile-first design
- Drag-and-drop file uploads
- Real-time progress visualization
- Job history and management

### AI Pipeline
1. **Vocal Separation**: Demucs AI model separates vocals from instrumentals
2. **Transcription**: OpenAI Whisper transcribes lyrics with timestamps
3. **Pitch Detection**: Crepe detects melodic pitch for each syllable
4. **File Generation**: Creates UltraStar .txt format with synchronized lyrics and notes

## Prerequisites

### For Local Development
- **Python 3.10+**
- **Node.js 18+**
- **FFmpeg** (for audio processing)
- **8GB+ RAM** (16GB recommended)
- **NVIDIA GPU with 4GB+ VRAM** (optional, but significantly faster)

### For Docker Deployment
- **Docker** 20.10+
- **Docker Compose** 2.0+
- **NVIDIA Docker** (optional, for GPU acceleration)

## Installation

### Option 1: Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/rakuri255/UltraSinger.git
   cd UltraSinger
   ```

2. **Configure environment** (optional)
   ```bash
   cp .env.example .env
   # Edit .env to customize settings
   ```

3. **Start the application**
   ```bash
   cd webapp/docker
   docker-compose up -d
   ```

4. **Access the web interface**
   - Frontend: http://localhost
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Option 2: Local Development

#### Backend Setup

1. **Install UltraSinger dependencies**
   ```bash
   pip install -r requirements-linux.txt  # or requirements-windows.txt
   ```

2. **Install webapp backend dependencies**
   ```bash
   cd webapp/backend
   pip install -r requirements.txt
   ```

3. **Run the backend server**
   ```bash
   python main.py
   ```

   The API will be available at http://localhost:8000

#### Frontend Setup

1. **Install frontend dependencies**
   ```bash
   cd webapp/frontend
   npm install
   ```

2. **Run the development server**
   ```bash
   npm run dev
   ```

   The UI will be available at http://localhost:3000

## Configuration

Edit `.env` file to customize settings:

```bash
# Backend
BACKEND_PORT=8000
MAX_CONCURRENT_JOBS=2        # How many songs to process simultaneously
JOB_RETENTION_HOURS=24       # Auto-delete completed jobs after X hours

# UltraSinger
WHISPER_MODEL=medium         # tiny|small|medium|large-v2 (larger = more accurate)
CREPE_MODEL=full            # tiny|small|medium|large|full (larger = more accurate)
FORCE_CPU=false             # Set to true to disable GPU (slower)

# Storage
MAX_FILE_SIZE=104857600     # 100MB max upload size
```

## Usage Guide

### Creating a Karaoke File

1. **Choose Input Source**
   - **YouTube URL**: Paste a YouTube video link
   - **Upload File**: Drag and drop or browse for an audio file

2. **Select Language**
   - Choose between Italian, English, or Polish for lyrics transcription

3. **Choose Quality Preset**
   - **Fast**: ~5 minutes per song (tiny Whisper, tiny Crepe)
   - **Balanced**: ~10 minutes per song (small Whisper, medium Crepe)
   - **Accurate**: ~20 minutes per song (medium Whisper, full Crepe)

4. **Generate**
   - Click "Generate Karaoke File"
   - Monitor real-time progress in the Job Queue
   - Download the .txt file when complete

### Processing Steps

The application shows progress through these stages:

1. **Downloading** (YouTube only): Fetching audio from YouTube
2. **Separating Vocals**: AI separates voice from music
3. **Transcribing Lyrics**: Whisper transcribes with timestamps
4. **Detecting Pitch**: Crepe analyzes melodic notes
5. **Generating File**: Creates final UltraStar .txt

## API Documentation

### Endpoints

#### Upload File
```
POST /api/upload
Content-Type: multipart/form-data

Returns: { filename, size, upload_id }
```

#### Create Job
```
POST /api/jobs/create
Content-Type: application/json

Body: {
  "source": "youtube" | "upload",
  "youtube_url": "https://youtube.com/...",
  "language": "it" | "en" | "pl",
  "quality": "fast" | "balanced" | "accurate"
}

Returns: Job object with job_id
```

#### Get Job Status
```
GET /api/jobs/{job_id}

Returns: Job object with progress and status
```

#### List Jobs
```
GET /api/jobs?limit=50&offset=0

Returns: { jobs: [...], total: N }
```

#### Download Result
```
GET /api/jobs/{job_id}/download

Returns: UltraStar .txt file
```

#### WebSocket Progress
```
WS /api/ws/{job_id}

Receives: {
  "job_id": "...",
  "status": "processing",
  "step": "transcribing",
  "percentage": 50,
  "message": "Transcribing lyrics...",
  "elapsed_seconds": 120.5
}
```

For interactive API documentation, visit http://localhost:8000/docs

## Development Workflow

### Backend Development

```bash
cd webapp/backend

# Run with auto-reload
python main.py

# Or use uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development

```bash
cd webapp/frontend

# Development server with hot reload
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Deployment

### Docker Production Deployment

1. **Build images**
   ```bash
   cd webapp/docker
   docker-compose build
   ```

2. **Start services**
   ```bash
   docker-compose up -d
   ```

3. **View logs**
   ```bash
   docker-compose logs -f
   ```

4. **Stop services**
   ```bash
   docker-compose down
   ```

### GPU Support

For NVIDIA GPU acceleration:

1. Install [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)

2. Verify GPU is available:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

3. Start with GPU support:
   ```bash
   docker-compose up -d
   ```

## Troubleshooting

### Common Issues

#### "FFmpeg not found"
**Solution**: Install FFmpeg
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: Download from https://ffmpeg.org/download.html

#### "CUDA out of memory"
**Solutions**:
1. Reduce `MAX_CONCURRENT_JOBS` to 1 in `.env`
2. Use smaller models: `WHISPER_MODEL=small`, `CREPE_MODEL=medium`
3. Enable CPU mode: `FORCE_CPU=true`

#### "YouTube download failed"
**Solutions**:
1. Check if URL is valid
2. Update yt-dlp: `pip install -U yt-dlp`
3. Try uploading the audio file instead

#### "WebSocket connection failed"
**Solutions**:
1. Check backend is running on port 8000
2. Verify CORS settings in `.env`
3. Check browser console for errors

#### "Job stuck in 'processing' state"
**Solutions**:
1. Check backend logs for errors
2. Ensure sufficient disk space in `/tmp/ultrasinger`
3. Restart backend: `docker-compose restart backend`

### Performance Tips

1. **GPU Acceleration**: Use NVIDIA GPU for 5-10x faster processing
2. **Concurrent Jobs**: Increase `MAX_CONCURRENT_JOBS` if you have >8GB VRAM
3. **Model Selection**: Larger models = better quality but slower processing
4. **File Formats**: WAV files process faster than compressed formats

## Project Structure

```
webapp/
├── backend/
│   ├── main.py                    # FastAPI application entry point
│   ├── api/
│   │   ├── models.py              # Pydantic models for API
│   │   └── routes.py              # API endpoints
│   ├── services/
│   │   ├── ultrasinger_service.py # UltraSinger integration
│   │   ├── queue_service.py       # Job queue management
│   │   └── youtube_service.py     # YouTube download handling
│   ├── utils/
│   │   ├── config.py              # Configuration settings
│   │   └── validators.py          # Input validation
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.vue                # Main application component
│   │   ├── components/
│   │   │   ├── UploadForm.vue     # File/URL upload form
│   │   │   ├── JobQueue.vue       # Job list display
│   │   │   └── ProgressCard.vue   # Individual job progress
│   │   ├── composables/
│   │   │   └── useWebSocket.js    # WebSocket connection hook
│   │   └── services/
│   │       └── api.js             # API client
│   ├── package.json
│   └── vite.config.js
└── docker/
    ├── Dockerfile.backend
    ├── Dockerfile.frontend
    ├── docker-compose.yml
    └── nginx.conf
```

## Credits

This web application is built on top of [UltraSinger](https://github.com/rakuri255/UltraSinger) by [@rakuri255](https://github.com/rakuri255).

### AI Models Used
- **Whisper** by OpenAI - Speech recognition
- **Crepe** - Pitch detection
- **Demucs** by Facebook Research - Source separation

### Technologies
- **Backend**: FastAPI, Python, uvicorn, yt-dlp
- **Frontend**: Vue 3, Vite, Tailwind CSS, Axios
- **Deployment**: Docker, Docker Compose, Nginx

## License

This project follows the same license as UltraSinger. See the main repository for details.

## Support

For issues related to:
- **Web Interface**: Open an issue in this repository
- **UltraSinger Core**: Visit https://github.com/rakuri255/UltraSinger

## Roadmap

Future enhancements:
- [ ] Batch processing for multiple songs
- [ ] Manual lyrics correction interface
- [ ] Preview generated file before download
- [ ] User accounts and saved history
- [ ] Advanced settings customization
- [ ] Playlist import from YouTube
- [ ] Audio preview player
- [ ] Mobile app (PWA)

---

**Enjoy creating karaoke files with AI!** 🎤🎵
