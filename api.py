"""
FastAPI Backend for Subtitle Generator.

Provides REST API endpoints for video subtitle generation and translation.

Run with:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

import os
import sys
import uuid
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from enum import Enum

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.audio_processor import AudioProcessor
from src.transcriber import Transcriber
from src.translator import Translator
from src.subtitle_generator import SubtitleGenerator

# ============================================
# App Configuration
# ============================================

app = FastAPI(
    title="Subtitle Generator & Translator API",
    description="""
    🎬 **Multi-Language Subtitle Generator API**
    
    Generate subtitles from video files and translate them to **10+ Indic languages** using 
    custom-trained Neural Machine Translation models.
    
    ## Features
    - 🎥 Upload video files (MP4, AVI, MKV, MOV, WebM)
    - 🎤 Automatic speech-to-text transcription (Whisper)
    - 🌐 Neural machine translation to **Indic languages**:
      - Assamese (as), Bengali (bn), Gujarati (gu), Hindi (hi)
      - Kannada (kn), Malayalam (ml), Marathi (mr), Odia (or)
      - Punjabi (pa), Tamil (ta), Telugu (te)
    - 📄 Download SRT/VTT subtitle files
    - ⚡ Background processing for long videos
    - 🔄 Lazy model loading (memory efficient)
    
    ## Quick Start
    1. Check `/languages` to see available translation languages
    2. Upload video with `/upload?translate=true&target_lang=hi`
    3. Poll `/jobs/{job_id}` for status
    4. Download subtitles with `/download/{job_id}/translated`
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware (allow frontend from any origin during development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Global State & Models
# ============================================

# Job tracking
jobs: dict = {}

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class SubtitleFormat(str, Enum):
    SRT = "srt"
    VTT = "vtt"

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str
    created_at: str
    progress: Optional[float] = None
    original_subtitles: Optional[str] = None
    translated_subtitles: Optional[str] = None
    error: Optional[str] = None

class TranslateRequest(BaseModel):
    text: str
    source_lang: str = "en"
    target_lang: str = "hi"

class TranslateResponse(BaseModel):
    original: str
    translated: str
    source_lang: str
    target_lang: str

class HealthResponse(BaseModel):
    status: str
    whisper_model: str
    whisper_device: str
    nmt_available: bool
    available_languages: List[str]
    loaded_languages: List[str]

# ============================================
# Initialize Components (Lazy Loading)
# ============================================

_audio_processor: Optional[AudioProcessor] = None
_transcriber: Optional[Transcriber] = None
_translator: Optional[Translator] = None
_subtitle_generator: Optional[SubtitleGenerator] = None

def get_audio_processor() -> AudioProcessor:
    global _audio_processor
    if _audio_processor is None:
        _audio_processor = AudioProcessor()
    return _audio_processor

def get_transcriber() -> Transcriber:
    global _transcriber
    if _transcriber is None:
        _transcriber = Transcriber()
    return _transcriber

def get_translator() -> Translator:
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator

def get_subtitle_generator() -> SubtitleGenerator:
    global _subtitle_generator
    if _subtitle_generator is None:
        _subtitle_generator = SubtitleGenerator()
    return _subtitle_generator

# ============================================
# Background Tasks
# ============================================

async def process_video_task(job_id: str, video_path: str, translate: bool, target_lang: str, format: str):
    """Background task to process video and generate subtitles."""
    try:
        jobs[job_id]["status"] = JobStatus.PROCESSING
        jobs[job_id]["progress"] = 0.1
        
        audio_processor = get_audio_processor()
        transcriber = get_transcriber()
        translator = get_translator()
        subtitle_generator = get_subtitle_generator()
        
        video_name = Path(video_path).stem
        
        # Step 1: Extract audio (10%)
        jobs[job_id]["message"] = "Extracting audio..."
        audio_path = audio_processor.convert_video_to_audio(video_path)
        jobs[job_id]["progress"] = 0.2
        
        # Step 2: Transcribe (20% -> 70%)
        jobs[job_id]["message"] = "Transcribing audio..."
        transcriptions = await asyncio.to_thread(
            transcriber.transcribe_full_audio,
            audio_path,
            config.SOURCE_LANGUAGE
        )
        jobs[job_id]["progress"] = 0.7
        
        # Step 3: Generate original subtitles
        jobs[job_id]["message"] = "Generating subtitles..."
        original_path = subtitle_generator.generate_subtitles(
            transcriptions,
            f"{video_name}_original",
            format=format
        )
        jobs[job_id]["original_subtitles"] = str(original_path)
        jobs[job_id]["progress"] = 0.8
        
        # Step 4: Translate if requested
        if translate and translator.is_available(target_lang):
            from src.nmt.languages import get_language_name
            lang_name = get_language_name(target_lang)
            jobs[job_id]["message"] = f"Translating to {lang_name}..."
            translated_transcriptions = await asyncio.to_thread(
                translator.translate_subtitles,
                transcriptions,
                "en",
                target_lang
            )
            
            translated_path = subtitle_generator.generate_subtitles(
                translated_transcriptions,
                f"{video_name}_{target_lang}",
                format=format
            )
            jobs[job_id]["translated_subtitles"] = str(translated_path)
        
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["status"] = JobStatus.COMPLETED
        jobs[job_id]["message"] = "Processing complete!"
        
        # Cleanup uploaded video
        try:
            os.remove(video_path)
        except:
            pass
            
    except Exception as e:
        jobs[job_id]["status"] = JobStatus.FAILED
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["message"] = f"Error: {str(e)}"

# ============================================
# API Endpoints
# ============================================

@app.get("/", tags=["Info"])
async def root():
    """API root - returns basic info."""
    return {
        "name": "Subtitle Generator API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Check API health and component status."""
    translator = get_translator()
    
    return HealthResponse(
        status="healthy",
        whisper_model=config.WHISPER_MODEL_SIZE,
        whisper_device=config.WHISPER_DEVICE,
        nmt_available=translator.is_available(),
        available_languages=translator.get_available_languages(),
        loaded_languages=translator.get_loaded_languages()
    )

@app.post("/upload", tags=["Subtitles"])
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    translate: bool = Query(True, description="Translate subtitles"),
    target_lang: str = Query("hi", description="Target language code (as, bn, gu, hi, kn, ml, mr, or, pa, ta, te)"),
    format: SubtitleFormat = Query(SubtitleFormat.SRT, description="Subtitle format")
):
    """
    Upload a video file for subtitle generation.
    
    Returns a job ID that can be used to check processing status.
    
    - **file**: Video file (MP4, AVI, MKV, MOV, WebM)
    - **translate**: Whether to translate subtitles
    - **target_lang**: Target language code (default: hi for Hindi)
      - Available: as, bn, gu, hi, kn, ml, mr, or, pa, ta, te
    - **format**: Output format (srt or vtt)
    """
    # Validate file type
    allowed_extensions = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())[:8]
    
    # Save uploaded file
    upload_dir = Path(config.TEMP_DIR) / "uploads"
    upload_dir.mkdir(exist_ok=True)
    
    video_path = upload_dir / f"{job_id}_{file.filename}"
    
    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Create job record
    jobs[job_id] = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "message": "Job queued for processing",
        "created_at": datetime.now().isoformat(),
        "progress": 0.0,
        "original_subtitles": None,
        "translated_subtitles": None,
        "error": None,
        "filename": file.filename
    }
    
    # Start background processing
    background_tasks.add_task(
        process_video_task,
        job_id,
        str(video_path),
        translate,
        target_lang,
        format.value
    )
    
    return {
        "job_id": job_id,
        "message": "Video uploaded successfully. Processing started.",
        "status_url": f"/jobs/{job_id}"
    }

@app.get("/jobs/{job_id}", response_model=JobResponse, tags=["Jobs"])
async def get_job_status(job_id: str):
    """Get the status of a subtitle generation job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return JobResponse(**job)

@app.get("/jobs", tags=["Jobs"])
async def list_jobs():
    """List all jobs."""
    return {
        "count": len(jobs),
        "jobs": [
            {
                "job_id": j["job_id"],
                "status": j["status"],
                "filename": j.get("filename", "unknown"),
                "created_at": j["created_at"]
            }
            for j in jobs.values()
        ]
    }

@app.get("/download/{job_id}/{file_type}", tags=["Downloads"])
async def download_subtitles(
    job_id: str,
    file_type: str = "original"  # "original" or "translated"
):
    """
    Download generated subtitle file.
    
    - **job_id**: Job ID from upload
    - **file_type**: "original" or "translated"
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400, 
            detail=f"Job not completed. Status: {job['status']}"
        )
    
    if file_type == "original":
        file_path = job.get("original_subtitles")
    elif file_type == "translated":
        file_path = job.get("translated_subtitles")
    else:
        raise HTTPException(status_code=400, detail="Invalid file_type")
    
    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="Subtitle file not found")
    
    return FileResponse(
        path=file_path,
        filename=Path(file_path).name,
        media_type="text/plain"
    )

@app.get("/languages", tags=["Translation"])
async def get_languages():
    """
    Get available translation languages.
    
    Returns:
    - **supported**: All languages the system can potentially translate to
    - **available**: Languages with trained models currently available
    - **loaded**: Languages currently loaded in memory
    """
    translator = get_translator()
    
    from src.nmt.languages import SUPPORTED_LANGUAGES, get_language_name
    
    return {
        "supported": [
            {"code": code, "name": get_language_name(code)}
            for code in translator.get_supported_languages()
        ],
        "available": [
            {"code": code, "name": get_language_name(code)}
            for code in translator.get_available_languages()
        ],
        "loaded": translator.get_loaded_languages()
    }

@app.post("/translate", response_model=TranslateResponse, tags=["Translation"])
async def translate_text(request: TranslateRequest):
    """
    Translate a single text string.
    
    - **text**: Text to translate
    - **source_lang**: Source language (default: en)
    - **target_lang**: Target language (default: hi)
    """
    translator = get_translator()
    
    if not translator.is_available(request.target_lang):
        available = translator.get_available_languages()
        if available:
            raise HTTPException(
                status_code=503,
                detail=f"No model for '{request.target_lang}'. Available: {', '.join(available)}"
            )
        else:
            raise HTTPException(
                status_code=503,
                detail="No translation models available. Copy models with: bash scripts/copy_models.sh"
            )
    
    translated = await asyncio.to_thread(
        translator.translate,
        request.text,
        request.source_lang,
        request.target_lang
    )
    
    return TranslateResponse(
        original=request.text,
        translated=translated,
        source_lang=request.source_lang,
        target_lang=request.target_lang
    )

@app.post("/translate/batch", tags=["Translation"])
async def translate_batch(
    texts: list[str],
    source_lang: str = "en",
    target_lang: str = "hi"
):
    """
    Translate multiple texts at once (more efficient for batch processing).
    """
    translator = get_translator()
    
    if not translator.is_available(target_lang):
        available = translator.get_available_languages()
        if available:
            raise HTTPException(
                status_code=503,
                detail=f"No model for '{target_lang}'. Available: {', '.join(available)}"
            )
        else:
            raise HTTPException(
                status_code=503,
                detail="No translation models available. Copy models with: bash scripts/copy_models.sh"
            )
    
    translated = await asyncio.to_thread(
        translator.translate_batch,
        texts,
        source_lang,
        target_lang
    )
    
    return {
        "count": len(texts),
        "translations": [
            {"original": orig, "translated": trans}
            for orig, trans in zip(texts, translated)
        ]
    }

@app.delete("/jobs/{job_id}", tags=["Jobs"])
async def delete_job(job_id: str):
    """Delete a job and its associated files."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Delete subtitle files
    for key in ["original_subtitles", "translated_subtitles"]:
        file_path = job.get(key)
        if file_path and Path(file_path).exists():
            try:
                os.remove(file_path)
            except:
                pass
    
    del jobs[job_id]
    
    return {"message": f"Job {job_id} deleted"}

# ============================================
# Startup & Shutdown Events
# ============================================

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup (optional - for faster first request)."""
    print("🚀 Starting Subtitle Generator API...")
    
    # Ensure directories exist
    (Path(config.TEMP_DIR) / "uploads").mkdir(parents=True, exist_ok=True)
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Pre-load models (optional - comment out for lazy loading)
    # get_transcriber()
    # get_translator()
    
    print("✅ API ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("🛑 Shutting down...")

# ============================================
# Run with: uvicorn api:app --reload
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
