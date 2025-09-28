import os
import tempfile
import json
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import psycopg2
from typing import Optional, List
from pydantic import BaseModel
from celery.result import AsyncResult
from tasks import process_song_recognition # Import your new Celery task
import cloudinary
import cloudinary.uploader
from celery.result import AsyncResult
from tasks import process_song_recognition, celery_app

# Load environment variables

# Assume your combined code is in a file named 'song_recognizer.py'
from SearchMultipleSongCloud import connect_to_db, recognize_from_supabase, format_time_position

app = FastAPI(title="Song Recognition API", description="API for recognizing songs from audio files")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this according to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cloudinary.config(
  cloud_name = os.environ.get('CLOUDINARY_CLOUD_NAME'),
  api_key = os.environ.get('CLOUDINARY_API_KEY'),
  api_secret = os.environ.get('CLOUDINARY_API_SECRET')
)

# Response models
class SingleSongResult(BaseModel):
    song: str
    position: str
    confidence: float

class RecognitionResponse(BaseModel):
    results: List[SingleSongResult]

class HealthResponse(BaseModel):
    status: str
    database: Optional[str] = None
    error: Optional[str] = None

@app.post("/recognize")
async def submit_recognition_job(audio_file: UploadFile = File(...)):
    """
    This endpoint now runs instantly. It just uploads the file and
    creates a background job.
    """
    # 1. Upload the file to Cloudinary for persistent storage
    upload_result = cloudinary.uploader.upload(
        audio_file.file,
        resource_type="video" # Use 'video' or 'raw' for audio files
    )
    file_url = upload_result.get('secure_url')

    # 2. Create the background task with the file's URL
    task = process_song_recognition.delay(file_url)

    # 3. Immediately return the task's ID
    return JSONResponse(status_code=202, content={'job_id': task.id})

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """
    This endpoint allows Laravel to check the status of the job.
    """
    task_result = AsyncResult(job_id, app=celery_app)

    if task_result.ready():
        if task_result.successful():
            return {'status': 'COMPLETED', 'results': task_result.get()}
        else:
            return {'status': 'FAILED', 'error': str(task_result.info)}
    else:
        return {'status': 'PENDING'}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify the API is running"""
    try:
        conn = connect_to_db()
        if conn:
            conn.close()
            return HealthResponse(status="healthy", database="connected")
        else:
            return HealthResponse(status="unhealthy", database="disconnected")
    except Exception as e:
        return HealthResponse(status="unhealthy", error=str(e))

# Optional: Add a root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Song Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "/recognize": "POST - Upload audio file for song recognition",
            "/health": "GET - Check API health status",
            "/docs": "GET - API documentation (Swagger UI)",
            "/redoc": "GET - API documentation (ReDoc)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
