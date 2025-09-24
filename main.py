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

def recognize_single_song_supabase(conn, file_path, duration=20):
    """Recognizes a single song from a short audio clip using Supabase."""
    print("--- Single Song Detection (Supabase) ---")
    result = recognize_from_supabase(conn, file_path, duration=duration)
    
    if result:
        song, score, offset = result
        position_str = format_time_position(offset)
        print(f"✅ Match Found: '{song}'")
        print(f"  Position in song: {position_str} (at {offset:.2f} seconds)")
        print(f"  Confidence Score: {score}")
        return {"song": song, "confidence": score, "offset": offset, "position": position_str}
    else:
        print("❌ No match found in the database.")
        return None

def recognize_multiple_songs_supabase(conn, file_path, segment_duration=30, min_confidence=10, overlap=5):
    """Detects multiple songs in a single audio file using Supabase."""
    print("--- Multiple Song Detection (Supabase) ---")
    
    import librosa
    y, sr = librosa.load(file_path, sr=None)
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    print(f"Analyzing file of duration {total_duration:.2f} seconds...")
    
    detected_songs = []
    current_pos = 0
    last_detected_song = None
    
    while current_pos < total_duration:
        result = recognize_from_supabase(conn, file_path, current_pos, segment_duration)
        
        if result and result[1] >= min_confidence:
            song, score, offset = result
            song_start_in_file = current_pos + offset
            
            if last_detected_song and last_detected_song["song"] == song:
                last_detected_song["end_time"] = current_pos + segment_duration
                last_detected_song["confidence"] = max(last_detected_song["confidence"], score)
            else:
                song_info = {
                    "song": song,
                    "start_time": song_start_in_file,
                    "end_time": current_pos + segment_duration,
                    "confidence": score
                }
                detected_songs.append(song_info)
                last_detected_song = song_info
                
                print(f"Detected '{song}' starting at {format_time_position(song_start_in_file)} (confidence: {score})")
        
        current_pos += segment_duration - overlap

    # Merge adjacent segments of the same song
    merged_detections = []
    if detected_songs:
        merged_detections.append(detected_songs[0])
        for i in range(1, len(detected_songs)):
            current_song = detected_songs[i]
            prev_song = merged_detections[-1]
            if current_song["song"] == prev_song["song"]:
                prev_song["end_time"] = current_song["end_time"]
                prev_song["confidence"] = max(prev_song["confidence"], current_song["confidence"])
            else:
                merged_detections.append(current_song)
    
    return merged_detections

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
