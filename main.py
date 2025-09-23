import os
import tempfile
import json
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import psycopg2
from typing import Optional, List
from pydantic import BaseModel

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

@app.post("/recognize", response_model=RecognitionResponse)
async def recognize(
    audio_file: UploadFile = File(..., description="Audio file to recognize"),
    mode: str = Form(default="single", description="Recognition mode: 'single' or 'multiple'"),
    segment_duration: int = Form(default=30, description="Duration of each segment for multiple song detection"),
    min_confidence: int = Form(default=5, description="Minimum confidence score for detection"),
    overlap: int = Form(default=5, description="Overlap between segments in seconds")
):
    """
    Recognize songs from an uploaded audio file.
    
    - **audio_file**: The audio file to analyze
    - **mode**: 'single' for single song detection, 'multiple' for multiple songs
    - **segment_duration**: Duration of each segment for analysis (multiple mode only)
    - **min_confidence**: Minimum confidence score to consider a match valid
    - **overlap**: Overlap between segments in seconds (multiple mode only)
    """
    
    # Validate file
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No selected file")
    
    # Check file size (100MB limit)
    content = await audio_file.read()
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 100MB")
    
    # Reset file pointer
    await audio_file.seek(0)

    # Establish database connection
    conn = connect_to_db()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    # Save the uploaded file to a temporary location
    try:
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, audio_file.filename)
        
        # Write file content to temporary file
        with open(temp_path, "wb") as temp_file:
            content = await audio_file.read()
            temp_file.write(content)

        results = None
        if mode == "single":
            result = recognize_single_song_supabase(conn, temp_path)
            if result:
                results = [result]
        elif mode == "multiple":
            results = recognize_multiple_songs_supabase(
                conn,
                temp_path,
                segment_duration=segment_duration,
                min_confidence=min_confidence,
                overlap=overlap
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid mode. Use 'single' or 'multiple'")
        
        # Clean up the temporary file
        os.remove(temp_path)

        # Format the results for JSON output
        formatted_results = []
        if results:
            for res in results:
                # Handle both single and multiple song detection formats
                if "offset" in res:  # Single song detection format
                    formatted_results.append({
                        "song": res.get("song"),
                        "position": res.get("position"),  
                        "confidence": res.get("confidence")
                    })
                else:  # Multiple song detection format
                    # Format the position information
                    start_time = res.get("start_time", 0)
                    end_time = res.get("end_time", 0)
                    position_str = f"{format_time_position(start_time)} - {format_time_position(end_time)}"
                    
                    formatted_results.append({
                        "song": res.get("song"),
                        "position": position_str,
                        "confidence": res.get("confidence")
                    })
        
        return RecognitionResponse(results=formatted_results)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during recognition: {e} WHAT IS THE ERRORRRR RAGHHHHH")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Always close the database connection
        if conn:
            conn.close()

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