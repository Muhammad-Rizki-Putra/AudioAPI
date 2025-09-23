import os
import librosa
import numpy as np
from scipy.ndimage import maximum_filter
import argparse
from collections import defaultdict
import tempfile
import glob
import psycopg2             # NEW: Use psycopg2 for PostgreSQL
import uuid                 # NEW: To generate unique varchar IDs for songs
from dotenv import load_dotenv # NEW: To load secrets from .env file
from datetime import datetime
from psycopg2.extras import execute_values

# --- Database Constants ---
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

# --- Core Fingerprinting and Recognition Functions ---

def connect_to_db():
    """Establishes a connection to the Supabase PostgreSQL database."""
    try:
        conn = psycopg2.connect(DB_URL)
        print("Database connection successful.")
        return conn
    except psycopg2.OperationalError as e:
        print(f"🔥 Could not connect to the database: {e}")
        return None

def fingerprint_song(file_path, start_time=0, duration=None):
  """
  Generates a landmark-based fingerprint for an audio file or segment.
  (This is your existing fingerprinting function)
  """
  try:
    if duration is not None:
      y, sr = librosa.load(file_path, offset=start_time, duration=duration)
    else:
      y, sr = librosa.load(file_path)

    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    neighborhood_size = 15
    local_max = maximum_filter(S_db, footprint=np.ones((neighborhood_size, neighborhood_size)), mode='constant')
    detected_peaks = (S_db == local_max)
    amplitude_threshold = -50.0
    peaks = np.where((detected_peaks) & (S_db > amplitude_threshold))
    
    if not peaks[0].any():
      return []

    n_fft = (D.shape[0] - 1) * 2
    peak_freqs_at_peaks = librosa.fft_frequencies(sr=sr, n_fft=n_fft)[peaks[0]]
    peak_times = librosa.frames_to_time(frames=peaks[1], sr=sr, n_fft=n_fft)
    peaks_list = list(zip(peak_times, peak_freqs_at_peaks))
    sorted_peaks = sorted(peaks_list, key=lambda p: p[0])

    fingerprints = []
    TARGET_ZONE_START_TIME = 0.1
    TARGET_ZONE_TIME_DURATION = 0.8
    TARGET_ZONE_FREQ_WIDTH = 200

    for i, anchor_peak in enumerate(sorted_peaks):
      anchor_time, anchor_freq = anchor_peak
      t_min = anchor_time + TARGET_ZONE_START_TIME
      t_max = t_min + TARGET_ZONE_TIME_DURATION
      f_min = anchor_freq - TARGET_ZONE_FREQ_WIDTH
      f_max = anchor_freq + TARGET_ZONE_FREQ_WIDTH
      
      for j in range(i + 1, len(sorted_peaks)):
        target_peak = sorted_peaks[j]
        target_time, target_freq = target_peak
        if target_time > t_max:
          break
        if t_min <= target_time <= t_max and f_min <= target_freq <= f_max:
          time_delta = target_time - anchor_time
          h = hash((anchor_freq, target_freq, time_delta))
          fingerprints.append((h, anchor_time))
          
    return fingerprints

  except Exception as e:
    print(f"Could not process query file {file_path}. Error: {e}")
    return []

def recognize_from_supabase(conn, query_path, start_time=0, duration=None):
    """
    Use Python processing approach for debugging and consistency with SQLite version
    """
    print(f"Fingerprinting query file: {query_path}...")
    query_fingerprints = fingerprint_song(query_path, start_time, duration)

    if not query_fingerprints:
        print("Could not generate fingerprints for the query clip.")
        return None

    try:
        cursor = conn.cursor()

        # Process in chunks like SQLite version
        query_hashes = [str(h) for h, _ in query_fingerprints]
        query_hash_to_time = {str(h): t for h, t in query_fingerprints}
        
        CHUNK_SIZE = 900
        db_matches = []
        
        for i in range(0, len(query_hashes), CHUNK_SIZE):
            chunk = query_hashes[i:i + CHUNK_SIZE]
            placeholders = ', '.join(['%s'] * len(chunk))
            
            sql_query = f"""
                SELECT f."intHash", s."szsongtitle", f."offSetTime"
                FROM public."CALA_MDM_FINGERPRINTS" AS f
                JOIN public."cala_mdm_songs" AS s ON s."szsongid" = f."szSongID"
                WHERE f."intHash"::text IN ({placeholders})
            """
            
            cursor.execute(sql_query, chunk)
            db_matches.extend(cursor.fetchall())

        # Process matches exactly like SQLite version
        if not db_matches:
            print("No database matches found.")
            return None
        
        matches = defaultdict(int)
        for db_hash, song_name, db_timestamp in db_matches:
            if str(db_hash) in query_hash_to_time:
                query_timestamp = query_hash_to_time[str(db_hash)]
                offset = round(db_timestamp - query_timestamp, 2)
                key = (song_name, offset)
                matches[key] += 1
        
        if not matches:
            print("No valid matches after processing.")
            return None
            
        best_match = max(matches.items(), key=lambda item: item[1])
        (song_name, offset), score = best_match
        
        print(f"Debug: Found {len(db_matches)} raw matches, {len(matches)} grouped matches")
        print(f"Debug: Best match score: {score}")
        
        return (song_name, score, offset)

    except psycopg2.Error as e:
        print(f"❌ A database error occurred: {e}")
        conn.rollback()
        return None
    finally:
        cursor.close()

def format_time_position(seconds):
  """Convert seconds to minutes:seconds format"""
  minutes = int(seconds // 60)
  seconds_remaining = int(seconds % 60)
  return f"{minutes}:{seconds_remaining:02d}"


if __name__ == "__main__":
    db_connection = connect_to_db()

    if db_connection:
        try:
            query_audio_file = "D:\Berkas_Rizki\Semester_7\Magang\songs\Hindia\hindia - everything u are.mp3" # 👈 CHANGE THIS
            result = recognize_from_supabase(db_connection, query_audio_file, duration=100)

            if result:
                song_title, confidence, time_offset = result
                print("\n--- Match Found! ---")
                print(f"🎵 Song: {song_title}")
                print(f"⭐ Confidence Score: {confidence}")
                print(f"🕒 Time Offset: {time_offset:.2f} seconds")
            else:
                print("\n--- No Match Found ---")

        finally:
            db_connection.close()
            print("Database connection closed.")