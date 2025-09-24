import os
from celery import Celery
from SearchMultipleSongCloud import recognize_multiple_songs_supabase, connect_to_db, download_file_from_url

# Initialize Celery, linking it to your Redis instance
celery_app = Celery(
    'tasks',
    broker=os.environ.get('REDIS_URL'),
    backend=os.environ.get('REDIS_URL')
)

@celery_app.task
def process_song_recognition(file_url):
    """
    This is the background task.
    """
    temp_path = None # Ensure temp_path is defined
    conn = connect_to_db()
    if not conn:
        raise Exception("Database connection failed in worker.")

    try:
        # Download the file from Cloudinary to a temporary path
        temp_path = download_file_from_url(file_url)
        # Run the heavy recognition logic
        results = recognize_multiple_songs_supabase(conn, temp_path)
        return results
    finally:
        if conn:
            conn.close()
        # Clean up the downloaded file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
