import os
from celery import Celery
import ssl
from SearchMultipleSongCloud import recognize_multiple_songs_supabase, connect_to_db, download_file_from_url

# Initialize Celery, linking it to your Redis instance
celery_app = Celery('tasks')

celery_app.conf.update(
    broker_url=os.environ.get('REDIS_URL'),
    result_backend=os.environ.get('REDIS_URL'),
    # Add these SSL settings for Heroku Redis
    broker_use_ssl={'ssl_cert_reqs': ssl.CERT_NONE},
    redis_backend_use_ssl={'ssl_cert_reqs': ssl.CERT_NONE}
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
