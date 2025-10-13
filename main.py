# main.py

from tasks import run_sharepoint_download
import os

if _name_ == '__main__':
    # Mencetak pesan log untuk menandakan job dimulai
    # os.getenv('DYNO', 'local') akan menampilkan nama dyno jika berjalan di Heroku
    print(f"Memulai eksekusi skrip pada dyno: {os.getenv('DYNO', 'local')}")
    
    # Memanggil fungsi utama yang berisi semua logika dari file tasks.py
    run_sharepoint_download()
    
    print("Eksekusi skrip selesai.")
