# tasks.py

from office365_api import SharePoint
import os
from pathlib import PurePath

# --- Konfigurasi ---
# Membaca konfigurasi dari Environment Variables yang diatur di Heroku.
# Nilai default (setelah koma) digunakan jika variabel tidak ditemukan.
FOLDER_NAME = os.getenv("FOLDER_NAME", "General")
FOLDER_DEST = os.getenv("FOLDER_DEST", "./downloaded_files")
CRAWL_FOLDERS = os.getenv("CRAWL_FOLDERS", "Yes")

# --- Fungsi Helper ---

def save_file(file_n, file_obj, subfolder):
    """Menyimpan objek file ke dalam direktori lokal."""
    # Pastikan FOLDER_DEST sudah ada
    if not os.path.exists(FOLDER_DEST):
        os.makedirs(FOLDER_DEST)
        
    dir_path = PurePath(FOLDER_DEST, subfolder)
    file_dir_path = PurePath(dir_path, file_n)
    
    try:
        with open(file_dir_path, 'wb') as f:
            f.write(file_obj)
        print(f"  -> Berhasil menyimpan: {file_dir_path}")
    except Exception as e:
        print(f"  -> GAGAL menyimpan {file_dir_path}: {e}")

def create_dir(path):
    """Membuat direktori lokal jika belum ada."""
    dir_path = PurePath(FOLDER_DEST, path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Membuat direktori: {dir_path}")

def get_file(file_n, folder):
    """Mengunduh satu file dari SharePoint dan menyimpannya."""
    try:
        file_obj = SharePoint().download_file(file_n, folder)
        save_file(file_n, file_obj, folder)
    except Exception as e:
        print(f"  -> GAGAL mengunduh {file_n} dari folder {folder}: {e}")

def get_files_from_folder(folder):
    """Mengambil semua file dari satu folder spesifik."""
    print(f"\nMencari file di folder SharePoint: '{folder}'...")
    try:
        files_list = SharePoint()._get_files_list(folder)
        if not files_list:
            print("  -> Tidak ada file ditemukan.")
            return
        for file in files_list:
            get_file(file.name, folder)
    except Exception as e:
        print(f"  -> GAGAL mengambil daftar file dari folder {folder}: {e}")

def get_subfolders(folder):
    """Mengambil daftar subfolder dari satu folder spesifik."""
    subfolders = []
    try:
        folder_obj = SharePoint().get_folder_list(folder)
        for subfolder_obj in folder_obj:
            # Menggabungkan path folder induk dengan nama subfolder
            full_subfolder_path = '/'.join([folder, subfolder_obj.name])
            subfolders.append(full_subfolder_path)
    except Exception as e:
        print(f"  -> GAGAL mengambil daftar subfolder dari {folder}: {e}")
    return subfolders

# --- Fungsi Utama (Tugas Inti) ---

def run_sharepoint_download():
    """
    Ini adalah fungsi utama yang akan dieksekusi.
    Fungsi ini mengorkestrasi seluruh proses download.
    """
    print("--- Memulai Tugas Download SharePoint ---")
    
    folders_to_process = []

    if CRAWL_FOLDERS.lower() == 'yes':
        print(f"Mode Crawl aktif. Memindai semua subfolder dari '{FOLDER_NAME}'...")
        # Daftar folder yang akan dipindai, dimulai dengan folder utama
        pending_folders = [FOLDER_NAME]
        processed_folders = []

        while pending_folders:
            current_folder = pending_folders.pop(0)
            if current_folder in processed_folders:
                continue
            
            print(f"Memindai subfolder di: {current_folder}")
            subfolders = get_subfolders(current_folder)
            pending_folders.extend(subfolders)
            processed_folders.append(current_folder)
        
        folders_to_process = processed_folders
    else:
        print(f"Mode Crawl tidak aktif. Hanya memproses folder: '{FOLDER_NAME}'")
        folders_to_process = [FOLDER_NAME]

    print("\nDaftar lengkap folder yang akan diproses:")
    for f in folders_to_process:
        print(f"- {f}")

    # Memulai proses pembuatan direktori dan download file
    for folder in folders_to_process:
        create_dir(folder)
        get_files_from_folder(folder)

    print("\n--- Tugas Download SharePoint Selesai ---")
