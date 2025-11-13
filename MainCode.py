import os
import csv
import yt_dlp
import subprocess

# Base dataset path
base_path = r"D:\Bsc.Thesis_Datasets\vggsound"

# Create separate folders for full and trimmed clips
full_videos_path = os.path.join(base_path, "full_videos")
trimmed_videos_path = os.path.join(base_path, "trimmed_videos")
os.makedirs(full_videos_path, exist_ok=True)
os.makedirs(trimmed_videos_path, exist_ok=True)

# Log file path
log_file = ("./Logs_download_trim.csv")

# Create log file with headers if it doesn't exist
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "download", "trim"])

# List of example videos with start time and labels
videos = [
    ("--0PQM4-hqg", 30, "waterfall_burbling"),
    ("--56QUhyDQM", 185, "playing_tennis"),
    ("--5OkAjCI7g", 40, "people_belly_laughing")
]

for video_id, start_sec, label in videos:
    full_file = os.path.join(full_videos_path, f"{video_id}_full.mp4")
    clip_file = os.path.join(trimmed_videos_path, f"{video_id}.mp4")

    download_status = "failed"
    trim_status = "unknown"

    # Download full video if not already present
    if not os.path.exists(full_file):
        print(f"Downloading full video {video_id}...")
        ydl_opts = {'format': 'mp4', 'outtmpl': full_file}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
            print(f"Downloaded full video {video_id} successfully.")
            download_status = "success"
        except Exception as e:
            print(f"Failed to download {video_id}: {e}")
    else:
        print(f"Full video {video_id} already exists. Skipping download.")
        download_status = "success"

    # Trim 10-second clip using ffmpeg if not already present
    if download_status == "success":
        if not os.path.exists(clip_file):
            print(f"Trimming {video_id} to 10 seconds...")
            try:
                subprocess.run([
                    "ffmpeg",
                    "-y",  # overwrite if exists
                    "-ss", str(start_sec),
                    "-i", full_file,
                    "-t", "10",
                    "-c", "copy",
                    clip_file
                ], check=True)
                print(f"Trimmed {video_id} successfully.")
                trim_status = "success"
            except subprocess.CalledProcessError as e:
                print(f"Failed to trim {video_id}: {e}")
                trim_status = "failed"
        else:
            print(f"Trimmed clip {video_id} already exists. Skipping trimming.")
            trim_status = "success"

    # Log the failures TODO: make a fucntion which prevents duplicate entries
    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
        if download_status == "failed" or trim_status in ("failed", "unknown"):    
            writer = csv.writer(f)
            writer.writerow([video_id, download_status, trim_status])


