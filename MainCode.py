import os
import csv
import yt_dlp
import subprocess

# Base dataset path
base_path = r"D:\Bsc.Thesis_Datasets\vggsound"

# Create separate folders for full and trimmed clips
full_videos_path = os.path.join(base_path, "full_videos")
trimmed_videos_path = os.path.join(base_path, "trimmed_videos")
video_frames_path = os.path.join(base_path, "frames")
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

def download_and_trim_videos(videos, full_videos_path, trimmed_videos_path, log_file, clip_length=10):
    """
    Downloads and trims videos from YouTube.

    Args:
        videos (list): List of tuples (video_id, start_sec, label)
        full_videos_path (str): Path to save full videos
        trimmed_videos_path (str): Path to save trimmed clips
        log_file (str): Path to CSV file to log failures
        clip_length (int): Length of trimmed clips in seconds (default 10)
    """
    os.makedirs(full_videos_path, exist_ok=True)
    os.makedirs(trimmed_videos_path, exist_ok=True)

    # Create log file with headers if it doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["video_id", "download", "trim"])

    for video_id, start_sec, label in videos:
        full_file = os.path.join(full_videos_path, f"{video_id}_full.mp4")
        clip_file = os.path.join(trimmed_videos_path, f"{video_id}.mp4")

        download_status = "failed"
        trim_status = "unknown"

        # Download full video
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

        # Trim video
        if download_status == "success":
            if not os.path.exists(clip_file):
                print(f"Trimming {video_id} to {clip_length} seconds...")
                try:
                    subprocess.run([
                        "ffmpeg",
                        "-y",  # overwrite if exists
                        "-ss", str(start_sec),
                        "-i", full_file,
                        "-t", str(clip_length),
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

        # Log failures TODO: write a fucntion to prevent the faulty video being logged multiple times
        if download_status == "failed" or trim_status in ("failed", "unknown"):
            with open(log_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([video_id, download_status, trim_status])


# download_and_trim_videos(videos, full_videos_path, trimmed_videos_path, log_file, clip_length=10)

def extract_frames_from_videos(trimmed_videos_dir, frames_dir, fps=5):
    """
    Extract frames from all trimmed .mp4 videos in `trimmed_videos_dir` and save them 
    into subfolders in `frames_dir`, one subfolder per video.

    Args:
        trimmed_videos_dir (str): Path to the folder containing trimmed .mp4 videos.
        frames_dir (str): Path to save extracted frames.
        fps (int): Number of frames per second to extract (default: 5).
    """
    os.makedirs(frames_dir, exist_ok=True)

    # Only consider trimmed videos
    videos = [f for f in os.listdir(trimmed_videos_dir) if f.endswith(".mp4")]
    print(f"Found {len(videos)} trimmed videos for frame extraction.")

    for video in videos:
        video_path = os.path.join(trimmed_videos_dir, video)
        video_id = os.path.splitext(video)[0]
        output_folder = os.path.join(frames_dir, video_id)
        os.makedirs(output_folder, exist_ok=True)

        # Output pattern for frames
        output_pattern = os.path.join(output_folder, "frame_%04d.jpg")

        # ffmpeg command
        cmd = [
            r"C:\Users\hosse\Downloads\ffmpeg-8.0-essentials_build\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe",
            "-i", video_path,
            "-vf", f"fps={fps}",
            output_pattern,
            "-hide_banner",
            "-loglevel", "error"  # suppress ffmpeg spam
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Extracted frames for {video_id} successfully.")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to extract frames for {video_id}: {e}")


extract_frames_from_videos(trimmed_videos_path, video_frames_path, fps=5)