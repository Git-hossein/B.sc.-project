import os
import csv
import yt_dlp
import subprocess
import numpy as np
import soundfile as sf
import wav2clip

# Base dataset path
base_path = r"D:\Bsc.Thesis_Datasets\vggsound"

# Create separate folders for full and trimmed clips
full_videos_path = os.path.join(base_path, "full_videos")
trimmed_videos_path = os.path.join(base_path, "trimmed_videos")
video_frames_path = os.path.join(base_path, "frames")
audios_path = os.path.join(base_path, "audios")
audio_embeddings_path = os.path.join(base_path, "audio_embeddings")
os.makedirs(full_videos_path, exist_ok=True)
os.makedirs(trimmed_videos_path, exist_ok=True)
os.makedirs(audios_path, exist_ok=True)
os.makedirs(audio_embeddings_path, exist_ok=True)

# Log files path
download_and_trim_log_file = ("./Logs_download_trim.csv")
extract_audio_log_file     = ("./Logs_extract_audio.csv")
embeddings_audio_log_file   = ("./Logs_audio_embeddings.csv")

# Create log file with headers if it doesn't exist
if not os.path.exists(download_and_trim_log_file):
    with open(download_and_trim_log_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "download", "trim"])

if not os.path.exists(extract_audio_log_file):
    with open(extract_audio_log_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "extract_audio_status"])

if not os.path.exists(embeddings_audio_log_file):
    with open(embeddings_audio_log_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "audio_embedding_status"])

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


# download_and_trim_videos(videos, full_videos_path, trimmed_videos_path, download_and_trim_log_file, clip_length=10)

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

       # Check if frames already exist
        existing_frames = [f for f in os.listdir(output_folder) if f.startswith("frame_") and f.endswith(".jpg")]
        if existing_frames:
            print(f"⏭ Frames appear to already exist for {video_id}, skipping...")
            continue

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


# extract_frames_from_videos(trimmed_videos_path, video_frames_path, fps=5)

def extract_audio_from_videos(trimmed_videos_dir, audios_dir, sample_rate=16000, log_file=extract_audio_log_file):
    """
    Extracts audio from trimmed videos as .wav files (mono, resampled to sample_rate)
    and logs failures.

    Args:
        trimmed_videos_dir (str): Path to folder with trimmed .mp4 videos.
        audios_dir (str): Path to save extracted audio files.
        sample_rate (int): Target sample rate for audio (default 16 kHz for Wav2CLIP).
        log_file (str): CSV file to log failures.
    """
    
    videos = [f for f in os.listdir(trimmed_videos_dir) if f.endswith(".mp4")]
    print(f"Found {len(videos)} trimmed videos for audio extraction.")
    
    for video in videos:
        video_path = os.path.join(trimmed_videos_dir, video)
        video_id = os.path.splitext(video)[0]
        audio_file = os.path.join(audios_dir, f"{video_id}.wav")
        
        if os.path.exists(audio_file):
            print(f"Audio for {video_id} already exists. Skipping.")
            continue
        
        cmd = [
            r"C:\Users\hosse\Downloads\ffmpeg-8.0-essentials_build\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe",
            "-y",  # overwrite if exists
            "-i", video_path,
            "-ac", "1",  # mono
            "-ar", str(sample_rate),  # resample
            audio_file,
            "-hide_banner",
            "-loglevel", "error"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Extracted audio for {video_id} successfully.")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to extract audio for {video_id}: {e}")
            # Log failure
            with open(log_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([video_id, "failed"])

# extract_audio_from_videos(trimmed_videos_path, audios_path, sample_rate=16000)



# ============= The embeddings woll now be computed for audio as well as video ====================


# Load Wav2CLIP model once (clip-level)
wav2clip_model = wav2clip.get_model()

def extract_audio_embeddings(audios_dir, embeddings_dir, log_file="./Logs_audio_embeddings.csv"):
    """
    Extracts clip-level Wav2CLIP embeddings for each .wav audio file.
    Saves each embedding as a .npy file: <video_id>.npy
    Logs failures in a CSV.
    """

    audio_files = [f for f in os.listdir(audios_dir) if f.endswith(".wav")]
    print(f"Found {len(audio_files)} audio files for embedding.")

    for audio_file in audio_files:
        video_id = os.path.splitext(audio_file)[0]
        audio_path = os.path.join(audios_dir, audio_file)
        embedding_path = os.path.join(embeddings_dir, f"{video_id}.npy")

        # Skip existing
        if os.path.exists(embedding_path):
            print(f"Embedding for {video_id} already exists. Skipping.")
            continue

        try:
            # Load waveform (Wav2CLIP expects raw PCM float32)
            audio_waveform, sr = sf.read(audio_path)

            # If stereo → convert to mono by averaging channels
            if len(audio_waveform.shape) == 2:
                audio_waveform = audio_waveform.mean(axis=1)

            # Ensure float32
            audio_waveform = audio_waveform.astype(np.float32)

            # Compute embedding
            embedding = wav2clip.embed_audio(audio_waveform, wav2clip_model)

            # Save
            np.save(embedding_path, embedding)

            print(f"✔ Embedded audio for {video_id}")

        except Exception as e:
            print(f"⚠️ Failed embedding for {video_id}: {e}")
            with open(log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([video_id, "failed"])

extract_audio_embeddings(audios_path, audio_embeddings_path)