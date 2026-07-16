import os 
import yt_dlp
import csv
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from path_settings import paths_config
from typing import Iterator, Literal, Any


def vggsound_training_videos_generator(
        vggsound_path, 
        nmany=1033, 
        start=0, 
        type: Literal["train", "test"] = "train"
        ) -> Iterator[tuple[str, int, str]]:
    """
    Yields (video_id, start_sec, label) for 'train' rows.
    Skips the first `start` training rows before yielding.
    """
    print("extracting training videos from csv...")
    with open(vggsound_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        skipped = 0
        yielded = 0
        for row in reader:
            if row[3].strip() != type:
                continue  # skip non-training rows

            # Skip first `start` items
            if skipped < start:
                skipped += 1
                continue
            # Stop once we yield nmany items
            if yielded >= nmany:
                break

            yielded += 1
            yield (row[0], int(row[1]), row[2])  # (YouTube ID, start_sec, caption)



def download_youtube_video(video_id, output_path):
    """Downloads a single video if it doesn't exist."""
    if not os.path.exists(output_path):
        print(f"⬇️ Downloading full video {video_id}...")
        ydl_opts: Any = {
            'format': 'mp4',
            'outtmpl': output_path,
            'cookiesfrombrowser': ('firefox',),
            'retries': 10,
            
            # --- ADD THESE FOR THE 1,000 VIDEO BATCH ---
            'fragment_retries': 10,      # If a single chunk of video drops, it retries that chunk
            'continuedl': True,          # If internet cuts, it picks up where it left off
            'quiet': False,              # Keeps you informed of progress
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
            print(f"✅ Downloaded full video {video_id} successfully.")
            return True
        except Exception as e:
            print(f"Failed to download {video_id}: {e}")
            return False
    else:
        print(f"video {output_path} already exists. Skipping download.")
        return True


def trim_video(input_path, output_path, start_sec, duration = 10):
    """Trims a local video file using ffmpeg."""
    if os.path.exists(output_path):
        print(f"video {os.path.basename(input_path)}  has already been trimmed at {output_path}. Skipping trimming.")
        return True
    
    try:
        print(f"✂️ trimming video {os.path.basename(input_path)}...")
        subprocess.run([
                    "ffmpeg", "-y",
                    "-ss", str(start_sec),     # Fast seek to the start time
                    "-i", input_path,          # Input file
                    "-t", str(duration),       # Duration of the clip
                    "-c:v", "libx264",         # Use H.264 video codec (fixes artifacts)
                    "-crf", "18",              # High quality (18 is nearly lossless, 23 is default)
                    "-preset", "veryfast",     # Encoding speed vs compression trade-off
                    "-c:a", "aac",             # Use AAC audio codec
                    output_path
                ], check=True, capture_output=True)
        print(f"🟢 trimmed full video {os.path.basename(input_path)} successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to trim {os.path.basename(input_path)} : {e}")
        return False
    

def vggsound_batch_down_trim(video_list, full_videos_path, trimmed_videos_path, log_file, clip_length=10):
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

    for video_id, start_sec, label in video_list:
        full_file = os.path.join(full_videos_path, f"{video_id}_full.mp4")
        trim_file = os.path.join(trimmed_videos_path, f"{video_id}.mp4")
        dl_success = False
        trim_success = False

        # Download full video
        print("start downloading and trimming video ")
        dl_success = download_youtube_video(video_id, full_file)

        # Trim video
        if dl_success:
            trim_success = trim_video(full_file, trim_file, start_sec, clip_length)

        # Log failures TODO: write a fucntion to prevent the faulty video being logged multiple times
        if not dl_success or not trim_success:
            with open(log_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([video_id,
                                "success" if dl_success else "failed",
                                "success" if trim_success else "unknown" if not dl_success else "failed"])
                


def check_single_file(vid_path):
    """Worker function to check one file."""
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(vid_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return vid_path.name
    except Exception:
        return vid_path.name
    return None

def check_integrity(full_dir= paths_config.training_full_videos_path, trimmed_dir= paths_config.training_trimmed_videos_path):
    print("🚀 Starting the integrity check...")
    full_path = Path(full_dir)
    trimmed_path = Path(trimmed_dir)

    valid_full = {}
    valid_trimmed = {}
    invalid_naming_full = []
    invalid_naming_trimmed = []

    # --- 1. SCAN FOLDERS (This part is very fast) ---
    for f in full_path.iterdir():
        if f.is_file():
            if f.name.endswith('_full.mp4'):
                valid_full[f.name.replace('_full.mp4', '')] = f
            else:
                invalid_naming_full.append(f.name)

    for f in trimmed_path.iterdir():
        if f.is_file():
            if f.name.endswith('.mp4') and not f.name.endswith('_full.mp4'):
                valid_trimmed[f.name.replace('.mp4', '')] = f
            else:
                invalid_naming_trimmed.append(f.name)

    # --- 2. PRINT SYNC REPORT ---
    full_ids = set(valid_full.keys())
    trimmed_ids = set(valid_trimmed.keys())
    missing_from_trimmed = full_ids - trimmed_ids
    missing_from_full = trimmed_ids - full_ids

    print("\n" + "="*50)
    print(f"📊 DATASET SYNC: Full Folder ({len(full_ids)}) | Trimmed Folder ({len(trimmed_ids)})")
    print("="*50)

    if missing_from_trimmed:
        print(f"⚠️  MISSING IN TRIMMED ({len(missing_from_trimmed)} IDs) : \n{print(missing_from_trimmed)}")
    if missing_from_full:
        print(f"⚠️  MISSING IN FULL ({len(missing_from_full)} IDs) : \n{print(missing_from_full)}")

    if invalid_naming_full or invalid_naming_trimmed:
        print(f"🚫 INVALID NAMES: Full ({len(invalid_naming_full)}) | Trimmed ({len(invalid_naming_trimmed)})")

    # --- 3. THE TURBO HEALTH CHECK (ffprobe) ---
    print("\n" + "-"*50)
    print("🛠️  CORRUPTION CHECK (Parallel ffprobe)")
    print("-" * 50)
    
    all_files = list(valid_full.values()) + list(valid_trimmed.values())
    total_files = len(all_files)
    corrupted = []
    
    print(f"Checking {total_files} files using 16 CPU threads...")

    # We use 16 workers to check 16 files at once
    with ThreadPoolExecutor(max_workers=16) as executor:
        # We wrap the results in a list to wait for them to finish
        results = list(executor.map(check_single_file, all_files))
        
    # Filter out the None results (healthy files) to get the corrupted names
    corrupted = [name for name in results if name is not None]

    if corrupted:
        print(f"❌ Found {len(corrupted)} unreadable files!")
        for c in corrupted:
            print(f"   - {c}")
    else:
        print("✅ All video headers are healthy.")