import os
import csv
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Iterator, Any
import yt_dlp
from wakepy import keep
from path_settings import paths_config

log_lock = threading.Lock()

def vggsound_training_videos_generator(
        vggsound_path, 
        nmany=1033, 
        start=0, 
        type: Literal["train", "test"] = "train"
        ) -> Iterator[tuple[str, int, str]]:
    print("extracting training videos from csv...")
    with open(vggsound_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        skipped = 0
        yielded = 0
        for row in reader:
            if row[3].strip() != type:
                continue
            if skipped < start:
                skipped += 1
                continue
            if yielded >= nmany:
                break
            yielded += 1
            yield (row[0], int(row[1]), row[2])

log_lock = threading.Lock()

def download_youtube_video_stable(video_id, output_path):
    """Uses the linear version's stable full-download logic, but quiet for threads."""
    if not os.path.exists(output_path):
        ydl_opts: Any = {
            'format': 'mp4',
            'outtmpl': output_path,
            'cookiesfrombrowser': ('firefox',),
            'retries': 10,
            'fragment_retries': 10,
            'continuedl': True,
            'quiet': True, # disable progress bar
            'noprogress': True, 
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
            return True
        except Exception as e:
            print(f"❌ Failed to download {video_id}: {e}")
            return False
    else:
        return True

def trim_video_stable(input_path, output_path, start_sec, duration=10):
    """Exactly matches your linear ffmpeg settings for uniform ML encoding."""
    if os.path.exists(output_path):
        return True
    
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(start_sec),     
            "-i", input_path,          
            "-t", str(duration),       
            "-c:v", "libx264",         
            "-crf", "18",              
            "-preset", "veryfast",     
            "-c:a", "aac",             
            output_path
        ], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to trim {os.path.basename(input_path)}")
        return False

def process_single_video_p_stable(video_info, full_videos_path, trimmed_videos_path, log_file, clip_length):
    """Downloads full, trims, and cleans up after itself to save disk space."""
    video_id, start_sec, label = video_info
    full_file = os.path.join(full_videos_path, f"{video_id}_full.mp4")
    trim_file = os.path.join(trimmed_videos_path, f"{video_id}.mp4")
    
    dl_success = download_youtube_video_stable(video_id, full_file)

    trim_success = False
    if dl_success:
        trim_success = trim_video_stable(full_file, trim_file, start_sec, clip_length)


    if not dl_success or not trim_success:
        with log_lock:
            with open(log_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    video_id,
                    "success" if dl_success else "failed",
                    "success" if trim_success else "failed"
                ])
    else:
        print(f"✅ Successfully processed {video_id}")

# @keep.running
def vggsound_batch_down_trim_parallel_stable(video_list, full_videos_path, trimmed_videos_path, log_file, clip_length=10, max_workers=3):
    """Runs the stable pipeline concurrently."""
    os.makedirs(full_videos_path, exist_ok=True)
    os.makedirs(trimmed_videos_path, exist_ok=True)

    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["video_id", "download", "trim"])

    print(f"🚀 Starting stable parallel processing with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_video_p_stable, v, full_videos_path, trimmed_videos_path, log_file, clip_length)
            for v in video_list
        ]
        for future in futures:
            future.result()




if __name__ == "__main__":
    pass

