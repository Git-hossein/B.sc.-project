import os
import csv
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Iterator, Any
import yt_dlp
from wakepy import keep
from path_settings import paths_config

# A lock to prevent multiple threads from writing to the log file at the same time
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
            # Crucial for parallel: hide progress bars so the terminal doesn't glitch out
            'quiet': True,
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
    
    # 1. Download the FULL video
    dl_success = download_youtube_video_stable(video_id, full_file)

    # 2. Trim from the actual start_sec (not 0)
    trim_success = False
    if dl_success:
        trim_success = trim_video_stable(full_file, trim_file, start_sec, clip_length)



    # 4. Thread-safe Logging
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

# @keep.running (if you are using a custom decorator, keep it here)
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

# --- HOW TO CALL IT ---


if __name__ == "__main__":
    from audio.audio_processing import vggsound_extract_audio_embeddings_batch, vggsound_extract_audio_from_videos_batch
    from video.video_processing import vggsound_extract_video_embeddings_batch, vggsound_batch_extract_frames_from_vids
    from .vggsound_processing import vggsound_batch_down_trim
    print("hallllllllllllllo")
    # Log files path
    
    extract_audio_log_file = "/home/hossein/Desktop/B.sc.-project/audio/Logs_extract_audio.csv"
    embeddings_audio_log_file = "/home/hossein/Desktop/B.sc.-project/audio/Logs_audio_embeddings.csv"
    vid_emb_log_file = "/home/hossein/Desktop/B.sc.-project/video/Logs_video_embeddings.csv"

    frames_path = r"/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/training/frames"
    full_videos_path = r"/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/training/full_videos"
    trimmed_videos_path = r"/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/training/trimmed_videos"
    vid_emb_path = "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/training/video_embeddings"
    aud_em_path = "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/training/audio_embeddings"
    aud_path = "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/training/audios"
    vggsound_csv = "/home/hossein/Desktop/B.sc.-project/vggsound/vggsound.csv"


    vggsound_batch_extract_frames_from_vids(trimmed_videos_path, frames_path)
    vggsound_extract_video_embeddings_batch(frames_path, vid_emb_path, vid_emb_log_file )
    vggsound_extract_audio_from_videos_batch(trimmed_videos_path, aud_path, log_file=extract_audio_log_file)
    vggsound_extract_audio_embeddings_batch(aud_path, aud_em_path, log_file= embeddings_audio_log_file)

