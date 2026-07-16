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

def download_youtube_video_range(video_id, output_path, start_sec, duration=10):
    """MODIFIED: Downloads only the specific range to save bandwidth."""
    if not os.path.exists(output_path):
        print(f"⬇️ Downloading range for {video_id}...")
        ydl_opts: Any = {
            'format': 'mp4',
            'outtmpl': output_path,
            'cookiesfrombrowser': ('firefox',),
            'retries': 10,
            'fragment_retries': 10,
            'continuedl': True,
            'quiet': True,
            'noprogress': True,
            'remote_components': ['ejs:github'],
            # Range download logic
            'download_ranges': lambda info_dict, ydl: [{
                'start_time': start_sec,
                'end_time': start_sec + duration,
            }],
            'force_keyframes_at_cuts': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
            return True
        except Exception as e:
            print(f"Failed to download {video_id}: {e}")
            return False
    else:
        return True

def trim_video(input_path, output_path, start_sec, duration=10):
    """UNCHANGED: Exact same parameters as your old code for parity."""
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
    except subprocess.CalledProcessError:
        return False

def process_single_video_p(video_info, full_videos_path, trimmed_videos_path, log_file, clip_length):
    """New helper to manage the download->trim pipeline for one thread."""
    video_id, start_sec, label = video_info
    full_file = os.path.join(full_videos_path, f"{video_id}_full.mp4")
    trim_file = os.path.join(trimmed_videos_path, f"{video_id}.mp4")
    
    # 1. Download the range
    dl_success = download_youtube_video_range(video_id, full_file, start_sec, clip_length)

    # 2. Trim/Re-encode
    trim_success = False
    if dl_success:
        # NOTE: Because yt-dlp already cut the video, 'full_file' now starts at 0.
        # We use 0 here so FFmpeg re-encodes the clip exactly like your old ones.
        trim_success = trim_video(full_file, trim_file, 0, clip_length)

    # 3. Thread-safe Logging
    if not dl_success or not trim_success:
        with log_lock:
            with open(log_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([video_id,
                                "success" if dl_success else "failed",
                                "success" if trim_success else "failed"])



@keep.running
def vggsound_batch_down_trim_parralel(video_list, full_videos_path, trimmed_videos_path, log_file, clip_length=10, max_workers=4):
    """MODIFIED: Uses ThreadPoolExecutor for concurrent processing."""
    os.makedirs(full_videos_path, exist_ok=True)
    os.makedirs(trimmed_videos_path, exist_ok=True)

    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["video_id", "download", "trim"])

    print(f"🚀 Starting parallel processing with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_video_p, v, full_videos_path, trimmed_videos_path, log_file, clip_length)
            for v in video_list
        ]
        # This loop waits for all threads to finish
        for future in futures:
            future.result()

# --- HOW TO CALL IT ---


if __name__ == "__main__":
    from audio.audio_processing import vggsound_extract_audio_embeddings_batch, vggsound_extract_audio_from_videos_batch
    from video.video_processing import vggsound_extract_video_embeddings_batch, vggsound_batch_extract_frames_from_vids
    from .vggsound_processing import vggsound_batch_down_trim
    os.makedirs(paths_config.test_audios_path, exist_ok= True)
    os.makedirs(paths_config.test_audio_embeddings_path, exist_ok= True)
    os.makedirs(paths_config.test_video_embeddings_path, exist_ok= True)
    os.makedirs(paths_config.test_video_frames_path, exist_ok= True)

    # Log files path
    download_and_trim_log_file = paths_config.test_download_and_trim_log_file
    extract_audio_log_file = paths_config.test_extract_audio_log_file
    embeddings_audio_log_file = paths_config.test_embeddings_audio_log_file
    video_embeddings_log_file = paths_config.test_video_embeddings_log_file

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

    if not os.path.exists(video_embeddings_log_file):
        with open(video_embeddings_log_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["video_id", "embed_status"])
    # 1. Setup your paths
    CSV_PATH = paths_config.vggsound_path
    FULL_PATH = paths_config.test_full_videos_path
    TRIMMED_PATH = paths_config.test_trimmed_videos_path
    LOG_PATH = paths_config.test_download_and_trim_log_file

    for directory in [TRIMMED_PATH, FULL_PATH]:
        for file in os.listdir(directory):
            # Construct the full absolute path to the file
            file_path = os.path.join(directory, file)
            
            # Check if it's a file (skips folders), is 0 bytes, or doesn't end with .mp4
            if os.path.isfile(file_path):
                if os.path.getsize(file_path) == 0 or not file.lower().endswith(".mp4"):
                    print(f"Deleting invalid file: {file_path}")
                    os.remove(file_path)

    # 2. Initialize the generator and convert to a list for the batch processor
    videos_to_process = list(vggsound_training_videos_generator(CSV_PATH, nmany=4000, start=5000, type="train"))

    # 3. Run the batch
    vggsound_batch_down_trim_parralel(
        video_list=videos_to_process,
        full_videos_path="/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/training/extra_training/full_videos",
        trimmed_videos_path="/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/training/extra_training/trimmed_videos",
        log_file="/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/training/extra_training/extra_training_Logs_download_trim.csv",
        clip_length=10,
        max_workers=4  # Adjust this based on your network speed, tried and true is 3 workers!!
    )
    # downed= [f"{vid.rsplit("_full")[0]}.mp4" for vid in os.listdir(paths_config.test_full_videos_path)]

    # trimmed = [vid for vid in os.listdir(paths_config.test_trimmed_videos_path)]

    # gen_vids = []
    # for vid in downed:
    #     if vid not in trimmed:
    #         vid_id = os.path.splitext(vid)[0]
    #         gen_vids.append((vid_id, 0, "test")) 

    # # vggsound_batch_down_trim(gen_vids, paths_config.test_full_videos_path, paths_config.test_trimmed_videos_path, paths_config.test_download_and_trim_log_file)
    vggsound_batch_extract_frames_from_vids(TRIMMED_PATH, paths_config.test_video_frames_path)
    vggsound_extract_video_embeddings_batch(paths_config.test_video_frames_path, paths_config.test_video_embeddings_path, paths_config.test_video_embeddings_log_file)
    vggsound_extract_audio_from_videos_batch(TRIMMED_PATH, paths_config.test_audios_path, log_file=paths_config.test_extract_audio_log_file)
    vggsound_extract_audio_embeddings_batch(paths_config.test_audios_path, paths_config.test_audio_embeddings_path, log_file= paths_config.test_embeddings_audio_log_file)

    # # trim_video("/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/test/full_videos/L9nIN8KrWfQ_full.mp4", 
    # #            "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/test/trimmed_videos/L9nIN8KrWfQ_trimmed.mp4",
    # #            0,
    # #            10)