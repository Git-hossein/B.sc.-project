from path_settings import paths_config
import os
import csv
import yt_dlp
import subprocess
import numpy as np
import soundfile as sf
import wav2clip
from PIL import Image
import torch
import clip
import csv
import random
import pprint
import json
import shutil

# Base dataset path
paths_config.set_to_linux_paths()

# Create separate folders for full and trimmed clips
full_videos_path = paths_config.full_videos_path
trimmed_videos_path = paths_config.trimmed_videos_path
video_frames_path = paths_config.video_frames_path
audios_path = paths_config.audios_path
audio_embeddings_path = paths_config.audio_embeddings_path
video_embeddings_path = paths_config.video_embeddings_path
inferred_example_path = paths_config.inferred_example_path
os.makedirs(full_videos_path, exist_ok=True)
os.makedirs(trimmed_videos_path, exist_ok=True)
os.makedirs(audios_path, exist_ok=True)
os.makedirs(audio_embeddings_path, exist_ok=True)
os.makedirs(video_embeddings_path, exist_ok=True)
os.makedirs(inferred_example_path, exist_ok=True)

# Log files path
download_and_trim_log_file = paths_config.download_and_trim_log_file
extract_audio_log_file = paths_config.extract_audio_log_file
embeddings_audio_log_file = paths_config.embeddings_audio_log_file
video_embeddings_log_file = paths_config.video_embeddings_log_file

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

def vggsound_training_videos_generator(vggsound_path, nmany=1033, start=0):
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
            if row[3].strip() != "train":
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

            

videos_training = vggsound_training_videos_generator("vggsound.csv", 1033, 0) # last called with ("vggsound.csv", 2000 - 779, 779) and stoped at when i had 1026 trimmed files


def download_youtube_video(video_id, output_path):
    """Downloads a single video if it doesn't exist."""
    if not os.path.exists(output_path):
        print(f"Downloading full video {video_id}...")
        ydl_opts = {'format': 'mp4', 'outtmpl': output_path, 'cookiefile': 'cookies.txt'}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
            print(f"Downloaded full video {video_id} successfully.")
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
        print(f"trimmed full video {os.path.basename(input_path)} successfully.")
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
        clip_file = os.path.join(trimmed_videos_path, f"{video_id}.mp4")
        dl_success = False
        trim_success = False

        # Download full video
        dl_success = download_youtube_video(video_id, full_file)

        # Trim video
        if dl_success:
            trim_success = trim_video(full_file, clip_file, start_sec, clip_length)

        # Log failures TODO: write a fucntion to prevent the faulty video being logged multiple times
        if not dl_success or not trim_success:
            with open(log_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([video_id,
                                "success" if dl_success else "failed",
                                "success" if trim_success else "unknown" if not dl_success else "failed"])


#download_and_trim_videos(videos_training, full_videos_path, trimmed_videos_path, download_and_trim_log_file, clip_length=10)

def extract_frames_from_video(input_video, output_dir, fps = 5):
        """
        Extracts frames from a video and puts the resulting frames in the output directory. replaces if frames already exist 
        """
        video_name = os.path.splitext(os.path.basename(input_video))[0]
        os.makedirs(output_dir, exist_ok=True)

        # Output pattern for frames
        output_pattern = os.path.join(output_dir, "frame_%04d.jpg")

        # ffmpeg command
        cmd = [
            paths_config.ffmpeg_path,
            "-y",
            "-i", input_video,
            "-vf", f"fps={fps}",
            output_pattern,
            "-hide_banner",
            "-loglevel", "error"  # suppress ffmpeg spam
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Extracted frames for {video_name} successfully.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to extract frames for {video_name}: {e}")
            return False
        
       


def vggsound_batch_extract_frames_from_vids(trimmed_videos_dir, frames_dir, fps=5):
    """
    Extract frames from all trimmed .mp4 videos in `trimmed_videos_dir` and save them 
    into subfolders in `frames_dir`, one subfolder per video.

    Args:
        trimmed_videos_dir (str): Path to the folder containing trimmed .mp4 videos.
        frames_dir (str): Path to save extracted frames.
        fps (int): Number of frames per second to extract (default: 5).
    """
    os.makedirs(frames_dir, exist_ok=True)
    success_overall = True

    for video in os.listdir(trimmed_videos_dir):
        if not video.endswith(".mp4"):
            print(f"skipping {video}: non mp4 file...")
            continue
        video_path = os.path.join(trimmed_videos_dir, video)
        video_id = os.path.splitext(video)[0]
        output_folder = os.path.join(frames_dir, video_id)
        os.makedirs(output_folder, exist_ok=True)

        # Check if frames already exist
        existing_frames = [f for f in os.listdir(output_folder) if f.startswith("frame_") and f.endswith(".jpg")]
        if existing_frames:
            print(f"⏭ Frames appear to already exist for {video_id}, skipping...")
            continue

        success = extract_frames_from_video(video_path, output_folder, fps = fps)
        if not success: success_overall = False

    return success_overall



#extract_frames_from_videos(trimmed_videos_path, video_frames_path, fps=5)

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
            paths_config.ffmpeg_path,
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

#extract_audio_from_videos(trimmed_videos_path, audios_path, sample_rate=16000)



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

#extract_audio_embeddings(audios_path, audio_embeddings_path)

def extract_video_embedding(input_frame_dir, output_file, model=None, preprocess=None, device="cuda" if torch.cuda.is_available() else "cpu"):

    if model is None or preprocess is None:
        model, preprocess = clip.load("ViT-B/32", device=device)

    if not output_file.endswith(".npy"): 
        print(f"outfile {output_file} should have extension .npy")
        return False    

    frame_files = sorted([f for f in os.listdir(input_frame_dir) if f.endswith(".jpg")])
    if len(frame_files) == 0:
        print(f"⚠️ No frames found for {os.path.basename(output_file)}!")
        return False

    try:
        all_embeddings = []
        for f in frame_files:
            img_path = os.path.join(input_frame_dir, f)
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = model.encode_image(image)
                emb = emb / emb.norm(dim=-1, keepdim=True)  # normalize
                all_embeddings.append(emb.cpu().numpy())

        # Average embeddings
        video_emb = np.mean(np.vstack(all_embeddings), axis=0)

        # Save
        np.save(output_file, video_emb)  #<TODO i wanted it to replace if the output already exists!
        print(f"✅ Saved embedding for video {os.path.basename(input_frame_dir)} at {output_file}")
        return True

    except Exception as e:
        print(f"⚠️ Failed to embed video frame folder {os.path.basename(input_frame_dir)}: {e}")
        return False



def vggsound_extract_video_embeddings_batch(frames_dir, video_embeddings_dir, log_file=video_embeddings_log_file,
                             device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Extracts video embeddings by averaging CLIP embeddings of frames.
    Logs any failures to a CSV.

    Args:
        frames_dir (str): Path containing subfolders for each video with frames (*.jpg).
        video_embeddings_dir (str): Path to save one .npy file per video.
        video_embeddings_log_file (str): Path to CSV log file.
        device (str): 'cuda' or 'cpu'.
    """

    # Load CLIP model
    print(f"Loading CLIP model on {device}...")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # List all videos (subfolders)
    video_ids = [v for v in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, v))]
    print(f"Found {len(video_ids)} frame folders (videos) for embedding extraction.")

    for vid in video_ids: 
        out_file = os.path.join(video_embeddings_dir, f"{vid}.npy")
        
        # Skip if embedding already exists
        if os.path.exists(out_file):
            print(f"⏭ Embedding already exists for {vid}, skipping...")
            continue

        frame_folder = os.path.join(frames_dir, vid)

        success = extract_video_embedding(frame_folder,out_file, model,preprocess)

        if not success:
            with open(log_file, mode='a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([vid, "embedding_failed"])

    print("... Batch embedding complete...")

# extract_video_embeddings(video_frames_path, video_embeddings_path, video_embeddings_log_file)



# =============                     The RAG section                             ====================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1D numpy arrays.
    """
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))

def softmax(scores, T = 1.0):
    scores = np.array(scores) / T
    scores_max = np.max(scores)
    exponentials = np.exp(scores - scores_max)
    return (exponentials / np.sum(exponentials)).tolist()



def find_top_k_similar(query_emb: np.ndarray, embeddings_dir: str, k: int = 5, T: float = 1.0):
    """
    Find the top-k most similar embeddings in a directory to the query embedding.

    Args:
        query_emb (np.ndarray): Query embedding (1D array).
        embeddings_dir (str): Path to folder containing .npy embeddings.
        k (int): Number of top results to return.
        T (int): Temperature param. used by softmax

    Returns:
        List of tuples: [(filename, similarity_score), ...] sorted by similarity descending.
    """
    similarities = []

    for file in os.listdir(embeddings_dir):
        if not file.endswith(".npy"):
            continue

        emb_path = os.path.join(embeddings_dir, file)
        emb = np.load(emb_path)
        # Ensure it's a 1D vector
        if emb.ndim > 1:
            emb = emb.squeeze()
        sim = cosine_similarity(query_emb, emb)
        similarities.append((file, sim))

    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    similarities = similarities[:k]

    # Apply softmax to similarity scores for better interpretability (optional)
    scores = np.array([pair[1] for pair in similarities])
    softmax_scores = softmax(scores, T)
    
    # Return top-k with corresponding softmax score
    return [(pair[0], prob) for pair, prob in zip(similarities, softmax_scores)]


# =============                     The Inference section                             ====================

# Example usage:
#print("example time:")
# query_embedding = np.load(os.path.join(video_embeddings_path, "--PlJNEnf-s.npy")) # bee, wasp, etc. buzzing
# top_similar = find_top_k_similar(query_embedding, audio_embeddings_path, k=5)
# for fname, score in top_similar:
#     print(fname, score)


def infer_similar_audio(query=None, top_k=5, 
                        single_mode=True, random_sample_count=1, temp:float = 1.0):
    """
    Retrieve top-k most similar audio embeddings for given video embeddings. If the query is the full path to a video embedding,
    it will be used directly. If the query is a YouTube ID, it will be assumed that its embedding is stored 
    as "query.npy" in the `video_embeddings_path` folder.

    Args:
        query (str or str path): Either YouTube ID (without .npy) or full path to video embedding.
                                 Only used in single_mode.
        top_k (int): Number of top similar audios to return.
        single_mode (bool): If True, use a single video embedding; 
                            if False, randomly sample multiple videos.
        random_sample_count (int): Number of random videos to sample in random-sample mode.
        T (int): Temperature param. used by softmax for inference

    Returns:
        dict: 
            {
                query_embedding_name: {
                    audio_filename1: similarity_score1,
                    audio_filename2: similarity_score2,
                    ...
                },
                ...
            }
    """
    results = {}
    audio_embeddings_dir = audio_embeddings_path  # fixed path from your base_path
    video_embeddings_dir = video_embeddings_path

    # Helper to load embedding given path or ID
    def load_embedding(query):
        if os.path.exists(query):  # full path
            return np.load(query), os.path.basename(query)
        else:  # assume query is YouTube ID
            emb_path = os.path.join(video_embeddings_dir, f"{query}.npy")
            if not os.path.exists(emb_path):
                raise FileNotFoundError(f"Video embedding not found for ID {query}")
            return np.load(emb_path), f"{query}.npy"

    # --- SINGLE MODE ---
    if single_mode:
        if query is None:
            raise ValueError("In single_mode, `query` must be provided (ID or path).")
        
        video_emb, key_name = load_embedding(query)
        top_similar = find_top_k_similar(video_emb, audio_embeddings_dir, k=top_k, T = temp)
        # convert to dict {filename: similarity}
        results[key_name] = {fname: score for fname, score in top_similar}

    # --- RANDOM SAMPLE MODE ---
    else:
        # List all video embeddings
        all_video_files = [f for f in os.listdir(video_embeddings_dir) if f.endswith(".npy")]
        if len(all_video_files) == 0:
            raise FileNotFoundError("No video embeddings found in the directory.")
        if random_sample_count > len(all_video_files):
            random_sample_count = len(all_video_files)
        
        sampled_videos = random.sample(all_video_files, random_sample_count)
        
        for vid_file in sampled_videos:
            vid_path = os.path.join(video_embeddings_dir, vid_file)
            video_emb = np.load(vid_path)
            top_similar = find_top_k_similar(video_emb, audio_embeddings_dir, k=top_k, T = temp)
            results[vid_file] = {fname: score for fname, score in top_similar}

    return results

example1 = infer_similar_audio(query="--PlJNEnf-s", top_k=5, single_mode=True, random_sample_count=3)



def create_inference_example(inference_dict):
    """
    Given a dictionary returned by `infer_similar_audio`, create a folder structure
    with trimmed videos and matched audio files for easy viewing.

    Folder structure:
    inferred_example_path/
        query_video_name/
            query_video_name.mp4
            matched_audio1.wav
            matched_audio2.wav
            ...

    Args:
        inference_dict (dict): output of `infer_similar_audio`
    """
    for video_key, audio_matches in inference_dict.items():
        # Remove .npy from video key to get folder/video name
        video_name = os.path.splitext(video_key)[0]
        video_folder = os.path.join(inferred_example_path, video_name)

        # Create or replace folder
        if os.path.exists(video_folder):
            shutil.rmtree(video_folder)
        os.makedirs(video_folder, exist_ok=True)

        # Copy trimmed video
        trimmed_video_file = os.path.join(trimmed_videos_path, f"{video_name}.mp4")
        if os.path.exists(trimmed_video_file):
            shutil.copy(trimmed_video_file, os.path.join(video_folder, f"{video_name}.mp4"))
        else:
            print(f"⚠️ Trimmed video not found for {video_name}, skipping video copy.")

        # Copy matched audio files
        for audio_file in audio_matches.keys():
            # Remove .npy if present to get actual wav filename
            audio_name = os.path.splitext(audio_file)[0] + ".wav" if audio_file.endswith(".npy") else audio_file
            audio_source = os.path.join(audios_path, audio_name)
            if os.path.exists(audio_source):
                shutil.copy(audio_source, os.path.join(video_folder, audio_name))
            else:
                print(f"⚠️ Audio file {audio_name} not found for {video_name}, skipping.")

    print(f"☑️ Inference examples created in {inferred_example_path}")



def evaluate_inference(data_dict):
    total_video = len(data_dict)
    num_og_audio_found = 0
    reciprocal_ranks = []

    for vid, audios in data_dict.items():
        rank = 0

        # audios is already sorted → just iterate
        for i, audio_name in enumerate(audios.keys(), start=1):
            if audio_name == vid:
                rank = i
                break

        if rank > 0:
            num_og_audio_found += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

    recall_at_k = num_og_audio_found / total_video
    mrr = sum(reciprocal_ranks) / total_video

    return {
        "total number of query videos": total_video,
        "recall_at_k": recall_at_k,
        "mrr": mrr
    }



if __name__ == "__main__":
    vids = ["--XInAaMS6k", "-0gYWIOfqdM", "-3M-k4nIYIM", "-4ItJ9yTz_c", "-4o0jRbgHr4", "-4rdRn-FRXo", "-6lkiUAf_cQ", "-6VFTlZsft4"]
   
    # for vid in vids:
    #     print(",\n")
    #     pprint.pp(infer_similar_audio(query= vid, top_k=5, single_mode=True, random_sample_count=3, temp= 0.01), sort_dicts=False)
    #     print(",\n")

    # random video
    # extract_frames_from_video("/home/hossein/Desktop/B.sc.-project/inferred_examples/-0gYWIOfqdM/-0gYWIOfqdM.mp4","/home/hossein/Desktop/randomclip", 5)
    # extract_video_embedding("/home/hossein/Desktop/randomclip", "/home/hossein/Desktop/rando.npy")
    pprint.pp(infer_similar_audio("/home/hossein/Desktop/rando.npy", 5, True,1, 0.01), sort_dicts=False)
    pprint.pp(infer_similar_audio(query= "-0gYWIOfqdM", top_k=5, single_mode=True, random_sample_count=3, temp= 0.01), sort_dicts=False)