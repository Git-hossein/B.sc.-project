from path_settings import paths_config
from pathlib import Path
import os
import csv
import subprocess
import numpy as np
from PIL import Image
import torch
import clip
import csv



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
            "-vf", f"fps={fps}:round=up",
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

    print("======== begin batch frame extraction ========")

    videos = [f for f in os.listdir(trimmed_videos_dir) if f.endswith(".mp4")]
    print(f"Found {len(videos)} trimmed videos")

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


def extract_video_embedding(
        input_frame_dir, 
        output_file, 
        model=None, 
        preprocess =None, 
        device="cuda" if torch.cuda.is_available() else "cpu"):

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
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device) # type: ignore

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


def extract_video_embedding_fast(
        input_frame_dir, 
        output_file, 
        model= None, 
        preprocess= None, 
        device="cuda" if torch.cuda.is_available() else "cpu"):
    
    
    if model is None or preprocess is None:
        model, preprocess = clip.load("ViT-B/32", device=device)

    frame_files = sorted([f for f in os.listdir(input_frame_dir) if f.endswith(".jpg")])
    if not frame_files: return False

    try:
        # Load all frames into a list first (CPU)
        frames = [preprocess(Image.open(os.path.join(input_frame_dir, f))) for f in frame_files]
        
        # Stack into one batch and move to GPU at once
        # shape: [num_frames, 3, 224, 224]
        batch = torch.stack(frames).to(device)

        with torch.no_grad():
            # Process the whole video in one 'gulp'
            # Note: If you get an 'Out of Memory' error, reduce FPS or process in chunks of 25
            features = model.encode_image(batch)
            features /= features.norm(dim=-1, keepdim=True)
            
            # Average the embeddings on the GPU
            video_emb = features.mean(dim=0)

        # Save to disk (np.save overwrites by default)
        np.save(output_file, video_emb.cpu().numpy())
        print(f"✅ Extracted: {os.path.basename(output_file)}")
        return True

    except Exception as e:
        print(f"❌ Error on {input_frame_dir}: {e}")
        return False


def vggsound_extract_video_embeddings_batch(
        frames_dir, 
        video_embeddings_dir, 
        log_file= paths_config.video_embeddings_log_file, 
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

        success = extract_video_embedding_fast(frame_folder,out_file, model,preprocess, device)

        if not success:
            with open(log_file, mode='a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([vid, "embedding_failed"])

    print("... Batch embedding complete...")