import subprocess
import numpy as np
import os
import csv
import MainCode as mc

# subprocess.run(["ffmpeg", "-version"])

################################## check tools:

# # Check PIL (Pillow)
# try:
#     from PIL import Image
#     print("PIL is installed")
# except ImportError:
#     print("PIL is NOT installed")

# # Check CLIP
# try:
#     import clip
#     print("CLIP is installed")
# except ImportError:
#     print("CLIP is NOT installed")

# # Check PyTorch
# try:
#     import torch
#     print("PyTorch is installed, version:", torch.__version__)
# except ImportError:
#     print("PyTorch is NOT installed")

##################################

#checking audio embeddings:
# emb = np.load(os.path.join(mc.audio_embeddings_path, "--0PQM4-hqg.npy"))
# print(type(emb))      # numpy.ndarray
# print(emb.shape)
# print(emb[:10]) 

##################################

# checking video embeddings:
# video_id1 = "--0PQM4-hqg"
# video_id2 = "--56QUhyDQM"
# embedding_file = f"D:/Bsc.Thesis_Datasets/vggsound/video_embeddings/{video_id2}.npy"

# # Load embedding
# emb = np.load(embedding_file)
# print("Shape:", emb.shape)      # Should be (512,) or (1, 512)
# print("Embedding vector:", emb[:10]) # Look at the first 10 values
# print("Min/Max:", emb.min(), emb.max())  # Just to sanity check values




# =================== intergrity of dataset files and checking the downloaded and trimmed videos health ===================
# Video_tester = mc.training_videos_generator("vggsound.csv", nmany=1026, start=0)

# Video_tester = list(Video_tester)

# print("Total videos in tester:", len(Video_tester))



def verify_and_cleanup_videos(full_videos_path, trimmed_videos_path):
    # --- Step 1: Clean up wrong files ---
    print("Checking file extensions...")
    for folder, suffix in [(full_videos_path, "_full.mp4"), (trimmed_videos_path, ".mp4")]:
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            if not fname.endswith(suffix):
                print(f"Deleting invalid file: {fpath}")
                os.remove(fpath)

    # --- Step 2: Check that every full video has a trimmed clip ---
    full_files = [f for f in os.listdir(full_videos_path) if f.endswith("_full.mp4")]
    trimmed_files = [f for f in os.listdir(trimmed_videos_path) if f.endswith(".mp4")]

    trimmed_set = set(trimmed_files)
    missing_trimmed = 0
    for full_file in full_files:
        video_id = full_file.replace("_full.mp4", "")
        trimmed_name = f"{video_id}.mp4"
        if trimmed_name not in trimmed_set:
            print(f"Video with ID {video_id} has no trimmed vid :rex X")
            missing_trimmed += 1

    print(f"Total full videos: {len(full_files)}")
    print(f"Total trimmed videos: {len(trimmed_files)}")
    print(f"Missing trimmed videos: {missing_trimmed}")

    # --- Step 3: Fast integrity check on trimmed clips using ffprobe ---
    print("Checking trimmed video integrity (fast)...")
    corrupted_count = 0
    for fname in trimmed_files:
        fpath = os.path.join(trimmed_videos_path, fname)
        try:
            # ffprobe checks metadata only, very fast
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", fpath],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            duration = float(result.stdout.strip())
            if duration <= 0:
                print(f"Corrupt or empty video detected: {fname}, deleting...")
                os.remove(fpath)
                corrupted_count += 1
        except Exception as e:
            print(f"Error checking {fname}: {e}, deleting...")
            os.remove(fpath)
            corrupted_count += 1

    print(f"Corrupt or empty trimmed videos removed: {corrupted_count}")
    print("Verification complete!")

# verify_and_cleanup_videos(mc.full_videos_path, mc.trimmed_videos_path)