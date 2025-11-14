import subprocess
import numpy as np
import os
import MainCode as mc

# subprocess.run(["ffmpeg", "-version"])


# emb = np.load(os.path.join(mc.audio_embeddings_path, "--0PQM4-hqg.npy"))
# print(type(emb))      # numpy.ndarray
# print(emb.shape)
# print(emb[:10]) 


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


video_id1 = "--0PQM4-hqg"
video_id2 = "--56QUhyDQM"
embedding_file = f"D:/Bsc.Thesis_Datasets/vggsound/video_embeddings/{video_id2}.npy"

# Load embedding
emb = np.load(embedding_file)
print("Shape:", emb.shape)      # Should be (512,) or (1, 512)
print("Embedding vector:", emb[:10]) # Look at the first 10 values
print("Min/Max:", emb.min(), emb.max())  # Just to sanity check values