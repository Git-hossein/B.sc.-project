import subprocess
import numpy as np
import os
import MainCode as mc

# subprocess.run(["ffmpeg", "-version"])


emb = np.load(os.path.join(mc.audio_embeddings_path, "--0PQM4-hqg.npy"))
print(type(emb))      # numpy.ndarray
print(emb.shape)
print(emb[:10]) 
