import os
import torch
import torchaudio
import torchaudio.transforms as T
import csv 
from audiocraft.models import AudioGen
#  Tell the 'transformers' library to stop complaining and just work
import transformers.utils.import_utils as import_utils
import_utils._torch_available = True 
import warnings
import json
from typing import Optional, Any, Literal
import shutil
import argparse

"""
for audiogeneration
"""

# --- PATH & CACHE SETUP ---
# Ensure the model doesn't re-download every job
torch_cache = os.path.expanduser("~/.cache/torch_models")
os.makedirs(torch_cache, exist_ok=True)
os.environ['TORCH_HOME'] = torch_cache



# --- MODULE 3: THE MASTER EXECUTION ---
def run_parallel_generation(
        vid_list: list[str], 
        descriptions: list[str], 
        cfg_coef = 3.0, 
        shared_model: Optional[AudioGen] = None,
        batch_size = 4
    )-> list[str]:
    """
    Processes a batch of video audio results by mixing them and re-synthesizing 
    using AudioGen.

    Args:
        data_dict (Dict[str, Dict[str, float]]): A nested dictionary where keys are 
            video filenames and values are dictionaries of inferred audio 
            segments with their corresponding softmax scores.
        descriptions (List[str]): A list of text descriptions/prompts for each 
            video, ordered to match data_dict.keys().
        shift (int, optional): Number of samples to shift/nudge the mixed audio 
            for synchronization. Defaults to 0.
        shared_model (Optional[AudioGen], optional): A pre-loaded AudioGen model 
            instance. If None, the model is loaded locally. Defaults to None.

    Returns:
        List[str]: A list of file paths to the successfully generated audio files.

    Raises:
        TypeError: If shared_model is provided but is not an AudioGen instance.
        ValueError: If the number of descriptions does not match the number of videos.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if shared_model is not None:
        if not isinstance(shared_model, AudioGen):
            raise TypeError(
                f"❌ Invalid shared_model type: {type(shared_model)}. "
                f"This function specifically requires audiogen-medium model to "
                f"handle continuation prompts."
            )
        model = shared_model
        print("Using verified shared AudioGen model...")
    else:
        warnings.warn("⚠️ No shared_model provided. Loading AudioGen-Medium locally.")
        model = AudioGen.get_pretrained('facebook/audiogen-medium', device=device)

    assert len(vid_list) == len(descriptions), "the number of descriptions and videis must match"

    generated_files = []

    output_dir = os.getenv('OUTPUT_DIR', '.')
    os.makedirs(output_dir, exist_ok=True)

    for i in range(0, len(vid_list), batch_size):


        batch_descriptions = descriptions[i: i+batch_size]
        batch_vids = vid_list[i: i+batch_size]
        batch_video_ids = []

        for video in batch_vids:
            # Extract the video id and its audio results
            video_id = os.path.splitext(video)[0]
            batch_video_ids.append(video_id)


        print(f"🤖 Processing on: {device.upper()}")
        
        print("✨ generation audio based on text description...")

        model.set_generation_params(duration=10.0, cfg_coef= cfg_coef)
        with torch.no_grad():
            output_batch = model.generate(descriptions= batch_descriptions,
                                          progress= True)
        
        # 3. Save and Preview
        for j, output_audio in enumerate(output_batch):
                    video_id = batch_video_ids[j]
                    final_path = os.path.join(output_dir, f"{video_id}_GEN.wav")
                    sr=16000
                    
                    # output_audio is already the specific tensor for this video
                    torchaudio.save(final_path, output_audio.cpu(), sr)
                    
                    print(f"✅ Success! Master file: {final_path}")
                    generated_files.append(final_path)


    return generated_files





if __name__ == "__main__":

    pass
