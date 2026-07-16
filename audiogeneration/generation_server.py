import os
import json
import subprocess
import time
import numpy as np
from inference import load_all_normed_embeddings
from path_settings import paths_config
import shutil
import warnings
import datetime
import glob
import tarfile

SBATCH_TCML_PATH = "/home/sherkat/B.sc.-Audiocraft-module/generate_audio.sbatch"
SESSION_ID_FORMAT = "%Y%m%d_%H%M%S_%f"
CONFIG_JSON = "config.json"
CHUNK_JSON = "chunk_{idx}.json"


def chunk_dict(data, n_chunks):
    """Splits a dictionary into n roughly equal parts."""
    keys = list(data.keys())
    # Calculate size of each chunk
    size = int(np.ceil(len(keys) / n_chunks))
    
    chunks = []
    for i in range(0, len(keys), size):
        chunk_keys = keys[i:i + size]
        chunks.append({k: data[k] for k in chunk_keys})
    return chunks


def prepare_parallel_batch(inferred_dict, num_GPU, options):
    # 1. Create a unique session folder locally
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    local_session_path = os.path.join(paths_config.TCML_server_input, session_id)
    os.makedirs(local_session_path, exist_ok=True)

    # 2. Save Global Config
    config_path = os.path.join(local_session_path, "config.json")
    with open(config_path, 'w') as f:
        json.dump(options, f, indent=4)

    # 3. Split and Save Chunks
    chunks = chunk_dict(inferred_dict, num_GPU)
    for idx, chunk in enumerate(chunks):
        chunk_path = os.path.join(local_session_path, f"chunk_{idx}.json")
        with open(chunk_path, 'w') as f:
            json.dump(chunk, f, indent=4)

    # 4. Build Media Upload Set (same as before, just looking at the whole dict)
    upload_set = set()
    for matches in inferred_dict.values():
        for audio_emb in matches.keys():
            target_audio_id = os.path.splitext(audio_emb)[0]
            target_audio_path = os.path.join(paths_config.audios_path, f"{target_audio_id}.wav")
            if os.path.exists(target_audio_path):
                upload_set.add(target_audio_path)
            else:
                raise FileNotFoundError(f"Missing: {target_audio_path}")
            
    return local_session_path, upload_set, session_id, len(chunks)



def upload_parallel_to_tcml(local_session_dir, upload_set, session_id):
    remote_host = "sherkat@login3.tcml.uni-tuebingen.de"
    # We now create a unique path for this specific run
    remote_base = "/home/sherkat/B.sc.-Audiocraft-module/Hossein/input/"
    remote_session_folder = f"{remote_base}{session_id}/"
    
    print(f"🚀 Creating remote session: {session_id}")
    subprocess.run(["ssh", remote_host, f"mkdir -p {remote_session_folder}"], check=True)

    # --- Upload Chunks and Config ---
    print("... Uploading JSON chunks and config")
    # Using '/*' to send the contents of the local folder to the remote session folder
    subprocess.run(["rsync", "-avz", f"{local_session_dir}/", f"{remote_host}:{remote_session_folder}"], check=True)

    # --- Upload Media (to the shared flat input folder to save space/time) ---
    # We keep media in the base folder so we don't re-upload the same .wav for different sessions
    media_manifest = "to_upload_manifest.txt"
    with open(media_manifest, "w") as f:
        for path in upload_set:
            f.write(f"{os.path.basename(path)}\n")

    print(f"... Syncing {len(upload_set)} media files to shared input folder")

    try:
        subprocess.run([
        "rsync", "-avz", "--ignore-existing", "--no-R",
        f"--files-from={media_manifest}", 
        os.path.join(paths_config.audios_path, ""), 
        f"{remote_host}:{remote_base}"
        ], capture_output= True, check=True)

        return True
       
    except subprocess.CalledProcessError as e:
        print(f"❌ rsync failed with return code {e.returncode}")
        if e.stderr:
            print(f"Error output: {e.stderr.decode()}")
        return False

    except Exception as e:
        print(f"💥 Unexpected error during media upload: {e}")
        return False
    
    finally:
        if os.path.exists(media_manifest):
            os.remove(media_manifest)



def trigger_parallel_sbatch(session_id, num_chunks):
    remote_host = "sherkat@login3.tcml.uni-tuebingen.de"
    array_range = f"0-{num_chunks - 1}"

    sbatch_command = (
        f"sbatch --parsable "
        f"--export=ALL,SESSION_ID={session_id} "
        f"--array={array_range} "
        f"{SBATCH_TCML_PATH}"
    )

    print(f"... submitting Job Array: {array_range} for Session: {session_id} ...")
    cmd = ["ssh", remote_host, sbatch_command]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Extract the ID using Regex
        job_id = result.stdout.strip()
        print(f"✅ Array Job submitted! ID: {job_id}")
        return job_id
            
    print(f"❌ Slurm submission failed: {result.stderr}")
    return None


def wait_for_job_completion(session_id, num_chunks, timeout_minutes):
    remote_host = "sherkat@login3.tcml.uni-tuebingen.de"
    sentinel_dir = f"/home/sherkat/audio_outputs/{session_id}"
    
    print(f"⏳ Waiting for {num_chunks} tasks to finish...")
    print("(You can go grab a coffee, this will take a while.)")

    start = time.time()
    while True:
        # Check if the sentinel file exists on the server
        count_cmd = f"ls {sentinel_dir}/DONE_* 2>/dev/null | wc -l"
        result = subprocess.run(["ssh", remote_host, count_cmd], capture_output=True, text=True)
        
        finished_count = int(result.stdout.strip()) if result.stdout.strip() else 0
        print(f"\r progress: [{finished_count}/{num_chunks}] tasks complete...", end="", flush=True)
        
        if finished_count >= num_chunks:
            print(f"\n✨ All {num_chunks} tasks finished successfully!")
            return True
        
        if time.time() - start > timeout_minutes * 60:
            print("⏰ Timeout reached - some tasks probably failed!")
            return False
            
        time.sleep(60)


def download_parallel_results(session_id, total_videos, local_down_dest = paths_config.TCML_server_output_generated):
    """
    Downloads all session results, flattens them into local_dest (overwriting existing),
    and verifies integrity locally.
    """
    remote_host = "sherkat@login3.tcml.uni-tuebingen.de"
    remote_session_dir = f"~/audio_outputs/{session_id}/"
    remote_source = f"{remote_host}:{remote_session_dir}"
    
    os.makedirs(local_down_dest, exist_ok=True)

    # --- 1. RSYNC DOWNLOAD ---
    # We download the session folder structure directly into your local destination
    print(f"📥 Downloading session {session_id}...")
    # -a (archive), -v (verbose), -z (compress)
    rsync_cmd = ["rsync", "-avz", remote_source, local_down_dest]
    subprocess.run(rsync_cmd, check=True)

    # --- 2. LOCAL FLATTENING & CLEANUP ---
    print(f"🔄 Flattening files into {local_down_dest}...")
    
    # We track how many audio files we find during the move
    audio_count = 0
    
    # Walk through the subdirectories created by rsync (task_0, task_1, etc.)
    for root, dirs, files in os.walk(local_down_dest, topdown=False):
        for file in files:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(local_down_dest, file)
            
            # 1. Skip moving if the file is already in the destination root
            if os.path.dirname(src_path) == os.path.abspath(local_down_dest):
                if file.endswith(".wav"): audio_count += 1
                continue
            
            # 2. Move and Replace
            # os.replace will overwrite dst_path if it exists
            os.replace(src_path, dst_path)
            
            if file.endswith(".wav"):
                audio_count += 1

        # 3. Cleanup empty subdirectories
        for d in dirs:
            dir_path = os.path.join(root, d)
            if not os.listdir(dir_path): # Only remove if empty
                os.rmdir(dir_path)

    # --- 3. LOCAL INTEGRITY VERIFICATION ---
    # We expect 2 audio files per video query (1 RAW_MIX + 1 GEN)
    expected_audio = total_videos * 2
    
    print(f"🧐 Verifying integrity...")
    if audio_count == expected_audio:
        print(f"✅ Success! All {audio_count} audio files present and accounted for.")
    else:
        warnings.warn(
            f"⚠️ Integrity Issue: Expected {expected_audio} audio files, "
            f"but found {audio_count}. Check the Slurm logs for failed tasks.",
            UserWarning
        )

    # Final cleanup of any leftover DONE_ sentinels in the root if you want it clean
    for sentinel in [f for f in os.listdir(local_down_dest) if f.startswith("DONE_")]:
        os.remove(os.path.join(local_down_dest, sentinel))

    return local_down_dest





def run_parallel_tcml_pipeline_continuation(inferred_dict, local_down_dst, timeout_minutes, num_GPU, options = None, auto_download = True):
    if options is None:
        options = {
            "cfg_coef": 3.0,
            "with_text_descr": True,
            "prompt_duration": 2,
            "num_audio_mix": 5,
            "weight_by": "softmax_score"
        }

    # 1. Chunk and Upload
    # (Using the prepare_parallel_batch and upload functions we discussed)
    local_path, file_set, session_id, num_chunks = prepare_parallel_batch(inferred_dict, num_GPU, options)
    
    success = upload_parallel_to_tcml(local_path, file_set, session_id)
    
    if success:
        # 2. Trigger Dynamic Array
        job_id = trigger_parallel_sbatch(session_id, num_chunks)
        
        # 3. Wait for all sentinels
        if auto_download and wait_for_job_completion(session_id, num_chunks, timeout_minutes):
            
            download_parallel_results_zipped(
                session_id=session_id, 
                local_down_dest=local_down_dst, 
                total_videos=len(inferred_dict))
            
    return session_id



def download_parallel_results_zipped(session_id, total_videos, local_down_dest=paths_config.TCML_server_output_generated):
    """
    Downloads session results by first archiving on the server, then downloading,
    flattening locally, and verifying integrity. Identical behavior to the legacy rsync version.
    """
    remote_host = "sherkat@login3.tcml.uni-tuebingen.de"
    
    # Paths on Server
    remote_parent_dir = "~/audio_outputs"
    remote_archive_name = f"{session_id}.tar.gz"
    remote_archive_path = f"{remote_parent_dir}/{remote_archive_name}"

    # Paths Locally
    local_archive_path = os.path.join(local_down_dest, remote_archive_name)
    os.makedirs(local_down_dest, exist_ok=True)

    # --- 1. REMOTE COMPRESSION ---
    print(f"📦 Compressing session {session_id} on TCML...")
    # -C changes to the parent dir so the archive contains the 'session_id' folder relatively
    remote_tar_cmd = [
        "ssh", remote_host,
        f"tar -czf {remote_archive_path} -C {remote_parent_dir} {session_id}"
    ]
    subprocess.run(remote_tar_cmd, check=True)

    # --- 2. DOWNLOAD THE SINGLE ARCHIVE ---
    print(f"📥 Downloading compressed archive...")
    # Using rsync to download the single file (allows resuming if interrupted)
    rsync_cmd = [
        "rsync", "-avz",
        f"{remote_host}:{remote_archive_path}",
        local_down_dest
    ]
    subprocess.run(rsync_cmd, check=True)

    # --- 3. LOCAL EXTRACTION ---
    print(f"🔓 Extracting locally...")
    with tarfile.open(local_archive_path, "r:gz") as tar:
        tar.extractall(path=local_down_dest)

    # --- 4. LOCAL FLATTENING & CLEANUP ---
    print(f"🔄 Flattening files into {local_down_dest}...")
    audio_count = 0
    
    # The extraction creates local_down_dest/session_id/...
    extracted_folder = os.path.join(local_down_dest, session_id)

    # Walk through the extracted session folder
    for root, dirs, files in os.walk(extracted_folder, topdown=False):
        for file in files:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(local_down_dest, file)
            
            # Move and Replace (matches original behavior)
            os.replace(src_path, dst_path)
            
            if file.endswith(".wav"):
                audio_count += 1

    # Cleanup the extracted folder shell and the local archive
    if os.path.exists(extracted_folder):
        shutil.rmtree(extracted_folder)
    if os.path.exists(local_archive_path):
        os.remove(local_archive_path)

    # --- 5. LOCAL INTEGRITY VERIFICATION ---
    expected_audio = total_videos * 2
    print(f"🧐 Verifying integrity...")
    if audio_count == expected_audio:
        print(f"✅ Success! All {audio_count} audio files present and accounted for.")
    else:
        warnings.warn(
            f"⚠️ Integrity Issue: Expected {expected_audio} audio files, "
            f"but found {audio_count}. Check the Slurm logs for failed tasks.",
            UserWarning
        )

    # --- 6. FINAL CLEANUP ---
    # Remove DONE_ sentinels (Identical to legacy cleanup)
    for sentinel in [f for f in os.listdir(local_down_dest) if f.startswith("DONE_")]:
        os.remove(os.path.join(local_down_dest, sentinel))

    # Remove the archive from the server to save space
    subprocess.run(["ssh", remote_host, f"rm -f {remote_archive_path}"], check=False)

    return local_down_dest

def evaluate_generation(
    gen_embs_dict: dict[str, np.ndarray], 
    og_embs_dict: dict[str, np.ndarray] ):
    """
    Aligns and compares generated (or raw mix) embeddings against original embeddings.
    """
    aligned_gen = []
    aligned_og = []
    matched_ids = []

    # 1. Alignment Phase
    for gen_key, gen_vec in gen_embs_dict.items():
        # Strip suffixes to find the base YouTube ID
        # Handles both "{id}_GEN" and "{id}_RAW_MIX"
        clean_id = gen_key.replace("_GEN", "").replace("_RAW_MIX", "")
        
        if clean_id in og_embs_dict:
            aligned_gen.append(gen_vec)
            aligned_og.append(og_embs_dict[clean_id])
            matched_ids.append(clean_id)
        else:
            print(f"⚠️ Warning: No ground truth found for {clean_id}")

    if not aligned_gen:
        raise Exception("❌ No matching pairs found!")

    # 2. Matrix Conversion
    # We convert to matrices to do the math in one 'gulp'
    gen_matrix = np.array(aligned_gen)
    og_matrix = np.array(aligned_og)

    # 3. Vectorized Math (Cosine Similarity)
    # Since they are pre-normalized, Dot Product = Cosine Similarity
    similarities = np.sum(gen_matrix * og_matrix, axis=1)

    # 4. Result Formatting
    results = {
        "mean_sim": float(np.mean(similarities)),
        "per_video": {
            vid_id: float(sim) for vid_id, sim in zip(matched_ids, similarities)
        }
    }

    return results




if __name__ == "__main__":
    import pprint
    from inference import infer_similar_audio_ultra_fast_ultimate, create_inference_example_ultimate
    from experiments import Experiment
    test_vids = [vid for vid in os.listdir(paths_config.test_video_embeddings_path)]
    my_dict = infer_similar_audio_ultra_fast_ultimate(test_vids[:2], paths_config.test_video_embeddings_path, None, 2, False, 1, 0.01)

    pprint.pp(my_dict, sort_dicts=False)
                