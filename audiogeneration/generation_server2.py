import os
import json
import subprocess
import time
import numpy as np
from path_settings import paths_config
import shutil
import warnings
import datetime
import glob
import tarfile


SBATCH_AUDIOCRAFT_GENERATION_PATH = "/home/sherkat/B.sc.-Audiocraft-module/run_audiogen_generation.sbatch"
SBATCH_AUDIOLDM2_GENERATION_PATH = "/home/sherkat/B.sc.-Audiocraft-module/run_audioldm2_pure_generation.sbatch"
AUDIOCRAFT_REMOTE_OUTPUT_DIR = "/home/sherkat/audiogen_outputs"
AUDIOLDM2_REMOTE_OUTPUT_DIR = "/home/sherkat/audioLDM2_generation_outputs"
TCML_UPLOAD_DIR = "/home/sherkat/B.sc.-Audiocraft-module/Hossein/input/"
SESSION_ID_FORMAT = "%Y%m%d_%H%M%S_%f"
CONFIG_JSON = "config.json"
VID_CHUNK_JSON = "chunk_{idx}.json"
LOCAL_PORT = 22
REMOTE_PORT = 443


def chunk_list(lst, n_chunks):
    """Splits a list into n roughly equal parts."""

    if not lst:
        raise ValueError("cannot chunk empty list")
    if n_chunks <= 0:
        raise ValueError("n_chunks cannot be 0 or negative")
    if n_chunks == 1:
        return [lst.copy()]

    # Calculate size of each chunk
    num_items = len(lst)
    n_chunks = min(n_chunks, num_items)
    size, remainder = divmod(num_items, n_chunks)
    
    chunks = []
    start = 0
    for i in range(n_chunks):
        end = start + size + (1 if i < remainder else 0)
        chunk = lst[start:end]
        start = end
        chunks.append(chunk)
    return chunks


def prepare_parallel_batch(vid_list, num_GPU, options):
    # 1. Create a unique session folder locally
    session_id = datetime.datetime.now().strftime(SESSION_ID_FORMAT)
    local_session_path = os.path.join(paths_config.TCML_server_input, session_id)
    os.makedirs(local_session_path, exist_ok=True)

    # 2. Save Global Config
    config_path = os.path.join(local_session_path, CONFIG_JSON)
    with open(config_path, 'w') as f:
        json.dump(options, f, indent=4)

    # 3. Split and Save Chunks
    chunks = chunk_list(vid_list, num_GPU)
    for idx, chunk in enumerate(chunks):
        chunk_path = os.path.join(local_session_path, VID_CHUNK_JSON.format(idx=idx))
        with open(chunk_path, 'w') as f:
            json.dump(chunk, f, indent=4)
            
    return local_session_path, session_id, len(chunks)



def upload_parallel_to_tcml(local_session_dir, session_id, port):
    remote_host = "sherkat@login3.tcml.uni-tuebingen.de"
    # We now create a unique path for this specific run
    remote_base = TCML_UPLOAD_DIR
    remote_session_folder = os.path.join(remote_base, session_id)
    
    print(f"🚀 Creating remote session: {session_id}")
    subprocess.run(["ssh", "-p", str(port), remote_host, f"mkdir -p {remote_session_folder}"], check=True)

    # --- Upload Chunks and Config ---
    print("... Uploading JSON chunks and config")
    # Using '/*' to send the contents of the local folder to the remote session folder
    subprocess.run(["rsync", "-avz", "-e", f"ssh -p {port}", os.path.join(local_session_dir, ""), f"{remote_host}:{remote_session_folder}"], check=True)
    return True


def trigger_parallel_sbatch(sbatch_script_path, session_id, port, num_chunks):
    if num_chunks<= 0: 
        raise ValueError("cannot trigger a parallel job with zero or negative num_chunks")

    remote_host = "sherkat@login3.tcml.uni-tuebingen.de"
    array_range = f"0-{num_chunks - 1}"

    sbatch_command = (
        f"sbatch --parsable "
        f"--export=ALL,SESSION_ID={session_id} "
        f"--array={array_range} "
        f"{sbatch_script_path}"
    )

    print(f"... submitting Job Array: {array_range} for Session: {session_id} ...")
    cmd = ["ssh", "-p", str(port), remote_host, sbatch_command]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Extract the ID using Regex
        array_job_id = result.stdout.strip()
        print(f"✅ Array Job submitted! ID: {array_job_id}")
        return array_job_id
            
    print(f"❌ Slurm submission failed: {result.stderr}")
    return None


def wait_for_job_completion(sentinel_base_dir, session_id, port, num_chunks, timeout_minutes):
    remote_host = "sherkat@login3.tcml.uni-tuebingen.de"
    sentinel_dir = os.path.join(sentinel_base_dir, session_id)
    
    print(f"⏳ Waiting for {num_chunks} tasks to finish...")
    print("(You can go grab a coffee, this will take a while.)")

    start = time.time()
    while True:
        # Check if the sentinel file exists on the server
        count_cmd = f"ls {sentinel_dir}/DONE_* 2>/dev/null | wc -l"
        result = subprocess.run(["ssh", "-p", str(port), remote_host, count_cmd], capture_output=True, text=True)
        
        finished_count = int(result.stdout.strip()) if result.stdout.strip() else 0
        print(f"\r progress: [{finished_count}/{num_chunks}] tasks complete...", end="", flush=True)
        
        if finished_count >= num_chunks:
            print(f"\n✨ All {num_chunks} tasks finished successfully!")
            return True
        
        if time.time() - start > timeout_minutes * 60:
            print("\n ⏰ Timeout reached - some tasks probably failed!")
            return False
            
        time.sleep(60)



def run_parallel_tcml_pipeline_audioldm2_generation(
    video_list, 
    local_down_dst, 
    timeout_minutes, 
    num_GPU, 
    port, 
    options=None, 
    auto_download=True
):
    """
    Orchestrates pure text-to-audio parallel generation for AudioLDM2 across TCML nodes.
    Prepares chunks, uploads session config, triggers Slurm array, monitors completion,
    and downloads/verifies generated audio files.
    """
    if options is None:
        options = {
            "negative_prompt": "low quality, distorted, noisy, glitch",
            "num_inference_steps": 100,
            "guidance_scale": 3.5,
            "audio_length_in_s": 10.0,
            "seed": 42
        }

    # 1. Chunk and Upload Session Data
    local_path, session_id, num_chunks = prepare_parallel_batch(video_list, num_GPU, options)
    
    success = upload_parallel_to_tcml(local_path, session_id, port)
    
    if not success:
        shutil.rmtree(local_path, ignore_errors=True)
        raise RuntimeError(f"❌ Upload failed for session {session_id}")

    try:
        # 2. Trigger Dynamic Slurm Array
        job_id = trigger_parallel_sbatch(
            sbatch_script_path=SBATCH_AUDIOLDM2_GENERATION_PATH,
            session_id=session_id,
            port=port,
            num_chunks=num_chunks
        )
        if job_id is None:
            raise RuntimeError(f"❌ Slurm job array was not properly triggered (job_id: {job_id})")

        # 3. Monitor Sentinel Files and Download
        if auto_download:
            job_finished = wait_for_job_completion(
                sentinel_base_dir=AUDIOLDM2_REMOTE_OUTPUT_DIR,
                session_id=session_id,
                port=port,
                num_chunks=num_chunks,
                timeout_minutes=timeout_minutes
            )

            if job_finished:
                download_parallel_results_zipped(
                    remote_target_dir=AUDIOLDM2_REMOTE_OUTPUT_DIR,
                    session_id=session_id,
                    port=port,
                    total_videos=len(video_list),
                    local_down_dest=local_down_dst
                )
            else:
                raise TimeoutError(
                    f"⏰ Generation timed out after {timeout_minutes} minutes. "
                    f"Session results not auto-downloaded: {session_id}"
                )
    finally:
        # 4. Clean up local preparation folder
        if os.path.exists(local_path):
            shutil.rmtree(local_path)

    return session_id


def run_parallel_tcml_pipeline_audiocraft_generation(video_list, local_down_dst, timeout_minutes, num_GPU, port, options = None, auto_download = True):
    if options is None:
        options = {
            "cfg_coef": 3.0,
        }

    # 1. Chunk and Upload
    # (Using the prepare_parallel_batch and upload functions we discussed)
    local_path, session_id, num_chunks = prepare_parallel_batch(video_list, num_GPU, options)
    
    success = upload_parallel_to_tcml(local_path, session_id, port)
    
    if success:
        try:
            # 2. Trigger Dynamic Array
            job_id = trigger_parallel_sbatch(SBATCH_AUDIOCRAFT_GENERATION_PATH, session_id, port, num_chunks)
            if job_id is None: 
                raise Exception(f"Slurm job array was not properly triggered : job_id = {job_id}")
            
            # 3. Wait for all sentinels
            if auto_download and wait_for_job_completion(AUDIOCRAFT_REMOTE_OUTPUT_DIR, session_id, port, num_chunks, timeout_minutes):
                
                download_parallel_results_zipped(
                    AUDIOCRAFT_REMOTE_OUTPUT_DIR,
                    session_id, 
                    port,
                    local_down_dest=local_down_dst, 
                    total_videos=len(video_list))
        finally:
            shutil.rmtree(local_path)

    else:
        shutil.rmtree(local_path, ignore_errors=True)
        raise Exception(f"❌ Upload failed for session {session_id}")
            
    return session_id



def download_parallel_results_zipped(remote_target_dir, session_id, port, total_videos, local_down_dest):
    """
    Downloads session results by first archiving on the server, then downloading,
    flattening locally, and verifying integrity. Identical behavior to the legacy rsync version.
    """
    remote_host = "sherkat@login3.tcml.uni-tuebingen.de"
    
    # Paths on Server
    remote_parent_dir = remote_target_dir
    remote_archive_name = f"{session_id}.tar.gz"
    remote_archive_path = os.path.join(remote_parent_dir, remote_archive_name)

    # Paths Locally
    os.makedirs(local_down_dest, exist_ok=True)
    local_archive_path = os.path.join(local_down_dest, remote_archive_name)
    extracted_folder = os.path.join(local_down_dest, session_id)

    try:
        # --- 1. REMOTE COMPRESSION ---
        print(f"📦 Compressing session {session_id} on TCML...")
        # -C changes to the parent dir so the archive contains the 'session_id' folder relatively
        remote_tar_cmd = [
            "ssh", "-p", str(port), remote_host,
            f"tar -czf {remote_archive_path} -C {remote_parent_dir} {session_id}"
        ]
        subprocess.run(remote_tar_cmd, check=True)

        # --- 2. DOWNLOAD THE SINGLE ARCHIVE ---
        print(f"📥 Downloading compressed archive...")
        # Using rsync to download the single file (allows resuming if interrupted)
        rsync_cmd = [
            "rsync", "-avz",
            "--info=progress2",
            "-e", f"ssh -p {port}",
            f"{remote_host}:{remote_archive_path}",
            local_down_dest
        ]
        subprocess.run(rsync_cmd, check=True)

        # --- 3. LOCAL EXTRACTION ---
        print(f"🔓 Extracting locally...")
        with tarfile.open(local_archive_path, "r:gz") as tar:
            if hasattr(tarfile, 'data_filter'):
                tar.extractall(path=local_down_dest, filter='data')
            else:
                tar.extractall(path=local_down_dest)

        # --- 4. LOCAL FLATTENING & CLEANUP ---
        print(f"🔄 Flattening files into {local_down_dest}...")
        audio_count = 0
        
        # Walk through the extracted session folder
        for root, dirs, files in os.walk(extracted_folder, topdown=False):
            for file in files:
                src_path = os.path.join(root, file)
                dst_path = os.path.join(local_down_dest, file)
                
                # Move and Replace (matches original behavior)
                os.replace(src_path, dst_path)
                
                if file.endswith(".wav"):
                    audio_count += 1

        # --- 5. LOCAL INTEGRITY VERIFICATION ---
        expected_audio = total_videos 
        print(f"🧐 Verifying integrity...")
        if audio_count == expected_audio:
            print(f"✅ Success! All {audio_count} audio files present and accounted for.")
        else:
            warnings.warn(
                f"⚠️ Integrity Issue: Expected {expected_audio} audio files, "
                f"but found {audio_count}. Check the Slurm logs for failed tasks.",
                UserWarning
            )
    finally:
        # --- 6. FINAL CLEANUP ---
        # Remove DONE_ sentinels (Identical to legacy cleanup)
        if os.path.exists(local_down_dest):
            for sentinel in [f for f in os.listdir(local_down_dest) if f.startswith("DONE_")]:
                os.remove(os.path.join(local_down_dest, sentinel))

        # Cleanup the extracted folder shell and the local archive
        if os.path.exists(extracted_folder):
            shutil.rmtree(extracted_folder)
        if os.path.exists(local_archive_path):
            os.remove(local_archive_path)

        # Remove the archive from the server to save space
        subprocess.run(["ssh", "-p", str(port), remote_host, f"rm -f {remote_archive_path}"], check=False)

    return local_down_dest



if __name__ == "__main__":
    print("we're here")
    from audioinference.inference import infer_similar_audio_ultra_fast_ultimate, load_all_normed_embeddings
    from audioinference.reranking import sync_rerank_and_slice




    my_queries = os.listdir(paths_config.test.cav_mae_video_clip)


    # audioldm2_options = {
    #         "negative_prompt": "low quality, distorted, noisy, glitch",
    #         "num_inference_steps": 100,
    #         "guidance_scale": 3.5,
    #         "audio_length_in_s": 10.0,
    #         "seed": 42
    #     }

    # run_parallel_tcml_pipeline_audioldm2_generation(
    #     video_list=my_queries,
    #     local_down_dst="/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/experiments/unseen/experiment_audioldm2_baseline_01",
    #     timeout_minutes=120,
    #     num_GPU=30,
    #     port=REMOTE_PORT,
    #     options=audioldm2_options,
    #     auto_download=False
    # )

    job_finished = wait_for_job_completion(
                    sentinel_base_dir=AUDIOLDM2_REMOTE_OUTPUT_DIR,
                    session_id="20260916_115448_883672",
                    port=LOCAL_PORT,
                    num_chunks=30,
                    timeout_minutes=120
                )
    
    if job_finished:
        download_parallel_results_zipped(
            remote_target_dir=AUDIOLDM2_REMOTE_OUTPUT_DIR,
            session_id="20260916_115448_883672",
            port=LOCAL_PORT,
            total_videos=len(my_queries),
            local_down_dest="/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/experiments/unseen/experiment_audioldm2_baseline_01"
        )