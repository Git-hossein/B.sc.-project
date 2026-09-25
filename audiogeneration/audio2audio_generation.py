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

SBATCH_AUDIOCRAFT_CONTINUATION_PATH = "/home/sherkat/B.sc.-Audiocraft-module/run_audiogen_continuation.sbatch"
SBATCH_AUDIOLDM2_PATH = "/home/sherkat/B.sc.-Audiocraft-module/run_audioldm2_audio2audio.sbatch"
AUDIOCRAFT_REMOTE_OUTPUT_DIR = "/home/sherkat/audiogen_outputs"
AUDIOLDM2_REMOTE_OUTPUT_DIR = "/home/sherkat/audioLDM2_generation_outputs"
TCML_UPLOAD_DIR = "/home/sherkat/B.sc.-Audiocraft-module/Hossein/input/"
LOCAL_PORT = 22
REMOTE_PORT = 443
SESSION_ID_FORMAT = "%Y%m%d_%H%M%S_%f"
CONFIG_JSON = "config.json"
CHUNK_JSON = "chunk_{idx}.json"


def chunk_dict(data, n_chunks):
    """Splits a dictionary into n balanced parts, preserving order."""
    if not data:
        raise ValueError("cannot chunk empty dictionary")
    if n_chunks <= 0:
        raise ValueError("n_chunks cannot be 0 or negative")
    if n_chunks == 1:
        return [data.copy()]

    items = list(data.items())
    num_items = len(items)
    n_chunks = min(n_chunks, num_items)
    chunk_size, remainder = divmod(num_items, n_chunks)

    chunks = []
    start = 0
    for i in range(n_chunks):
        # Distribute 1 extra item to the first 'remainder' chunks
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(dict(items[start:end]))
        start = end

    return chunks


def prepare_parallel_batch(inferred_dict, retrieved_audios_path, num_GPU, options):
    # 1. Create a unique session folder locally
    session_id = datetime.datetime.now().strftime(SESSION_ID_FORMAT)
    local_session_path = os.path.join(paths_config.TCML_server_input, session_id)
    os.makedirs(local_session_path, exist_ok=True)

    # 2. Save Global Config
    config_path = os.path.join(local_session_path, CONFIG_JSON)
    with open(config_path, 'w') as f:
        json.dump(options, f, indent=4)

    # 3. Split and Save Chunks
    chunks = chunk_dict(inferred_dict, num_GPU)
    for idx, chunk in enumerate(chunks):
        chunk_path = os.path.join(local_session_path, CHUNK_JSON.format(idx=idx))
        with open(chunk_path, 'w') as f:
            json.dump(chunk, f, indent=4)

    # 4. Build Media Upload Set 
    upload_set = set()
    for matches in inferred_dict.values():
        for audio_emb in matches.keys():
            target_audio_id = os.path.splitext(audio_emb)[0]                           
            target_audio_path = os.path.join(retrieved_audios_path, f"{target_audio_id}.wav")
            if os.path.exists(target_audio_path):
                upload_set.add(target_audio_path)
            else:
                raise FileNotFoundError(f"Missing: {target_audio_path}")
            
    return local_session_path, upload_set, session_id, len(chunks)



def upload_parallel_to_tcml(local_session_dir, upload_set, retrieved_audios_path, session_id, port):
    remote_host = "sherkat@login3.tcml.uni-tuebingen.de"
    # Create a unique path for this specific run
    remote_base = TCML_UPLOAD_DIR
    remote_session_folder = os.path.join(remote_base, session_id)
    
    print(f"🚀 Creating remote session: {session_id}")
    subprocess.run(["ssh", remote_host, "-p", str(port),f"mkdir -p {remote_session_folder}"], check=True)

    # --- Upload Chunks and Config ---
    print("... Uploading JSON chunks and config")
    # Using '/*' to send the contents of the local folder to the remote session folder
    subprocess.run(["rsync", "-avz", "-e", f"ssh -p {port}",f"{local_session_dir}/", f"{remote_host}:{remote_session_folder}"], check=True)

    # --- Upload Media (to the shared flat input folder to save space/time) ---
    # keep media in the base folder so we don't re-upload the same .wav for different sessions
    media_manifest_path = os.path.join(local_session_dir, "to_upload_manifest.txt")
    with open(media_manifest_path, "w") as f:
        for path in upload_set:
            f.write(f"{os.path.basename(path)}\n")

    print(f"... Syncing {len(upload_set)} media files to shared input folder")

    try:
        subprocess.run([
        "rsync", "-avz", "--ignore-existing", "--no-R",
        "-e", f"ssh -p {port}",
        f"--files-from={media_manifest_path}",
        os.path.join(retrieved_audios_path, ""), 
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
        if os.path.exists(media_manifest_path):
            os.remove(media_manifest_path)



def trigger_parallel_sbatch(sbatch_script_path ,session_id, port, num_chunks):
    if not num_chunks>0: 
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
        # Extract the ID
        array_job_id = result.stdout.strip()
        print(f"✅ Array Job submitted! ID: {array_job_id}")
        return array_job_id
            
    print(f"❌ Slurm submission failed: {result.stderr}")
    return None


def wait_for_job_completion(sentinel_base_dir ,session_id, port, num_chunks, timeout_minutes):
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


def download_parallel_results_zipped(remote_target_dir, session_id, port, total_videos, local_down_dest):
    """
    Downloads session results by first archiving on the server, then downloading,
    flattening locally, and verifying integrity.
    """
    remote_host = "sherkat@login3.tcml.uni-tuebingen.de"
    
    # Paths on Server
    remote_parent_dir = remote_target_dir
    remote_archive_name = f"{session_id}.tar.gz"
    remote_archive_path = os.path.join(remote_parent_dir, remote_archive_name)

    # Paths Locally (Define BEFORE try to prevent UnboundLocalError in finally)
    os.makedirs(local_down_dest, exist_ok=True)
    local_archive_path = os.path.join(local_down_dest, remote_archive_name)
    extracted_folder = os.path.join(local_down_dest, session_id)

    try:
        # --- 1. REMOTE COMPRESSION ---
        print(f"📦 Compressing session {session_id} on TCML...")
        remote_tar_cmd = [
            "ssh", "-p", str(port), remote_host,
            f"tar -czf {remote_archive_path} -C {remote_parent_dir} {session_id}"
        ]
        subprocess.run(remote_tar_cmd, check=True)

        # --- 2. DOWNLOAD THE SINGLE ARCHIVE ---
        print(f"📥 Downloading compressed archive...")
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
                
                os.replace(src_path, dst_path)
                
                if file.endswith(".wav"):
                    audio_count += 1

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

    finally:
        # --- 6. FINAL CLEANUP ---
        # Remove DONE_ sentinels
        for sentinel in [f for f in os.listdir(local_down_dest) if f.startswith("DONE_")]:
            os.remove(os.path.join(local_down_dest, sentinel))

        # Cleanup extracted folder shell and archive safely
        if os.path.exists(extracted_folder):
            shutil.rmtree(extracted_folder)
        if os.path.exists(local_archive_path):
            os.remove(local_archive_path)

        # Remove the archive from the server to save space
        subprocess.run(["ssh", "-p", str(port), remote_host, f"rm -f {remote_archive_path}"], check=False)

    return local_down_dest


def run_parallel_tcml_pipeline_audiocraft_continuation(inferred_dict, 
                                                       retrieved_audios_path, 
                                                       local_down_dst, 
                                                       timeout_minutes, 
                                                       num_GPU, port, 
                                                       options = None, 
                                                       auto_download = True):
    
    """Runs the parallel AudioGen continuation pipeline on the TCML remote server.

    Chunks the inference results, uploads necessary configurations and media assets, triggers 
    a Slurm job array to process audio continuation in parallel across multiple GPUs, 
    and optionally waits for completion, downloads, and extracts the generated results.

    Args:
        inferred_dict: Dictionary containing video queries and their retrieved audio matches.
        retrieved_audios_path: Local directory path containing the retrieved reference audio files.
        local_down_dst: Local destination directory to save the final downloaded results.
        timeout_minutes: Maximum duration to wait for remote job completion.
        num_GPU: Number of GPUs to use for the audiogeneration task (splits the workload into 'num_GPU' many subtasks).
        port: SSH port for remote connection to the TCML server.
        options: Dictionary of AudioGen continuation configuration parameters (uses defaults if None).
        auto_download: If True, automatically waits for task completion and downloads results.

    Returns:
        The unique session identifier string for this run.
    """
    
    if options is None:
        options = {
            "cfg_coef": 3.0,
            "with_text_descr": True,
            "shift": 0.0,
            "prompt_duration": 2,
            "num_audio_mix": 5,
            "weight_by": "softmax_score"
        }

    # 1. Chunk and Upload
    # (Using the prepare_parallel_batch and upload functions we discussed)
    local_path, file_set, session_id, num_chunks = prepare_parallel_batch(inferred_dict, retrieved_audios_path, num_GPU, options)
    
    success = upload_parallel_to_tcml(local_path, file_set, retrieved_audios_path, session_id, port)
    
    if success:
        # 2. Trigger Dynamic Array
        try:
            job_id = trigger_parallel_sbatch(SBATCH_AUDIOCRAFT_CONTINUATION_PATH ,session_id, port, num_chunks)
            if job_id is None: 
                raise Exception(f"job was not properly triggered: job_id = {job_id}")
            
            # 3. Wait for all sentinels
            if auto_download and wait_for_job_completion(AUDIOCRAFT_REMOTE_OUTPUT_DIR, session_id, port, num_chunks, timeout_minutes):
                
                download_parallel_results_zipped(
                    remote_target_dir= AUDIOCRAFT_REMOTE_OUTPUT_DIR,
                    session_id=session_id, 
                    port= port,
                    local_down_dest=local_down_dst, 
                    total_videos=len(inferred_dict))

        finally:
            shutil.rmtree(local_path)
    else:
        raise Exception(f"❌ Upload failed for session {session_id}")

    return session_id





def run_parallel_tcml_pipeline_audioldm2(inferred_dict, 
                                         retrieved_audios_path, 
                                         local_down_dst, 
                                         timeout_minutes, 
                                         num_GPU, port, 
                                         options = None, 
                                         auto_download = True):

    """Runs the parallel AudioLDM2 style transfer pipeline on the TCML remote server.

    Chunks the inference results, uploads necessary configurations and media assets, triggers 
    a Slurm job array to process audio-to-audio generation and style transfer in parallel 
    across multiple GPUs, and optionally waits for completion, downloads, and extracts the results.

    Args:
        inferred_dict: Dictionary containing video queries and their retrieved audio matches.
        retrieved_audios_path: Local directory path containing the retrieved reference audio files.
        local_down_dst: Local destination directory to save the final downloaded results.
        timeout_minutes: Maximum duration to wait for remote job completion.
        num_GPU: Number of GPUs to use for the audiogeneration task (splits the workload into 'num_GPU' many subtasks).
        port: SSH port for remote connection to the TCML server.
        options: Dictionary of AudioLDM2 configuration parameters (uses defaults if None).
        auto_download: If True, automatically waits for task completion and downloads results.

    Returns:
        The unique session identifier string for this run.
    """

    if options is None:
        options = {
        "strength": 0.45,
        "num_inference_steps": 100,
        "guidance_scale": 3.0,
        "with_text_descr": True,
        "num_audio_mix": 1,
        "weight_by": "softmax_score"
        }

    # 1. Chunk and Upload
    # (Using the prepare_parallel_batch and upload functions we discussed)
    local_path, file_set, session_id, num_chunks = prepare_parallel_batch(inferred_dict, retrieved_audios_path, num_GPU, options)
    
    success = upload_parallel_to_tcml(local_path, file_set,retrieved_audios_path, session_id, port)
    
    if success:
        # 2. Trigger Dynamic Array
        job_id = trigger_parallel_sbatch(SBATCH_AUDIOLDM2_PATH ,session_id, port, num_chunks)
        
        # 3. Wait for all sentinels
        if auto_download and wait_for_job_completion(AUDIOLDM2_REMOTE_OUTPUT_DIR, session_id, port, num_chunks, timeout_minutes):
            
            download_parallel_results_zipped(
                remote_target_dir= AUDIOLDM2_REMOTE_OUTPUT_DIR,
                session_id=session_id, 
                port= port,
                local_down_dest=local_down_dst, 
                total_videos=len(inferred_dict))
            
    return session_id





if __name__ == "__main__":
    pass
