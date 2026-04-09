import os
import json
import subprocess
import time
import numpy as np
from inference import load_all_normed_embeddings
from path_settings import paths_config
import warnings



def prepare_batch_for_server(inferred_dict):
    # 1. Save the JSON
    
    os.makedirs(paths_config.TCML_server_input, exist_ok=True)
    json_path = os.path.join(paths_config.TCML_server_input, "inferred.json")
    with open(json_path, 'w') as f:
        json.dump(inferred_dict, f, indent=4)
    
    # 2. Build the upload list
    upload_set: set[str]= set()

    for target_video, matches in inferred_dict.items():

        # target_video_id = os.path.splitext(target_video)[0]
        # target_video_path = os.path.join(trimmed_videos_path ,f"{target_video_id}.mp4")
        # if os.path.exists(target_video_path):
        #     upload_set.add(target_video_path)
        # else:
        #     raise Exception(f"path {target_video_path} doesn't exist for video id {target_video_id}")
    
        for audio_emb in matches.keys():

            target_audio_id = os.path.splitext(audio_emb)[0]
            target_audio_path = os.path.join(paths_config.audios_path ,f"{target_audio_id}.wav")
            if os.path.exists(target_audio_path):
                upload_set.add(target_audio_path)
            else:
                raise Exception(f"path {target_audio_path} doesn't exist for video id {target_audio_id}")
            
    return json_path, upload_set



def upload_to_tcml(json_path: str, upload_set: set[str])->bool:
    remote_host = "sherkat@login3.tcml.uni-tuebingen.de"
    remote_folder = "/home/sherkat/B.sc.-Audiocraft-module/Hossein/input/"
    
    # --- 1. PRE-FLIGHT CHECK: Does the remote folder exist? ---
    # We run a simple shell command '[ -d path ]' over SSH. 
    # It returns 0 if true, 1 if false.
    print(f"🔍 Checking remote directory: {remote_folder}")
    check_cmd = ["ssh", remote_host, f"[ -d {remote_folder} ]"]
    folder_check = subprocess.run(check_cmd)
    
    if folder_check.returncode != 0:
        raise FileNotFoundError(
            f"❌ Remote error: The directory '{remote_folder}' does not exist on the server. "
            "Please create it manually before running the script."
        )

    # --- 2. UPLOAD JSON (Always overwrite) ---
    # We do this first and separately so the server always has the latest instructions
    print(f"... Uploading Json file: {os.path.basename(json_path)}")
    json_cmd = ["rsync", "-avz", json_path, f"{remote_host}:{remote_folder}"]
    subprocess.run(json_cmd, check=True)

    # --- 3. UPLOAD MEDIA (Skip if exists) ---
    manifest_file = "to_upload_manifest.txt"
    with open(manifest_file, "w") as f:
        for path in upload_set:
            f.write(f"{path}\n")

    print(f"... Syncing {len(upload_set)} media files (Skipping existing)...")
    
    # --ignore-existing: If the file name exists on the server, don't even check it, just skip.
    # --no-R: Ensures files land flat in the folder without local path structures.
    media_cmd = [
        "rsync", "-avz",
        "--ignore-existing",
        "--no-R",
        f"--files-from={manifest_file}",
        "/", 
        f"{remote_host}:{remote_folder}"
    ]
    
    result = subprocess.run(media_cmd, capture_output=True, text=True)
    
    # Cleanup the manifest locally
    if os.path.exists(manifest_file):
        os.remove(manifest_file)

    if result.returncode == 0:
        print("✅ Sync Complete. inferred audio files sent to TCML server.")
        return True
    else:
        print(f"❌ Sync Failed: {result.stderr}")
        return False

    
# def sendResultsToTCML(data_dict):
#     try:

#         print("... Preparing batch for transfer...")

#         json_path, file_set = prepare_batch_for_server(data_dict)
#         success = upload_to_tcml(json_path, file_set)

#         if success:
#             print("...Data is now on the server. Ready to trigger SLURM...")
#             # --- NEXT STEP: Trigger the sbatch here ---
#             # job_id = trigger_sbatch_remote() 
#             # return job_id
#             return True
#         else:
#             print("❌ Upload failed during the rsync process.")
#             return False

#     except Exception as e:
#         print(f"💥 An error occurred in the bridge: {e}")
#         return False


def trigger_sbatch_remote():
    remote_host = "sherkat@login3.tcml.uni-tuebingen.de"
    sbatch_command = "sbatch --parsable /home/sherkat/generate_audio.sbatch"

    print("...Submitting job to Slurm...")
    cmd = ["ssh", remote_host, sbatch_command]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Extract the ID using Regex
        job_id = result.stdout.strip()
        print(f"✅ Job submitted! ID: {job_id}")
        return job_id
            
    print(f"❌ Slurm submission failed: {result.stderr}")
    return None


def wait_for_job_completion(job_id):
    remote_host = "sherkat@login3.tcml.uni-tuebingen.de"
    sentinel_path = f"/home/sherkat/audio_outputs/JOB_{job_id}_DONE"
    
    print(f"⏳ Waiting for Job {job_id} to finish...")
    print("(You can go grab a coffee, this will take a while.)")

    while True:
        # Check if the sentinel file exists on the server
        check_cmd = ["ssh", remote_host, f"[ -f {sentinel_path} ]"]
        result = subprocess.run(check_cmd)
        
        if result.returncode == 0:
            print(f"\n ...Job {job_id} complete! Sentinel file detected...")
            return True
            
        # Optional: Print a dot to show we are still alive
        print(".", end="", flush=True)
        time.sleep(30) # Check every 30 seconds


def download_results(local_dest, inferred_dict = None):
    num_sentinel_file = 1
    num_output_audio_per_video = 2
    remote_host = "sherkat@login3.tcml.uni-tuebingen.de"
    remote_dir = "~/audio_outputs"
    remote_source = f"{remote_host}:{remote_dir}/"

    os.makedirs(local_dest, exist_ok=True)
    if inferred_dict is not None:
        original_count = len(inferred_dict)

        remote_cmd = f"ls -1 {remote_dir} | wc -l"
        cmd = ["ssh", remote_host, remote_cmd]
        temp = subprocess.run(cmd, capture_output=True, text=True)
        
        if temp.returncode != 0:
            raise Exception(f"Failed to reach server: {temp.stderr}")

        actual_count = int(temp.stdout.strip())
        expected_total = (num_output_audio_per_video * original_count) + num_sentinel_file
    
        if expected_total != actual_count:
            warnings.warn(
                f"Integrity mismatch: expected {expected_total} files "
                f"but found {actual_count}!",
                UserWarning
            )
    
    print(f"...Downloading results to {local_dest}...")
    # -a (archive), -v (verbose), -z (compress)
    cmd = ["rsync", "-av", remote_source, local_dest]
    
    subprocess.run(cmd, check=True)
    print("...All files downloaded successfully!")


def run_tcml_audio_pipeline(data_dict, output_dir = paths_config.TCML_server_output_generated):


    json_path, file_set = prepare_batch_for_server(data_dict)

    print("prepare_batch_for_server done")
    success = upload_to_tcml(json_path, file_set)
    print("upload_to_tcml done")
    if success:
        job_id = trigger_sbatch_remote()
        complete = wait_for_job_completion(job_id)
        if complete:
            download_results(output_dir, data_dict)




results = {
            '--XInAaMS6k.npy': {'-C6cbmMaENE.npy': 0.6466576988106167,
                        '-8KFpJHyspw.npy': 0.14904139426690938,
                        '-03N_1zOM4E.npy': 0.10659110957951061,
                        '-KQ7U3gS1wQ.npy': 0.05174921559957428,
                        '-HWoFxKmyyo.npy': 0.04596058174338897}
    ,



    '-0gYWIOfqdM.npy': {'-0gYWIOfqdM.npy': 0.6202382082358212,
                        '-4yCSY_5Zns.npy': 0.31029253698116976,
                        '-D7Od7iYq0A.npy': 0.04162212451171154,
                        '-A-xb-P-WxQ.npy': 0.02028793443981568,
                        '-HtBJbsbeHo.npy': 0.0075591958314815974}
    ,



    '-3M-k4nIYIM.npy': {'-9whJW7BUSU.npy': 0.3196014880596852,
                        '-HxQ9AoyRmY.npy': 0.24735819844202067,
                        '-60vY5Xw1qE.npy': 0.15602753270196487,
                        '-9vw5ZzChT0.npy': 0.14684911321636024,
                        '-3MNphBfq_0.npy': 0.13016366757996897}
    ,



    '-4ItJ9yTz_c.npy': {'-AioliAg12U.npy': 0.5865218525077042,
                        '-NPu34as_OY.npy': 0.18724029723388622,
                        '-6ZEGCtBKqs.npy': 0.10033209712998142,
                        '-Gbohom8C4Q.npy': 0.08084076879787816,
                        '-62pV95k9O0.npy': 0.045064984330550145}
    ,



    '-4o0jRbgHr4.npy': {'-NPu34as_OY.npy': 0.7716207865103134,
                        '-CexapzRAPQ.npy': 0.1304685089335117,
                        '-4o0jRbgHr4.npy': 0.05299553586319041,
                        '-9wRxzJ5j_Y.npy': 0.022736311823520462,
                        '-C8JU6yTJ40.npy': 0.022178856869464036}
    ,



    '-4rdRn-FRXo.npy': {'-Kc9P729mqM.npy': 0.36042281765518974,
                        '-7XYw1VrN64.npy': 0.33141335961690443,
                        '-MNP_aM09S8.npy': 0.1580765973593134,
                        '-CCbu3r-1pc.npy': 0.0915348425868341,
                        '-JdUSVmQq88.npy': 0.058552382781758214}
    ,



    '-6lkiUAf_cQ.npy': {'-6lkiUAf_cQ.npy': 0.7625829829950478,
                        '-ECRgvDx4xc.npy': 0.11362292998877925,
                        '-3xhrOw45ss.npy': 0.0643086250163537,
                        '-2JomCd5zzY.npy': 0.038541489457014064,
                        '-FfFD4bbCEI.npy': 0.020943972542805226}
    ,



    '-6VFTlZsft4.npy': {'-AltV1ftMk8.npy': 0.31224017893390565,
                        '-EWyYYBHsbQ.npy': 0.2170137781686798,
                        '-CcGuq0yoKo.npy': 0.20042663093106244,
                        '-0NxpZlO348.npy': 0.18478375634990274,
                        '-1EeNriiRN0.npy': 0.08553565561644937}

    }



results2 = {'-3suCV1UqMc.npy': {'-3SE2nOj6d4.npy': 0.5483170078614558,
                     '-LbwHG1fr3Q.npy': 0.23239105781365485,
                     '-9RkREsqm1o.npy': 0.21929193432488933},
 '-3GsMfOpuRM.npy': {'-LrNmB50-nA.npy': 0.5451024700459897,
                     '-9hqBiuwp2Y.npy': 0.2663383156605519,
                     '--bvmgIdDC8.npy': 0.18855921429345837},
 '-4eXuXHZw_A.npy': {'-7hQNEhU6Vk.npy': 0.998502996945732,
                     '-LLWKg5fHJs.npy': 0.0007533287131479051,
                     '-Hw2mdbqXZc.npy': 0.0007436743411202587},
 '-3L1rzGAD_o.npy': {'-E1zyCQE3es.npy': 0.6240114992250659,
                     '-39sHTky_6o.npy': 0.19476032674939506,
                     '-4ELUORuKtk.npy': 0.18122817402553895}}


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
    run_tcml_audio_pipeline(results2)