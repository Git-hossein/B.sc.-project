import os
import json
import subprocess
from path_settings import paths_config



def prepare_batch_for_server(inferred_dict):
    # 1. Save the JSON
    json_path = os.path.join(paths_config._TCML_server_input, "inferred.json")
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
    
        for segment_file in matches.keys():

            target_audio_id = os.path.splitext(segment_file)[0]
            target_audio_path = os.path.join(paths_config.audios_path ,f"{target_audio_id}.mp4")
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

    
def sendResultsToTCML(data_dict):
    try:

        print("... Preparing batch for transfer...")

        json_path, file_set = prepare_batch_for_server(data_dict)
        success = upload_to_tcml(json_path, file_set)

        if success:
            print("...Data is now on the server. Ready to trigger SLURM...")
            # --- NEXT STEP: Trigger the sbatch here ---
            # job_id = trigger_sbatch_remote() 
            # return job_id
            return True
        else:
            print("❌ Upload failed during the rsync process.")
            return False

    except Exception as e:
        print(f"💥 An error occurred in the bridge: {e}")
        return False
