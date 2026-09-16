
import subprocess
import shutil
import numpy as np
import os
import cv2
import csv
from wakepy import keep


# TODO: make the following functions safer and refactor them to avoid repetition


def get_audio_duration(file_path):
    """Uses ffprobe to get the exact duration of a media file."""
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        file_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        # If ffprobe can't read it, the file is totally broken. Treat as 0 length.
        return 0.0
    

def scan_for_visual_corruption(FRAMES_PATH, dark_threshold=5.0, freeze_threshold=1.0, max_frozen_frames=3):
    """
    Scans extracted frames for blackouts and staggered/frozen video.
    Returns a list of bad video IDs.
    """
    print("👁️ Scanning frames for visual corruption (black frames and freezes)...")
    bad_ids = []
    
    for folder in os.listdir(FRAMES_PATH):
        folder_path = os.path.join(FRAMES_PATH, folder)
        if not os.path.isdir(folder_path):
            continue
            
        frames = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])
        if not frames:
            continue
            
        video_id = folder
        is_corrupt = False
        freeze_count = 0
        prev_img = None
        
        for frame_file in frames:
            img_path = os.path.join(folder_path, frame_file)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            # 1. Black/Dark Frame Check (0 is pitch black, 255 is pure white)
            mean_brightness = np.mean(img)
            if mean_brightness < dark_threshold:
                print(f"⬛ Black/Dark frame detected in {video_id} (Brightness: {mean_brightness:.2f})")
                is_corrupt = True
                break
                
            # 2. Stagger/Freeze Check
            if prev_img is not None:
                # Compare current frame to previous frame
                diff = cv2.absdiff(prev_img, img)
                mean_diff = np.mean(diff)
                
                # If the difference is tiny, the video is frozen/staggering
                if mean_diff < freeze_threshold:
                    freeze_count += 1
                else:
                    freeze_count = 0 # reset if video moves again
                    
                if freeze_count >= max_frozen_frames:
                    print(f"❄️ Video stagger/freeze detected in {video_id}")
                    is_corrupt = True
                    break
            
            prev_img = img
            
        if is_corrupt:
            bad_ids.append(video_id)
            
    print(f"✅ Found {len(bad_ids)} visually corrupted videos.")
    return bad_ids


def purge_video_assets(video_id, paths):
    """
    Universal cleanup function. Deletes all related files for a given video_id 
    across all specified directories if they exist.
    """
    for expected_key in ['FULL', 'TRIMMED', 'AUDIO', 'FRAMES', 'AUDIO_EMB', 'VID_EMB', 'CAV_MAE_AUD', 'CAV_MAE_VID_CLIP', 'CAV_MAE_VID_FRAME',
                         'VID_EMB_GOLDEN', 'CAV_MAE_SYNC_VID', 'CAV_MAE_SYNC_AUD']: #TODO: add  'VID_EMB_GOLDEN', 'VID_EMB_ATTENTION'
        assert expected_key in paths, f"Please provide a path for {expected_key}"

    expected_files = {
        'FULL': f"{video_id}_full.mp4",
        'TRIMMED': f"{video_id}.mp4",
        'AUDIO': f"{video_id}.wav",
        'AUDIO_EMB': f"{video_id}.npy",
        'VID_EMB': f"{video_id}.npy",
        'CAV_MAE_AUD': f"{video_id}.npy",
        'CAV_MAE_VID_CLIP': f"{video_id}.npy",
        'CAV_MAE_VID_FRAME': f"{video_id}.npy",
        'VID_EMB_GOLDEN':f"{video_id}.npy", 
        'CAV_MAE_SYNC_VID': f"{video_id}.npy",
        'CAV_MAE_SYNC_AUD': f"{video_id}.npy"
        # 'VID_EMB_ATTENTION': f"{video_id}.npy"
    }
    
    # Clean up standard files
    for key, filename in expected_files.items():
        file_path = os.path.join(paths[key], filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            print(f"couldn't find {key} file for {video_id}")
            
    # Clean up frames folder separately
    frames_path = os.path.join(paths['FRAMES'], video_id)
    if os.path.isdir(frames_path):
        shutil.rmtree(frames_path)
    else:
        print(f"couldn't find FRAMES for {video_id}")


def synchronize_and_clean_dataset(paths, sync_dirs=True, rm_short_audio=True, min_audio_duration=9.0, rm_short_frames=True, min_frames=44, rm_corrupt_vids=True):
    
    for expected_key in ['FULL', 'TRIMMED', 'AUDIO', 'FRAMES', 'AUDIO_EMB', 'VID_EMB', 'CAV_MAE_AUD', 'CAV_MAE_VID_CLIP', 'CAV_MAE_VID_FRAME',
                         'VID_EMB_GOLDEN', 'CAV_MAE_SYNC_VID', 'CAV_MAE_SYNC_AUD',]: #TODO:  'VID_EMB_GOLDEN', 'VID_EMB_ATTENTION'
        assert expected_key in paths, f"Please provide a path for {expected_key}"
        
    if rm_short_frames:
        assert min_frames > 0, "Please provide the minimum required frames > 0"
    if rm_short_audio:
        assert min_audio_duration > 0, "Please provide the minimum required wave length > 0"
    
    # Safely extract IDs using replace instead of rstrip
    full_set = {f.replace("_full.mp4", "") for f in os.listdir(paths['FULL']) if f.endswith("_full.mp4")}
    trimmed_set = {f.replace(".mp4", "") for f in os.listdir(paths['TRIMMED']) if f.endswith(".mp4")}
    audio_set = {f.replace(".wav", "") for f in os.listdir(paths['AUDIO']) if f.endswith(".wav")}
    frames_set = {f for f in os.listdir(paths['FRAMES']) if os.path.isdir(os.path.join(paths['FRAMES'], f))}
    audio_emb_set = {f.replace(".npy", "") for f in os.listdir(paths['AUDIO_EMB']) if f.endswith(".npy")}
    vid_emb_set = {f.replace(".npy", "") for f in os.listdir(paths['VID_EMB']) if f.endswith(".npy")}
    cav_mae_vid_clip_set = {f.replace(".npy", "") for f in os.listdir(paths['CAV_MAE_VID_CLIP']) if f.endswith(".npy")}
    cav_mae_vid_frame_set = {f.replace(".npy", "") for f in os.listdir(paths['CAV_MAE_VID_FRAME']) if f.endswith(".npy")}
    cav_mae_aud_set = {f.replace(".npy", "") for f in os.listdir(paths['CAV_MAE_AUD']) if f.endswith(".npy")}
    vid_emb_golden_set = {f.replace(".npy", "") for f in os.listdir(paths['VID_EMB_GOLDEN']) if f.endswith(".npy")}
    cav_mae_sync_vid_set = {f.replace(".npy", "") for f in os.listdir(paths['CAV_MAE_SYNC_VID']) if f.endswith(".npy")}
    cav_mae_sync_aud_set = {f.replace(".npy", "") for f in os.listdir(paths['CAV_MAE_SYNC_AUD']) if f.endswith(".npy")}
    # vid_emb_attention_set = {f.replace(".npy", "") for f in os.listdir(paths['VID_EMB_ATTENTION']) if f.endswith(".npy")}
    
    # Create a master list of EVERY id across all folders
    all_ids = (full_set | trimmed_set | audio_set | frames_set | audio_emb_set | vid_emb_set 
               | cav_mae_vid_clip_set| cav_mae_vid_frame_set 
               | cav_mae_aud_set
               | cav_mae_sync_vid_set | cav_mae_sync_aud_set
               | vid_emb_golden_set
          ) #TODO  'VID_EMB_GOLDEN', 'VID_EMB_ATTENTION'

    valid_set = set(all_ids) 


    
    if sync_dirs:
        # If an ID is missing from ANY set, it is incomplete and must be purged
        incomplete_ids= set()
        for vid_id in valid_set:
            if not (vid_id in full_set and vid_id in trimmed_set and 
                                    vid_id in audio_set and vid_id in frames_set and 
                                    vid_id in audio_emb_set and vid_id in vid_emb_set and 
                                    vid_id in cav_mae_vid_clip_set and vid_id in cav_mae_vid_frame_set and vid_id in cav_mae_aud_set ):
                incomplete_ids.add(vid_id)

        print(f"incomplete ids={incomplete_ids}")
        confirm = input(f" {len(incomplete_ids)} videos will be removed. Type 'Continue' to continue...") == 'Continue'
        if not confirm : raise Exception("jumping out")
        for vid_id in incomplete_ids:
                purge_video_assets(vid_id, paths)
                valid_set.discard(vid_id) # discard is safe and won't throw errors

    if rm_short_audio:
        # Convert valid_set to a list for iteration so we can modify the set safely during the loop
        for vid_id in list(valid_set):
            audio_path = os.path.join(paths['AUDIO'], f"{vid_id}.wav")
            duration = get_audio_duration(audio_path)
            if duration < min_audio_duration:
                print(f"✂️ Short audio detected: {vid_id} ({duration:.2f}s)")
                purge_video_assets(vid_id, paths)
                valid_set.discard(vid_id)

    if rm_short_frames:
        for vid_id in list(valid_set):
            frames_path = os.path.join(paths['FRAMES'], vid_id)
            if os.path.exists(frames_path) and len(os.listdir(frames_path)) < min_frames:
                purge_video_assets(vid_id, paths)
                valid_set.discard(vid_id)

    if rm_corrupt_vids:
        # scan_for_visual_corruption processes the current state of disk, which is fine
        corrupt_vid_ids = scan_for_visual_corruption(paths['FRAMES'])
        for vid_id in corrupt_vid_ids:
            purge_video_assets(vid_id, paths)
            valid_set.discard(vid_id)
            
    return valid_set





if __name__ == "__main__":
    from path_settings import paths_config
    paths = {        
        'FULL': paths_config.test.full_videos,
        'TRIMMED': paths_config.test.trimmed_videos,
        'FRAMES': paths_config.test.frames,
        'AUDIO': paths_config.test.audios,
        'AUDIO_EMB': paths_config.test.audio_embeddings,
        'VID_EMB': paths_config.test.video_embeddings,
        'CAV_MAE_AUD': paths_config.test.cav_mae_audio,
        'CAV_MAE_VID_CLIP': paths_config.test.cav_mae_video_clip,
        'CAV_MAE_VID_FRAME': paths_config.test.cav_mae_video_frame,
        'VID_EMB_GOLDEN':paths_config.test.video_embeddings_golden_frame, 
        'CAV_MAE_SYNC_VID': paths_config.test.cav_mae_sync_video,
        'CAV_MAE_SYNC_AUD': paths_config.test.cav_mae_sync_audio
        }

    synchronize_and_clean_dataset(paths, True, False, 9, False, 44, False)