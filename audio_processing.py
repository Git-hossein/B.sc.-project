
import wav2clip
import os
import subprocess
import csv
import numpy as np
import soundfile as sf
from path_settings import paths_config

def vggsound_extract_audio_from_videos_batch(trimmed_videos_dir, audios_dir, sample_rate=16000, log_file=paths_config.extract_audio_log_file):
    """
    Extracts audio from trimmed videos as .wav files (mono, resampled to sample_rate)
    and logs failures.

    Args:
        trimmed_videos_dir (str): Path to folder with trimmed .mp4 videos.
        audios_dir (str): Path to save extracted audio files.
        sample_rate (int): Target sample rate for audio (default 16 kHz for Wav2CLIP).
        log_file (str): CSV file to log failures.
    """
    
    print("======== begin batch audio extraction ========")

    videos = [f for f in os.listdir(trimmed_videos_dir) if f.endswith(".mp4")]
    print(f"Found {len(videos)} trimmed videos")

    for video in os.listdir(trimmed_videos_dir):
        if not video.endswith(".mp4"):
            print(f"skipping {video}: non mp4 file")
            continue
        video_path = os.path.join(trimmed_videos_dir, video)
        video_id = os.path.splitext(video)[0]
        audio_file = os.path.join(audios_dir, f"{video_id}.wav")
        
        if os.path.exists(audio_file):
            print(f"Audio for {video_id} already exists. Skipping.")
            continue
        
        cmd = [
            paths_config.ffmpeg_path,
            "-y",  # overwrite if exists
            "-i", video_path,
            "-ac", "1",  # mono
            "-ar", str(sample_rate),  # resample
            audio_file,
            "-hide_banner",
            "-loglevel", "error"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Extracted audio for {video_id} successfully ✔✔")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to extract audio for {video_id}: {e}")
            # Log failure
            with open(log_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([video_id, "failed"])


def vggsound_extract_audio_embeddings_batch(audios_dir, embeddings_dir, wav2clip_model = None, log_file=paths_config.embeddings_audio_log_file):
    """
    Extracts clip-level Wav2CLIP embeddings for each .wav audio file.
    Saves each embedding as a .npy file: <video_id>.npy
    Logs failures in a CSV.
    """

    if wav2clip_model is None:
        wav2clip_model = wav2clip.get_model()

    audio_files = [f for f in os.listdir(audios_dir) if f.endswith(".wav")]
    print(f"Found {len(audio_files)} audio files for embedding.")

    for audio_file in audio_files:
        video_id = os.path.splitext(audio_file)[0]
        audio_path = os.path.join(audios_dir, audio_file)
        embedding_path = os.path.join(embeddings_dir, f"{video_id}.npy")

        # Skip existing
        if os.path.exists(embedding_path):
            print(f"Embedding for {video_id} already exists. Skipping.")
            continue

        try:
            # Load waveform (Wav2CLIP expects raw PCM float32)
            audio_waveform, sr = sf.read(audio_path)

            # If stereo → convert to mono by averaging channels
            if len(audio_waveform.shape) == 2:
                audio_waveform = audio_waveform.mean(axis=1)

            # Ensure float32
            audio_waveform = audio_waveform.astype(np.float32)

            # Compute embedding
            embedding = wav2clip.embed_audio(audio_waveform, wav2clip_model)

            # Save
            np.save(embedding_path, embedding)

            print(f"✔ Embedded audio for {video_id}")

        except Exception as e:
            print(f"⚠️ Failed embedding for {video_id}: {e}")
            with open(log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([video_id, "failed"])