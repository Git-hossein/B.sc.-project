import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
import os
import shutil
from pathlib import Path
from typing import Optional

def _load_mono(path: str):
    """Loads audio using soundfile and converts stereo to mono if needed."""
    data, sr = sf.read(path, dtype='float32')
    # If multichannel (channels-last in soundfile), average to mono
    if data.ndim > 1:
        data = np.mean(data, axis=-1)
    return data, sr

def _compute_rms_envelope(y: np.ndarray, sr: int, fps: int = 25) -> np.ndarray:
    """Computes frame-level RMS energy envelope purely in NumPy."""
    hop_length = int(sr / fps)
    frame_length = int(hop_length * 2)
    
    # Pad signal so frames are centered
    pad_width = frame_length // 2
    y_padded = np.pad(y, pad_width, mode='reflect')
    
    # Calculate number of frames
    n_frames = 1 + (len(y_padded) - frame_length) // hop_length
    
    # Vectorized sliding window using stride tricks (fast & zero-copy)
    shape = (n_frames, frame_length)
    strides = (y_padded.strides[0] * hop_length, y_padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(y_padded, shape=shape, strides=strides)
    
    # RMS = sqrt(mean(frame^2))
    rms = np.sqrt(np.mean(frames ** 2, axis=-1))
    return rms

def compute_aa_alignment(
    audio_path_gen: str,
    audio_path_gt: str,
    fps: int = 25,
    smooth_sigma_sec: float = 0.1,
    return_offset: bool = False
):
    """
    Computes Audio-to-Audio Envelope Alignment (AA-Align) using soundfile and numpy.

    Parameters:
    ----------
    audio_path_gen : str
        Path to the generated/retrieved audio file.
    audio_path_gt : str
        Path to the reference ground-truth audio file.
    fps : int, default=25
        Target frame rate for the energy envelope (25 fps = 40ms resolution).
    smooth_sigma_sec : float, default=0.1
        Gaussian smoothing std-dev in seconds (spreads narrow transients).
    return_offset : bool, default=False
        If True, also returns the optimal lag offset (in seconds) and peak correlation.

    Returns:
    -------
    float or tuple:
        - sync_0 (float): Zero-lag Pearson correlation [-1.0, 1.0].
        - If return_offset=True: (sync_0, optimal_offset_sec, max_corr)
    """
    # 1. Load audio files using soundfile
    y_gen, sr_gen = _load_mono(audio_path_gen)
    y_gt, sr_gt = _load_mono(audio_path_gt)

    # 2. Match duration in seconds
    dur_gen = len(y_gen) / sr_gen
    dur_gt = len(y_gt) / sr_gt
    min_dur = min(dur_gen, dur_gt)
    
    if min_dur <= 0:
        return (0.0, 0.0, 0.0) if return_offset else 0.0

    y_gen = y_gen[:int(min_dur * sr_gen)]
    y_gt = y_gt[:int(min_dur * sr_gt)]

    # 3. Compute frame-level RMS energy envelope at the shared target FPS
    env_gen = _compute_rms_envelope(y_gen, sr=sr_gen, fps=fps)
    env_gt = _compute_rms_envelope(y_gt, sr=sr_gt, fps=fps)

    # Align envelope lengths to match exactly
    min_frames = min(len(env_gen), len(env_gt))
    env_gen = env_gen[:min_frames]
    env_gt = env_gt[:min_frames]

    # 4. Apply 1D Gaussian smoothing
    sigma_frames = smooth_sigma_sec * fps
    if sigma_frames > 0:
        env_gen = gaussian_filter1d(env_gen, sigma=sigma_frames)
        env_gt = gaussian_filter1d(env_gt, sigma=sigma_frames)

    # 5. Standardize (zero-mean, unit variance)
    std_gen = np.std(env_gen)
    std_gt = np.std(env_gt)

    if std_gen < 1e-7 or std_gt < 1e-7:
        return (0.0, 0.0, 0.0) if return_offset else 0.0

    norm_gen = (env_gen - np.mean(env_gen)) / std_gen
    norm_gt = (env_gt - np.mean(env_gt)) / std_gt

    # 6. Zero-Lag Pearson Correlation (Sync0)
    sync_0 = float(np.mean(norm_gen * norm_gt))
    sync_0 = float(np.clip(sync_0, -1.0, 1.0))

    if not return_offset:
        return sync_0

    # 7. Cross-correlation across all time lags
    cross_corr = np.correlate(norm_gen, norm_gt, mode='full') / len(norm_gen)
    lags = np.arange(-len(norm_gen) + 1, len(norm_gen))

    max_idx = np.argmax(cross_corr)
    optimal_lag_frames = lags[max_idx]
    optimal_offset_sec = float(optimal_lag_frames / fps)
    max_corr = float(np.clip(cross_corr[max_idx], -1.0, 1.0))

    return sync_0, optimal_offset_sec, max_corr





if __name__ == "__main__":
    from vggsound.cleanup import purge_video_assets, synchronize_and_clean_dataset
    aud_dir = "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/validation/audios"
    import os


    paths = {
            'FULL': "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/validation/full_videos",
            'TRIMMED': "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/validation/trimmed_videos",
            'AUDIO': "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/validation/audios",
            'AUDIO_EMB': "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/validation/audio_embeddings",
            'VID_EMB': "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/validation/video_embeddings",
            'FRAMES': "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/validation/frames",
            'CAV_MAE_AUD': "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/validation/cav_mae/audio_embeddings", 
            'CAV_MAE_VID_CLIP': "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/validation/cav_mae/video_embeddings/10FrameClipLevel",
            'CAV_MAE_VID_FRAME': "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/validation/cav_mae/video_embeddings/10FrameFrameLevel",

        }

    #synchronize_and_clean_dataset(paths, sync_dirs=True, rm_short_audio=False, rm_short_frames=False, rm_corrupt_vids=False)
    for vid in os.listdir(paths['AUDIO']):
        aud_path = os.path.join(paths['AUDIO'], vid)
        score = compute_aa_alignment(aud_path, aud_path)
        if not np.isclose(score, 1.0, atol=1e-6):
           print(f"video {vid} has score {score}")
        #    purge_video_assets(os.path.splitext(vid)[0], paths)
