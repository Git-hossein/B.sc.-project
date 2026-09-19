import numpy as np
import soundfile as sf
import librosa
from scipy.ndimage import gaussian_filter1d
import os
import shutil
from pathlib import Path
from typing import Optional

def _load_mono(path: str, target_sr: int = 22050):
    """
    Loads audio using soundfile, averages channels to mono,
    and resamples to target_sr using librosa if needed.
    """
    data, sr = sf.read(path, dtype='float32')
    
    # Average multichannel to mono
    if data.ndim > 1:
        data = np.mean(data, axis=-1)
        
    # Resample to ensure identical hop intervals
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
        
    return data, sr


def _compute_onset_envelope(y: np.ndarray, sr: int, fps: int = 25) -> np.ndarray:
    """
    Computes spectral flux onset strength over Mel frequency bands.
    This emphasizes transient events (hits, barks, footsteps) over steady noise.
    """
    hop_length = int(sr / fps)
    
    # librosa onset strength extracts spectral flux
    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=hop_length,
        aggregate=np.median
    )
    return onset_env


def compute_aa_alignment(
    audio_path_gen: str,
    audio_path_gt: str,
    fps: int = 25,
    max_tolerance_sec: float = 0.5,
    smooth_sigma_sec: float = 0.08,
    return_offset: bool = False
):
    """
    Computes Audio-to-Audio Temporal Alignment via Onset Spectral Flux.

    Parameters:
    ----------
    audio_path_gen : str
        Path to the generated/retrieved audio file.
    audio_path_gt : str
        Path to the reference ground-truth audio file.
    fps : int, default=25
        Target frame rate for the envelope (25 fps = 40ms resolution).
    max_tolerance_sec : float, default=0.5
        Maximum acceptable lag offset (+/- window) for peak correlation search.
    smooth_sigma_sec : float, default=0.08
        Gaussian smoothing std-dev in seconds (softens transient timing mismatch).
    return_offset : bool, default=False
        If True, returns (sync_0, optimal_offset_sec, max_corr).

    Returns:
    -------
    float or tuple:
        - sync_0 (float): Zero-lag Pearson correlation [-1.0, 1.0].
        - optimal_offset_sec (float): Best temporal shift in seconds within the tolerance window.
        - max_corr (float): Pearson correlation at optimal offset [-1.0, 1.0].
    """
    # 1. Load and resample audio
    sr_shared = 22050
    y_gen, _ = _load_mono(audio_path_gen, target_sr=sr_shared)
    y_gt, _ = _load_mono(audio_path_gt, target_sr=sr_shared)

    # 2. Match durations
    min_len = min(len(y_gen), len(y_gt))
    if min_len == 0:
        return (0.0, 0.0, 0.0) if return_offset else 0.0

    y_gen = y_gen[:min_len]
    y_gt = y_gt[:min_len]

    # 3. Compute onset envelopes (spectral flux)
    env_gen = _compute_onset_envelope(y_gen, sr=sr_shared, fps=fps)
    env_gt = _compute_onset_envelope(y_gt, sr=sr_shared, fps=fps)

    # Match frame lengths
    min_frames = min(len(env_gen), len(env_gt))
    env_gen = env_gen[:min_frames]
    env_gt = env_gt[:min_frames]

    # 4. Temporal smoothing (allows slight sub-100ms tolerance for diffusion jitter)
    sigma_frames = smooth_sigma_sec * fps
    if sigma_frames > 0:
        env_gen = gaussian_filter1d(env_gen, sigma=sigma_frames)
        env_gt = gaussian_filter1d(env_gt, sigma=sigma_frames)

    # 5. Standardize (zero-mean, unit-variance)
    std_gen = np.std(env_gen)
    std_gt = np.std(env_gt)

    if std_gen < 1e-7 or std_gt < 1e-7:
        return (0.0, 0.0, 0.0) if return_offset else 0.0

    norm_gen = (env_gen - np.mean(env_gen)) / std_gen
    norm_gt = (env_gt - np.mean(env_gt)) / std_gt

    # 6. Zero-lag Pearson correlation (Sync0)
    sync_0 = float(np.mean(norm_gen * norm_gt))
    sync_0 = float(np.clip(sync_0, -1.0, 1.0))

    if not return_offset:
        return sync_0

    # 7. Cross-correlation across all time lags
    cross_corr = np.correlate(norm_gen, norm_gt, mode='full') / len(norm_gen)
    lags = np.arange(-len(norm_gen) + 1, len(norm_gen))

    # 8. Windowing: Constrain search to +/- max_tolerance_sec
    max_lag_frames = int(max_tolerance_sec * fps)
    center_idx = len(norm_gen) - 1

    start_idx = max(0, center_idx - max_lag_frames)
    end_idx = min(len(cross_corr), center_idx + max_lag_frames + 1)

    windowed_corr = cross_corr[start_idx:end_idx]
    windowed_lags = lags[start_idx:end_idx]

    max_idx = np.argmax(windowed_corr)
    optimal_lag_frames = windowed_lags[max_idx]
    optimal_offset_sec = float(optimal_lag_frames / fps)
    max_corr = float(np.clip(windowed_corr[max_idx], -1.0, 1.0))

    return sync_0, optimal_offset_sec, max_corr





if __name__ == "__main__":
    from vggsound.cleanup import purge_video_assets, synchronize_and_clean_dataset
    aud_dir = "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/validation/audios"
    import os


    # paths = {
    #         'FULL': "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/validation/full_videos",
    #         'TRIMMED': "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/validation/trimmed_videos",
    #         'AUDIO': "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/validation/audios",
    #         'AUDIO_EMB': "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/validation/audio_embeddings",
    #         'VID_EMB': "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/validation/video_embeddings",
    #         'FRAMES': "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/validation/frames",
    #         'CAV_MAE_AUD': "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/validation/cav_mae/audio_embeddings", 
    #         'CAV_MAE_VID_CLIP': "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/validation/cav_mae/video_embeddings/10FrameClipLevel",
    #         'CAV_MAE_VID_FRAME': "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/vggsound/validation/cav_mae/video_embeddings/10FrameFrameLevel",

    #     }

    print("AA ALIGN")
    from path_settings import paths_config
    gen_path = "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/experiments/unseen/experiment_audioldm2_comparison_03"
    gt_path = paths_config.test.audios

    gen_audios = os.listdir(gen_path)
    offset = 0
    sync_0s, optimal_offset_secs , max_corrs = [np.full((4966,), 3, dtype=np.float32) for _ in range(3)]

    for i, gen_aud in enumerate(gen_audios):
        if not gen_aud.endswith("_RAW_MIX.wav"):
            offset += 1
            continue

        gt_audio_name = gen_aud.replace("_RAW_MIX.wav", ".wav")
        if i-offset < 5:
            print(f"generated {gen_aud} vs GT {gt_audio_name}")
        sync_0, optimal_offset_sec, max_corr = compute_aa_alignment(os.path.join(gen_path, gen_aud), os.path.join(gt_path, gt_audio_name), return_offset=True)

        sync_0s[i-offset] = sync_0
        optimal_offset_secs[i-offset] = optimal_offset_sec
        max_corrs[i-offset] = max_corr

    print(f"mean sync_0: {np.mean(sync_0s)}")
    print(f"median sync_0: {np.median(sync_0s)}")

    print(f"mean optimal_offset_sec: {np.mean(optimal_offset_secs)}")
    print(f"median optimal_offset_sec: {np.median(optimal_offset_secs)}")

    print(f"mean sync_max_corr0: {np.mean(max_corrs)}")
    print(f"median max_corr: {np.median(max_corrs)}")