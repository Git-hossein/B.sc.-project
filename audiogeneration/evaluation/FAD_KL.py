import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor

import hear21passt.base
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader

from fadtk import FrechetAudioDistance
from fadtk.model_loader import VGGishModel
from fadtk.fad_batch import cache_embedding_files

# =====================================================================
# MONKEY-PATCH: Force torchaudio to bypass torchcodec and use soundfile
# =====================================================================
def _patched_audio_load(uri, **kwargs):
    wav, sr = sf.read(uri, dtype="float32")
    if wav.ndim == 1:
        wav = wav[None, :]
    else:
        wav = wav.T
    return torch.from_numpy(wav), sr

def _patched_audio_save(uri, src, sample_rate, **kwargs):
    wav = src.cpu().numpy()
    if wav.ndim == 2:
        wav = wav.T
    sf.write(uri, wav, sample_rate, subtype="PCM_16")

torchaudio.load = _patched_audio_load
torchaudio.save = _patched_audio_save


# ---------------------------------------------------------------------------
# 1. Parallel In-Place Normalization & Renaming
# ---------------------------------------------------------------------------
def _process_single_file_in_place(args: Tuple[Path, str, float]) -> None:
    file_path, suffix_to_remove, target_peak = args
    if not file_path.exists():
        return
    
    audio, sr = sf.read(file_path)
    
    assert sr == 16000, f"Audio {file_path.name} is {sr} Hz. Expected exactly 16000 Hz."
    
    max_val = np.max(np.abs(audio))
    if max_val > 1e-6:
        audio = (audio / max_val) * target_peak
        
    # remove suffix's like _GEN.wav and _RAW_MIX.wav
    if suffix_to_remove and file_path.name.endswith(suffix_to_remove):
        new_name = file_path.name.replace(suffix_to_remove, ".wav")
        new_path = file_path.parent / new_name
        sf.write(new_path, audio, sr)
        file_path.unlink()
    else:
        # In-place overwrite (for Ground Truth)
        sf.write(file_path, audio, sr)


def peak_normalize_in_place(
    scratch_gen_dir: str,
    scratch_gt_dir: str,
    scratch_raw_mix_dir: Optional[str] = None,
    target_db: float = -1.0,
    num_workers: int = 8,
):
    gen_path = Path(scratch_gen_dir)
    gt_out_path = Path(scratch_gt_dir)
    raw_path = Path(scratch_raw_mix_dir) if scratch_raw_mix_dir else None

    target_peak = 10.0 ** (target_db / 20.0)
    tasks: List[Tuple[Path, str, float]] = []

    # 1. Queue GEN files (stripping "_GEN.wav")
    gen_files = list(gen_path.glob("*_GEN.wav"))
    for file in gen_files:
        tasks.append((file, "_GEN.wav", target_peak))

    # 2. Queue RAW_MIX files (stripping "_RAW_MIX.wav") if they exist
    if raw_path and raw_path.exists():
        raw_files = list(raw_path.glob("*_RAW_MIX.wav"))
        for file in raw_files:
            tasks.append((file, "_RAW_MIX.wav", target_peak))

    # 3. Queue Ground Truth files (no stripping needed)
    gt_files = list(gt_out_path.glob("*.wav"))
    for file in gt_files:
        tasks.append((file, "", target_peak))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(_process_single_file_in_place, tasks))

    print(f"Normalized & Renamed: {len(gen_files)} _GEN files in {gen_path}", flush=True)
    print(f"Normalized: {len(gt_files)} GT files in {gt_out_path}", flush=True)
    if raw_path and raw_path.exists():
        print(f"Normalized & Renamed: {len(raw_files)} _RAW_MIX files in {raw_path}", flush=True)


# ---------------------------------------------------------------------------
# 2. BAtCHED Audio Dataset and Loader for KL Divergence
# ---------------------------------------------------------------------------
class PairedAudioDataset(Dataset):
    def __init__(self, gen_dir: str, gt_dir: str):
        gen_path = Path(gen_dir)
        gt_path = Path(gt_dir)
        
        self.pairs: List[Tuple[Path, Path]] = []
        for g_file in sorted(gen_path.glob("*.wav")):
            gt_file = gt_path / g_file.name
            if gt_file.exists():
                self.pairs.append((g_file, gt_file))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        gen_file, gt_file = self.pairs[idx]
        
        wav_gen, _ = sf.read(gen_file, dtype="float32")
        wav_gt, _ = sf.read(gt_file, dtype="float32")
        
        if wav_gen.ndim > 1:
            wav_gen = np.mean(wav_gen, axis=-1)
        if wav_gt.ndim > 1:
            wav_gt = np.mean(wav_gt, axis=-1)

        return torch.from_numpy(wav_gen), torch.from_numpy(wav_gt)


def collate_fn_pad(batch):
    gen_list, gt_list = zip(*batch)
    max_len_gen = max(x.shape[0] for x in gen_list)
    max_len_gt = max(x.shape[0] for x in gt_list)
    max_len = max(max_len_gen, max_len_gt)

    gen_padded = torch.stack([F.pad(x, (0, max_len - x.shape[0])) for x in gen_list])
    gt_padded = torch.stack([F.pad(x, (0, max_len - x.shape[0])) for x in gt_list])
    return gen_padded, gt_padded


def compute_kl_divergence_batched(
    gen_dir: str, gt_dir: str, device: str, batch_size: int = 16
) -> float:
    
    model = hear21passt.base.get_basic_model(mode="logits")
    model.eval().to(device)

    dataset = PairedAudioDataset(gen_dir, gt_dir)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_pad,
        pin_memory=(device == "cuda"),
    )

    resampler = torchaudio.transforms.Resample(16000, 32000).to(device)

    kl_accum = 0.0
    total_samples = 0

    with torch.no_grad():
        for b_gen, b_gt in loader:
            b_gen = b_gen.to(device, non_blocking=True)
            b_gt = b_gt.to(device, non_blocking=True)

            b_gen_32k = resampler(b_gen)
            b_gt_32k = resampler(b_gt)

            target_length = 320000

            def pad_or_truncate(x, target_len):
                if x.shape[-1] > target_len:
                    return x[..., :target_len]
                elif x.shape[-1] < target_len:
                    return F.pad(x, (0, target_len - x.shape[-1]))
                return x

            b_gen_32k = pad_or_truncate(b_gen_32k, target_length)
            b_gt_32k = pad_or_truncate(b_gt_32k, target_length)

            out_gen = torch.sigmoid(model(b_gen_32k))
            out_gt = torch.sigmoid(model(b_gt_32k))
            p_gen_norm = out_gen / (out_gen.sum(dim=-1, keepdim=True) + 1e-8)
            p_gt_norm = out_gt / (out_gt.sum(dim=-1, keepdim=True) + 1e-8)

            # D_KL(P_gen || P_gt)
            kl_batch = F.kl_div((p_gt_norm + 1e-8).log(), p_gen_norm, reduction="none").sum(dim=-1)
            
            kl_accum += kl_batch.sum().item()
            total_samples += b_gen.size(0)

    return (kl_accum / total_samples) if total_samples > 0 else 0.0


# ---------------------------------------------------------------------------
# 3. Main Evaluation Runner
# ---------------------------------------------------------------------------
def compute_metrics(gen_dir: str, gt_dir: str, num_workers: int = 8) -> Dict[str, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Computing FAD (VGGish)...", flush=True)
    fad_model = VGGishModel()
    
    gt_files = list(Path(gt_dir).glob("*.wav"))
    gen_files = list(Path(gen_dir).glob("*.wav"))
    
    cache_embedding_files(gt_files, fad_model, workers=num_workers)
    cache_embedding_files(gen_files, fad_model, workers=num_workers)

    fad_calc = FrechetAudioDistance(fad_model, audio_load_worker=num_workers, load_model=False)
    fad_score = fad_calc.score(gt_dir, gen_dir)

    print("Computing KL Divergence (PaSST, Batched GPU)...", flush=True)
    kl_score = compute_kl_divergence_batched(
        gen_dir, gt_dir, device=device, batch_size=16
    )

    return {
        "fad_vggish": float(fad_score),
        "kl_divergence": float(kl_score),
    }


if __name__ == "__main__":
    SCRATCH_GT = os.environ.get("SCRATCH_GT")
    SCRATCH_GEN = os.environ.get("SCRATCH_GEN")
    SCRATCH_RAW_MIX = os.environ.get("SCRATCH_RAW_MIX")

    if not SCRATCH_GT or not SCRATCH_GEN:
        raise ValueError("SCRATCH_GT and SCRATCH_GEN must be set in the environment.")

    # ---------------------------------------------------------
    # Sanity Checks & Assertions
    # ---------------------------------------------------------
    gen_files_count = len(list(Path(SCRATCH_GEN).glob("*_GEN.wav")))
    has_raw_mix = False
    
    if SCRATCH_RAW_MIX and Path(SCRATCH_RAW_MIX).exists():
        raw_files_count = len(list(Path(SCRATCH_RAW_MIX).glob("*_RAW_MIX.wav")))
        if raw_files_count > 0:
            assert raw_files_count == gen_files_count, (
                f"Mismatch: Found {raw_files_count} RAW_MIX files, but {gen_files_count} GEN files. "
                "Ensure both folders are populated evenly."
            )
            print(f"Assertion passed: {raw_files_count} RAW_MIX files match {gen_files_count} GEN files.", flush=True)
            has_raw_mix = True

    print("\nNormalizing audio files in-place on scratch drive...", flush=True)
    peak_normalize_in_place(
        scratch_gen_dir=SCRATCH_GEN,
        scratch_gt_dir=SCRATCH_GT,
        scratch_raw_mix_dir=SCRATCH_RAW_MIX if has_raw_mix else None,
        num_workers=8,
    )

    print("\n--- Evaluating GEN against Ground Truth ---", flush=True)
    results_gen = compute_metrics(gen_dir=SCRATCH_GEN, gt_dir=SCRATCH_GT, num_workers=8)
    for k, v in results_gen.items():
        print(f"{k}: {v:.4f}")

    if has_raw_mix:
        print("\n--- Evaluating RAW_MIX against Ground Truth ---", flush=True)
        results_raw = compute_metrics(gen_dir=SCRATCH_RAW_MIX, gt_dir=SCRATCH_GT, num_workers=8)
        for k, v in results_raw.items():
            print(f"{k}: {v:.4f}")