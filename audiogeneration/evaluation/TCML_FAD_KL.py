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
    # torchaudio expects (channels, time)
    if wav.ndim == 1:
        wav = wav[None, :]
    else:
        wav = wav.T
    return torch.from_numpy(wav), sr

def _patched_audio_save(uri, src, sample_rate, **kwargs):
    # torchaudio passes (channels, time) tensor
    wav = src.cpu().numpy()
    if wav.ndim == 2:
        wav = wav.T
    # "PCM_16" is the soundfile equivalent of torchaudio's 16-bit PCM request
    sf.write(uri, wav, sample_rate, subtype="PCM_16")

torchaudio.load = _patched_audio_load
torchaudio.save = _patched_audio_save


# ---------------------------------------------------------------------------
# 1. Parallel File Copy & Peak Normalization
# ---------------------------------------------------------------------------
def _process_single_file(args: Tuple[Path, Path, float]) -> None:
    src_file, dst_file, target_peak = args
    if dst_file.exists():
        return
    
    audio, sr = sf.read(src_file)
    
    # Fast-fail assert: Enforce 16kHz on the raw data immediately
    assert sr == 16000, f"Audio {src_file.name} is {sr} Hz. Expected exactly 16000 Hz."
    
    max_val = np.max(np.abs(audio))
    if max_val > 1e-6:
        audio = (audio / max_val) * target_peak
    sf.write(dst_file, audio, sr)


def peak_normalize_and_prep(
    source_dir: str,
    gt_dir: str,
    scratch_gen_dir: str,
    scratch_gt_dir: str,
    scratch_raw_mix_dir: Optional[str] = None,
    target_db: float = -1.0,
    num_workers: int = 4,
):
    src_path = Path(source_dir)
    gt_path = Path(gt_dir)
    gen_path = Path(scratch_gen_dir)
    gt_out_path = Path(scratch_gt_dir)
    raw_path = Path(scratch_raw_mix_dir) if scratch_raw_mix_dir else None

    gen_path.mkdir(parents=True, exist_ok=True)
    gt_out_path.mkdir(parents=True, exist_ok=True)
    if raw_path:
        raw_path.mkdir(parents=True, exist_ok=True)

    target_peak = 10.0 ** (target_db / 20.0)
    tasks: List[Tuple[Path, Path, float]] = []

    # Prepare GEN and RAW_MIX
    for file in src_path.glob("*.wav"):
        if file.name.endswith("_GEN.wav"):
            target_dest = gen_path / (file.name[:-8] + ".wav")
            tasks.append((file, target_dest, target_peak))
        elif raw_path and file.name.endswith("_RAW_MIX.wav"):
            target_dest = raw_path / (file.name[:-12] + ".wav")
            tasks.append((file, target_dest, target_peak))

    # Prepare Ground Truth
    for file in gt_path.glob("*.wav"):
        target_dest = gt_out_path / file.name
        tasks.append((file, target_dest, target_peak))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(_process_single_file, tasks))

    print(f"Prepared & normalized: {len(list(gen_path.glob('*.wav')))} _GEN files in {gen_path}", flush=True)
    print(f"Prepared & normalized: {len(list(gt_out_path.glob('*.wav')))} GT files in {gt_out_path}", flush=True)
    if raw_path:
        print(f"Prepared & normalized: {len(list(raw_path.glob('*.wav')))} _RAW_MIX files in {raw_path}", flush=True)


# ---------------------------------------------------------------------------
# 2. Fast Batched Audio Dataset & Loader for KL Divergence
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
    
    # num_workers=0 to prevent CUDA context fork error
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_pad,
        pin_memory=(device == "cuda"),
    )

    # Note: PaSST expects 32kHz audio input to generate its spectrograms properly.
    resampler = torchaudio.transforms.Resample(16000, 32000).to(device)

    kl_accum = 0.0
    total_samples = 0

    with torch.no_grad():
        for b_gen, b_gt in loader:
            b_gen = b_gen.to(device, non_blocking=True)
            b_gt = b_gt.to(device, non_blocking=True)

            b_gen_32k = resampler(b_gen)
            b_gt_32k = resampler(b_gt)

            out_gen = torch.sigmoid(model(b_gen_32k))
            out_gt = torch.sigmoid(model(b_gt_32k))

            p_gen_norm = out_gen / (out_gen.sum(dim=-1, keepdim=True) + 1e-8)
            p_gt_norm = out_gt / (out_gt.sum(dim=-1, keepdim=True) + 1e-8)

            # CORRECTED: D_KL(P_gen || P_gt). input=log(target_dist), target=input_dist
            kl_batch = F.kl_div((p_gt_norm + 1e-8).log(), p_gen_norm, reduction="none").sum(dim=-1)
            
            kl_accum += kl_batch.sum().item()
            total_samples += b_gen.size(0)

    return (kl_accum / total_samples) if total_samples > 0 else 0.0


# ---------------------------------------------------------------------------
# 3. Main Evaluation Runner
# ---------------------------------------------------------------------------
def compute_metrics(gen_dir: str, gt_dir: str, num_workers: int = 4) -> Dict[str, float]:
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
    SOURCE_FOLDER = os.environ.get("SOURCE_FOLDER")
    GT_FOLDER = os.environ.get("GT_FOLDER")
    SCRATCH_BASE_DIR = os.environ.get("SCRATCH_BASE_DIR")
    with_raw_mix = os.environ.get("WITH_RAW_MIX")

    if not SOURCE_FOLDER or not GT_FOLDER or not SCRATCH_BASE_DIR:
        raise ValueError("SOURCE_FOLDER, GT_FOLDER, and SCRATCH_BASE_DIR must be set.")

    scratch_gen = os.path.join(SCRATCH_BASE_DIR, "gen")
    scratch_gt = os.path.join(SCRATCH_BASE_DIR, "gt")
    scratch_raw_mix = os.path.join(SCRATCH_BASE_DIR, "raw_mix") if with_raw_mix else None

    print("Preparing and normalizing audio files in parallel...", flush=True)
    peak_normalize_and_prep(
        source_dir=SOURCE_FOLDER,
        gt_dir=GT_FOLDER,
        scratch_gen_dir=scratch_gen,
        scratch_gt_dir=scratch_gt,
        scratch_raw_mix_dir=scratch_raw_mix,
        num_workers=4,
    )

    print("\n--- Evaluating GEN against Ground Truth ---", flush=True)
    results_gen = compute_metrics(gen_dir=scratch_gen, gt_dir=scratch_gt, num_workers=4)
    for k, v in results_gen.items():
        print(f"{k}: {v:.4f}")

    if scratch_raw_mix:
        print("\n--- Evaluating RAW_MIX against Ground Truth ---", flush=True)
        results_raw = compute_metrics(gen_dir=scratch_raw_mix, gt_dir=scratch_gt, num_workers=4)
        for k, v in results_raw.items():
            print(f"{k}: {v:.4f}")