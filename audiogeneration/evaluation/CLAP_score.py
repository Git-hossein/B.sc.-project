import os
import glob
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import ClapModel, ClapFeatureExtractor
from tqdm import tqdm

# -------------------------------------------------------------------------
# Environment Variables & Paths
# -------------------------------------------------------------------------
SCRATCH_GT = os.environ.get("SCRATCH_GT")
SCRATCH_GEN = os.environ.get("SCRATCH_GEN")
SCRATCH_RAW_MIX = os.environ.get("SCRATCH_RAW_MIX")

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 4))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not SCRATCH_GT or not os.path.isdir(SCRATCH_GT):
    raise ValueError(f"SCRATCH_GT directory invalid or not found: {SCRATCH_GT}")
if not SCRATCH_GEN or not os.path.isdir(SCRATCH_GEN):
    raise ValueError(f"SCRATCH_GEN directory invalid or not found: {SCRATCH_GEN}")

# Check whether raw_mix directory exists and actually has files
EVAL_RAW_MIX = False
if SCRATCH_RAW_MIX and os.path.isdir(SCRATCH_RAW_MIX):
    raw_mix_files_check = glob.glob(os.path.join(SCRATCH_RAW_MIX, "*.wav"))
    if len(raw_mix_files_check) > 0:
        EVAL_RAW_MIX = True

print("==================================================")
print(f"Device           : {DEVICE}")
print(f"SCRATCH_GT       : {SCRATCH_GT}")
print(f"SCRATCH_GEN      : {SCRATCH_GEN}")
print(f"SCRATCH_RAW_MIX  : {SCRATCH_RAW_MIX} (Active: {EVAL_RAW_MIX})")
print("==================================================")

# -------------------------------------------------------------------------
# File Matching & Assertions
# -------------------------------------------------------------------------
gen_files = sorted(glob.glob(os.path.join(SCRATCH_GEN, "*_GEN.wav")))
if not gen_files:
    raise FileNotFoundError(f"No '*_GEN.wav' files found in {SCRATCH_GEN}")

if EVAL_RAW_MIX:
    raw_mix_files = sorted(glob.glob(os.path.join(SCRATCH_RAW_MIX, "*_RAW_MIX.wav")))
    assert len(raw_mix_files) == len(gen_files), (
        f"Length mismatch: {len(raw_mix_files)} RAW_MIX files found, "
        f"but {len(gen_files)} GEN files exist."
    )

valid_pairs = []
for gen_path in gen_files:
    filename = os.path.basename(gen_path)
    gt_filename = filename.replace("_GEN.wav", ".wav")
    gt_path = os.path.join(SCRATCH_GT, gt_filename)

    if not os.path.exists(gt_path):
        continue

    raw_mix_path = None
    if EVAL_RAW_MIX:
        raw_filename = filename.replace("_GEN.wav", "_RAW_MIX.wav")
        raw_mix_path = os.path.join(SCRATCH_RAW_MIX, raw_filename)
        if not os.path.exists(raw_mix_path):
            continue

    valid_pairs.append({
        "gt": gt_path,
        "gen": gen_path,
        "raw_mix": raw_mix_path
    })

print(f"Successfully paired {len(valid_pairs)} complete sample sets.")
assert len(valid_pairs) > 0, "Zero matching pairs found between GT and GEN."
if EVAL_RAW_MIX:
    assert len(valid_pairs) == len(gen_files), (
        f"Some files could not be matched across all three folders: "
        f"{len(valid_pairs)} matched vs {len(gen_files)} generated."
    )


# -------------------------------------------------------------------------
# Dataset & DataLoader (48 kHz mono required by CLAP)
# -------------------------------------------------------------------------
class AudioListDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.paths[idx])
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 48000:
            wav = torchaudio.functional.resample(wav, sr, 48000)
        return wav.squeeze(0).numpy()


MODEL_ID = "laion/clap-htsat-fused"
print(f"\nLoading model: {MODEL_ID}")
feature_extractor = ClapFeatureExtractor.from_pretrained(MODEL_ID)
model = ClapModel.from_pretrained(MODEL_ID).to(DEVICE).eval()


def collate_fn(batch):
    return feature_extractor(batch, sampling_rate=48000, return_tensors="pt")


@torch.no_grad()
def extract_embeddings(file_paths, desc="Encoding"):
    dataset = AudioListDataset(file_paths)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    all_embeds = []
    for batch in tqdm(loader, desc=desc):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        features = model.get_audio_features(**batch)
        norm_features = F.normalize(features, p=2, dim=-1)
        all_embeds.append(norm_features.cpu())

    return torch.cat(all_embeds, dim=0)


# -------------------------------------------------------------------------
# Similarity Computations
# -------------------------------------------------------------------------
gt_paths = [p["gt"] for p in valid_pairs]
gen_paths = [p["gen"] for p in valid_pairs]

print("\nExtracting Ground-Truth audio embeddings...")
E_gt = extract_embeddings(gt_paths, desc="GT Audio")

print("\nExtracting Generated audio embeddings...")
E_gen = extract_embeddings(gen_paths, desc="GEN Audio")

# Paired 1-to-1 Cosine similarity
sim_gen_gt = (E_gen * E_gt).sum(dim=-1)

# Shuffled baseline floor (rolls GT index by 1)
N = E_gt.shape[0]
sim_random_floor = (E_gen * E_gt[torch.roll(torch.arange(N), shifts=1)]).sum(dim=-1).mean().item() if N > 1 else float("nan")

print("\n" + "=" * 50)
print(f"EVALUATION RESULTS ({N} samples):")
print(f"Mean Cosine Sim (GEN vs GT)  : {sim_gen_gt.mean().item():.4f} +/- {sim_gen_gt.std().item():.4f}")
print(f"Shuffled Baseline Floor     : {sim_random_floor:.4f}")

if EVAL_RAW_MIX:
    raw_paths = [p["raw_mix"] for p in valid_pairs]
    print("\nExtracting RAW_MIX audio embeddings...")
    E_raw = extract_embeddings(raw_paths, desc="RAW_MIX Audio")

    sim_raw_gt = (E_raw * E_gt).sum(dim=-1)
    sim_gen_raw = (E_gen * E_raw).sum(dim=-1)

    print(f"Mean Cosine Sim (RAW vs GT)  : {sim_raw_gt.mean().item():.4f} +/- {sim_raw_gt.std().item():.4f}")
    print(f"Mean Cosine Sim (GEN vs RAW) : {sim_gen_raw.mean().item():.4f} +/- {sim_gen_raw.std().item():.4f}")

print("=" * 50)