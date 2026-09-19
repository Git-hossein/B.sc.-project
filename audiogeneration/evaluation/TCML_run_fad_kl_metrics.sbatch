#!/bin/bash
#SBATCH --job-name=eval_audio
#SBATCH --partition=day
#SBATCH --gres=gpu:2080ti:1
#SBATCH --cpus-per-task=8
#SBATCH --error=job.fad_kl_gen.%A.err
#SBATCH --output=job.fad_kl_gen.%A.out
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# -------------------------------------------------------------
# Scratch Directory Setup & Cleanup Trap
# -------------------------------------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SCRATCH_BASE_DIR="/scratch/${SLURM_JOB_ID}"

# Uncomment if evaluating RAW_MIX:
# export WITH_RAW_MIX="1"


echo "Preparing scratch folders"

LOCAL_GT_DIR="/home/sherkat/eval_generation/test_split_GT_audios"
LOCAL_GEN_DIR="/home/sherkat/eval_generation/experiment_audioldm2_ablation_03"

SCRATCH_GT="$SCRATCH_BASE_DIR/gt"
SCRATCH_GEN="$SCRATCH_BASE_DIR/gen"
SCRATCH_RAW_MIX="$SCRATCH_BASE_DIR/raw_mix"

export SCRATCH_GT
export SCRATCH_GEN
export SCRATCH_RAW_MIX

cleanup() {
    echo "Cleaning up scratch directory: $SCRATCH_BASE_DIR"

    rm -rf "${SCRATCH_BASE_DIR:?}/"*

    echo "Scratch cleanup complete."
}

trap cleanup EXIT SIGINT SIGTERM



mkdir -p "$SCRATCH_GT" "$SCRATCH_GEN"

echo "Copying ground truth..."
cp "$LOCAL_GT_DIR"/* "$SCRATCH_GT/"

echo "Copying generated files..."
cp "$LOCAL_GEN_DIR"/*_GEN.wav "$SCRATCH_GEN/"

if [[ -n "$WITH_RAW_MIX" ]]; then
    mkdir -p "$SCRATCH_RAW_MIX"

    echo "Copying RAW_MIX files..."
    cp "$LOCAL_GEN_DIR"/*_RAW_MIX.wav "$SCRATCH_RAW_MIX/"
fi

# -------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------

cd "$HOME/eval_generation"

echo "Starting FAD and KL evaluation..."

uv run python FAD_KL.py