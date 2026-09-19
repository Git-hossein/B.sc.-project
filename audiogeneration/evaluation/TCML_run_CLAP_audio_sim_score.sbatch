#!/bin/bash
#SBATCH --job-name=CLAP_sim_audio
#SBATCH --partition=day
#SBATCH --gres=gpu:2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --error=job.clap_gen.%A.err
#SBATCH --output=job.clap_gen.%A.out
#SBATCH --mem=16G
#SBATCH --time=02:00:00
echo "Preparing scratch folders"

LOCAL_GT_DIR="/home/sherkat/eval_generation/test_split_GT_audios"
LOCAL_GEN_DIR="/home/sherkat/eval_generation/experiment_audioldm2_ablation_03"

SCRATCH_BASE_DIR="/scratch/${SLURM_JOB_ID}"

export RAW_MIX="1"

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

if [[ -n "$RAW_MIX" ]]; then
    mkdir -p "$SCRATCH_RAW_MIX"

    echo "Copying RAW_MIX files..."
    cp "$LOCAL_GEN_DIR"/*_RAW_MIX.wav "$SCRATCH_RAW_MIX/"
fi

echo "Scratch preparation complete."


cd "$HOME/eval_generation"

echo "starting CLAP similarity computions..."

uv run python CLAP_audio_similarity.py