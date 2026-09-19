#!/bin/bash
#SBATCH --job-name=eval_audio
#SBATCH --partition=day
#SBATCH --gres=gpu:2080ti:1
#SBATCH --cpus-per-task=8
#SBATCH --error=job.eval_gen.%A.err
#SBATCH --output=job.eval_gen.%A.out
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# -------------------------------------------------------------
# Scratch Directory Setup & Cleanup Trap
# -------------------------------------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SCRATCH_BASE_DIR="/scratch/${SLURM_JOB_ID}"

cleanup() {
    echo "Triggering cleanup: clearing contents of $SCRATCH_BASE_DIR..."
    if [ -d "$SCRATCH_BASE_DIR" ]; then
        find "$SCRATCH_BASE_DIR" -mindepth 1 -delete
    fi
}

trap cleanup EXIT SIGINT SIGTERM

# -------------------------------------------------------------
# Input & Evaluation Paths
# -------------------------------------------------------------
#/home/sherkat/eval_generation/experiment_audiogen_baseline_01
export SOURCE_FOLDER="/home/sherkat/eval_generation/experiment_audioldm2_comparison_03"
export GT_FOLDER="/home/sherkat/eval_generation/test_split_GT_audios"

# Uncomment if evaluating RAW_MIX:
# export WITH_RAW_MIX="1"

# -------------------------------------------------------------
# Execution
# -------------------------------------------------------------
cd "$HOME/eval_generation"

uv run python eval_script.py