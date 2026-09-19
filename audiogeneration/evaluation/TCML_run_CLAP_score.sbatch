#!/bin/bash
#SBATCH ...

echo "Preparing scratch folders"

LOCAL_GT_DIR="/path/to/local/gt"
LOCAL_GEN_DIR="/path/to/local/gen"

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