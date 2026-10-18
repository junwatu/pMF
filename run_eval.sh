#!/bin/bash
# Self-contained evaluation script for HF Jobs
# Downloads checkpoint, installs deps, runs evaluation

set -e

MODEL=$1
WORKDIR=$2
NUM_IMAGES=${3:-50000}
GEN_BSZ=${4:-32}

echo "=== pMF Evaluation ==="
echo "Model: $MODEL"
echo "Workdir: $WORKDIR"
echo "Num images: $NUM_IMAGES"
echo "Gen batch size: $GEN_BSZ"

# Install dependencies
echo "=== Installing dependencies ==="
pip install torch opencv-python torch-fidelity >/dev/null 2>&1

# Clone the PyTorch repo
echo "=== Cloning pMF-torch repo ==="
cd /workspace
git clone --depth 1 --branch torch https://github.com/Lyy-iiis/pMF.git pMF 2>/dev/null || true
cd pMF

# Download checkpoint
echo "=== Downloading checkpoint ==="
CKPT_URL="https://huggingface.co/Lyy0725/pMF/resolve/main/${MODEL}.pt"
mkdir -p checkpoints
curl -L -o "checkpoints/${MODEL}.pt" "$CKPT_URL"

# Model-specific parameters
declare -A CFG_OMEGA=(
    ["pmfDiT_B_16"]=7.5 ["pmfDiT_L_16"]=7.0 ["pmfDiT_H_16"]=7.0
    ["pmfDiT_B_32"]=6.5 ["pmfDiT_L_32"]=7.5 ["pmfDiT_H_32"]=5.5
)
declare -A INTERVAL_MIN=(
    ["pmfDiT_B_16"]=0.1 ["pmfDiT_L_16"]=0.2 ["pmfDiT_H_16"]=0.2
    ["pmfDiT_B_32"]=0.1 ["pmfDiT_L_32"]=0.2 ["pmfDiT_H_32"]=0.1
)
declare -A INTERVAL_MAX=(
    ["pmfDiT_B_16"]=0.8 ["pmfDiT_L_16"]=0.7 ["pmfDiT_H_16"]=0.6
    ["pmfDiT_B_32"]=0.7 ["pmfDiT_L_32"]=0.6 ["pmfDiT_H_32"]=0.6
)

echo "=== Running evaluation ==="
python evaluate.py evaluate \
    --ckpt-path "checkpoints/${MODEL}.pt" \
    --workdir "$WORKDIR" \
    --model "$MODEL" \
    --cfg-omega "${CFG_OMEGA[$MODEL]}" \
    --interval-min "${INTERVAL_MIN[$MODEL]}" \
    --interval-max "${INTERVAL_MAX[$MODEL]}" \
    --num-images "$NUM_IMAGES" \
    --gen-bsz "$GEN_BSZ"

echo "=== Evaluation complete ==="
