#!/bin/bash

# Comprehensive fix and run script for Med-D3CG training
# This script fixes all known issues and runs the training

set -e

echo "=========================================="
echo "Med-D3CG Training - Comprehensive Fix"
echo "=========================================="

# Setup environment
echo ""
echo "Step 1: Setting up environment..."
export CD_DIR="/bask/projects/c/chenhp-data-gen/yifansun/project/Med-D3CG"
cd "${CD_DIR}"

source /bask/projects/q/qingjiem-heart-tte/yifansun/conda/miniconda/etc/profile.d/conda.sh
conda activate medd3cg

echo "✓ Environment setup completed"
echo "  Working directory: $(pwd)"
echo "  Python: $(which python)"

# Step 2: Fix PyTorch installation
echo ""
echo "Step 2: Fixing PyTorch installation..."

# Remove corrupted installation silently
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip cache purge 2>/dev/null || true

# Reinstall with explicit CUDA 12.1 support
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "✓ PyTorch reinstalled"

# Step 3: Verify installation
echo ""
echo "Step 3: Verifying installation..."

python << 'EOF'
import torch
import sys
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0")
    x = torch.randn(10, 10).to(device)
    print(f"  Tensor to GPU: OK")
else:
    print("WARNING: CUDA not available, will use CPU")
sys.exit(0)
EOF

if [ $? -ne 0 ]; then
    echo "ERROR: PyTorch verification failed!"
    exit 1
fi

echo "✓ PyTorch verification passed"

# Step 4: Run test script
echo ""
echo "Step 4: Running setup validation tests..."

cd "${CD_DIR}"
python test_setup.py

if [ $? -ne 0 ]; then
    echo "ERROR: Setup validation failed!"
    echo "Please check the errors above and fix them manually."
    exit 1
fi

echo "✓ All validation tests passed"

# Step 5: Run training
echo ""
echo "Step 5: Starting training..."
echo "=========================================="

# Configuration
DATA_ROOT="${CD_DIR}/data/acdc_wholeheart/25022_JPGs"
OUTPUT_DIR="${CD_DIR}/results/acdc_unconditional"
MODEL_NAME="D3CG_uncond_db4"

mkdir -p "${OUTPUT_DIR}"

python unconditional_train.py \
    --model_name "${MODEL_NAME}" \
    --dataset_dir "${DATA_ROOT}" \
    --out_name "acdc_uncond_db4" \
    --batch_size 8 \
    --n_epochs 500 \
    --image_size 96 \
    --ch 64 \
    --ch_mult 1 2 3 4 \
    --num_res_blocks 2 \
    --dropout 0.1 \
    --lr 2e-5 \
    --beta_1 1e-4 \
    --beta_T 0.02 \
    --T 1000 \
    --grad_clip 1.0 \
    --save_weight_dir "${OUTPUT_DIR}" \
    --save_interval 50 \
    --val_start_epoch 100 \
    --val_num 8 \
    --wave_type db4 \
    --transform_levels 1

echo ""
echo "=========================================="
echo "Training completed successfully!"
echo "=========================================="
echo "Results directory: ${OUTPUT_DIR}"
