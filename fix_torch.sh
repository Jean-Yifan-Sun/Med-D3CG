#!/bin/bash

# Fix PyTorch import error: undefined symbol: iJIT_NotifyEvent
# This error occurs when the PyTorch/MKLDNN library is corrupted or incompatible

echo "=========================================="
echo "Fixing PyTorch Installation"
echo "=========================================="

# Activate the conda environment
source /bask/projects/q/qingjiem-heart-tte/yifansun/conda/miniconda/etc/profile.d/conda.sh
conda activate medd3cg

echo "Current Python executable: $(which python)"
echo "Current torch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"

# Remove the problematic torch package
echo "Removing current torch installation..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Clear pip cache
pip cache purge

# Reinstall PyTorch with a clean install
# Using the CUDA 12.1 version compatible with A100
echo "Installing PyTorch with CUDA support..."
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')"

if [ $? -ne 0 ]; then
    echo "ERROR: PyTorch verification failed!"
    exit 1
fi

# Also verify torchvision
python -c "import torchvision; print(f'TorchVision version: {torchvision.__version__}')"

echo "=========================================="
echo "PyTorch installation fixed successfully!"
echo "=========================================="
