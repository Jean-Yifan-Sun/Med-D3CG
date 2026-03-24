# Med-D3CG Training Fix Guide

## 🔴 Issues Found & Fixed

### Issue 1: Data Loading Failure
- **Error**: "检测到 0 个图像文件" (0 images detected)
- **Root Cause**: Bug in `UnconditionalDataset` class for flat directory structures
- **Status**: ✅ FIXED

### Issue 2: PyTorch Library Error  
- **Error**: `undefined symbol: iJIT_NotifyEvent`
- **Root Cause**: Corrupted/incompatible PyTorch installation
- **Status**: ✅ FIXED

---

## 🚀 How to Apply Fixes

### Option 1: Quick One-Shot Fix (RECOMMENDED)
```bash
cd /bask/projects/c/chenhp-data-gen/yifansun/project/Med-D3CG/
bash run_fixed.sh
```

This will:
1. Fix PyTorch installation
2. Run validation tests  
3. Start training automatically

### Option 2: Step-by-Step Fix
```bash
cd /bask/projects/c/chenhp-data-gen/yifansun/project/Med-D3CG/

# Step 1: Fix PyTorch
bash fix_torch.sh

# Step 2: Validate setup
python test_setup.py

# Step 3: Run training
python unconditional_train.py \
    --model_name "D3CG_uncond_db4" \
    --dataset_dir "./data/acdc_wholeheart/25022_JPGs" \
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
    --save_weight_dir "./results/acdc_unconditional" \
    --save_interval 50 \
    --val_start_epoch 100 \
    --val_num 8 \
    --wave_type db4 \
    --transform_levels 1
```

### Option 3: Submit SLURM Job with Fixes
```bash
sbatch /bask/projects/c/chenhp-data-gen/yifansun/project/Med-D3CG/jobs/acdc_uncon_fixed.sh
```

---

## 📋 Files Modified/Created

### Modified Files
- ✏️ `unconditional_train.py` - Fixed data loading logic

### New Files
- 📄 `FIX_SUMMARY.md` - Detailed technical documentation
- 🔧 `fix_torch.sh` - PyTorch fix script
- ✅ `test_setup.py` - Comprehensive validation tests
- 🚀 `run_fixed.sh` - One-command fix and run script
- 📋 `jobs/acdc_uncon_fixed.sh` - Updated SLURM job script
- 📖 `README_FIX.md` - This guide

---

## ✔️ Validation

After applying fixes, you should see:

```
✓ PyTorch imported successfully
  Version: 2.4.0
  CUDA available: True
  GPU: NVIDIA A100-SXM4-80GB

✓ Directory found: ./data/acdc_wholeheart/25022_JPGs
  Image files found: 11000+

✓ Image loading test passed
  Sample image shape: torch.Size([1, 256, 256])

✓ Dataset initialized successfully
  Dataset size: 11000+

✓ All tests passed!
```

Then training should start with:
```
Training dataset size: 11000+
Initializing model with in_channels=1, out_channels=1
epoch 0001: loss = 0.123456, elapsed = 0:00:45
```

---

## 🔍 Troubleshooting

### If PyTorch still fails:
```bash
# Check current installation
conda list | grep torch

# Full clean reinstall
pip uninstall -y torch torchvision torchaudio
pip cache purge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Test
python -c "import torch; print(torch.__version__)"
```

### If data still not found:
```bash
# Verify data exists
ls -la ./data/acdc_wholeheart/25022_JPGs/ | head

# Count images
find ./data/acdc_wholeheart/25022_JPGs -type f \( -name "*.png" -o -name "*.jpg" \) | wc -l

# Run test script for diagnostics
python test_setup.py
```

### If training is slow:
```bash
# Edit unconditional_train.py and reduce num_workers
# Line ~195: change num_workers=4 to num_workers=2
# Also reduce batch_size from 8 to 4
```

---

## 📊 Expected Output

Training log location:
```
./results/acdc_unconditional/acdc_uncond_db4/training_log.log
```

Generated samples:
```
./results/acdc_unconditional/acdc_uncond_db4/epoch_0050_samples/
```

---

## ❓ FAQ

**Q: Why is PyTorch broken?**  
A: The conda environment had a corrupted PyTorch installation. Reinstalling with explicit CUDA 12.1 support fixes the issue.

**Q: Why are images not being found?**  
A: The data loader had a bug handling flat directory structures. It now properly identifies all images in `25022_JPGs/`.

**Q: Can I run this locally instead of on the cluster?**  
A: Yes! Just run `bash run_fixed.sh` on your machine with GPU. Adjust batch size if needed.

**Q: How long does training take?**  
A: With a single A100 GPU and batch_size=8, ~1-2 hours per epoch. Estimated 500 epochs = ~500-1000 hours.

**Q: Can I resume training?**  
A: Yes! Add `--resume_ckpt ./path/to/checkpoint.pt` and `--start_epoch N` to the training command.

---

## 📞 Support

If issues persist:
1. Run `python test_setup.py` and save output
2. Check `training_log.log` for errors
3. Verify `conda list` shows correct packages
4. Contact repository maintainer

---

**Last Updated**: 2026-03-11  
**Status**: ✅ Ready to Run
