# Evaluation Pipeline - Setup Summary

## Overview

I've built a complete evaluation pipeline for your D3CG medical image generation model. The pipeline can:

1. **Generate 2000 images** and compute **LPIPS scores** (perceptual quality)
2. **Generate 50000 images** and compute **FID scores** (distribution quality)
3. **Evaluate multiple checkpoints** automatically
4. **Save results** in JSON format for analysis

## Files Created

### 1. **`evaluate.py`** - Main Evaluation Pipeline
The comprehensive evaluation script that:
- Loads trained model checkpoints
- Generates specified number of samples
- Computes LPIPS and FID scores
- Evaluates multiple checkpoints in a directory
- Saves results to JSON

**Usage:**
```bash
python evaluate.py \
  --checkpoint_dir results/acdc_unconditional/acdc_uncond_db4 \
  --real_data_dir data/acdc_wholeheart/images \
  --output_dir eval_results/acdc_uncond_db4
```

### 2. **`utils/inception.py`** - Inception V3 Model
Implements the InceptionV3 network required by the FID score computation.
- Auto-compatible with PyTorch
- Handles both newer and older torchvision versions
- Used internally by `fid_score.py`

### 3. **`example_evaluation.py`** - Simple Usage Examples
Demonstrates how to:
- Generate a single sample (quick test)
- Run full evaluation with both metrics (detailed walkthrough)
- Use individual components

**Usage:**
```bash
# Generate one sample (quick test)
python example_evaluation.py --example single

# Run full evaluation
python example_evaluation.py --example full
```

### 4. **`run_evaluation.sh`** - Quick Start Script
Bash script with example evaluation command ready to execute.

### 5. **`EVALUATION_GUIDE.md`** - Complete Documentation
Comprehensive guide covering:
- Parameter explanations
- Example commands for different scenarios
- Output structure and results interpretation
- Performance notes and memory requirements
- Troubleshooting and FAQs

## Quick Start

### Step 1: Check Your Data
```bash
# Verify checkpoint directory exists
ls results/acdc_unconditional/acdc_uncond_db4/

# Verify real images exist
ls data/acdc_wholeheart/images/ | head -5
```

### Step 2: Run Evaluation (Small Test)
```bash
python evaluate.py \
  --checkpoint_dir results/acdc_unconditional/acdc_uncond_db4 \
  --real_data_dir data/acdc_wholeheart/images \
  --num_lpips_samples 100 \
  --num_fid_samples 100 \
  --batch_size 32
```

### Step 3: Run Full Evaluation (Production)
```bash
python evaluate.py \
  --checkpoint_dir results/acdc_unconditional/acdc_uncond_db4 \
  --real_data_dir data/acdc_wholeheart/images \
  --num_lpips_samples 2000 \
  --num_fid_samples 50000 \
  --batch_size 32 \
  --output_dir eval_results/acdc_uncond_db4
```

## Output Structure

After evaluation, results are organized as:

```
eval_results/
├── evaluation_summary.json          # Summary of all metrics
└── best_model_epoch_0001/
    ├── lpips_samples/               # 2000 generated images
    │   ├── sample_00000.png
    │   └── ...
    └── fid_samples/                 # 50000 generated images
        ├── sample_00000.png
        └── ...
```

**evaluation_summary.json** contains:
```json
[
  {
    "checkpoint": "path/to/checkpoint.pt",
    "checkpoint_name": "best_model_epoch_0001",
    "lpips_score": 0.2456,
    "lpips_std": 0.0123,
    "fid_score": 15.3421,
    "sample_counts": {
      "lpips": 2000,
      "fid": 50000
    }
  }
]
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint_dir` | Required | Directory with `.pt` checkpoints |
| `--real_data_dir` | None | Directory with real reference images |
| `--output_dir` | `./eval_results/` | Where to save results |
| `--num_lpips_samples` | 2000 | Number of samples for LPIPS |
| `--num_fid_samples` | 50000 | Number of samples for FID |
| `--batch_size` | 32 | Generation batch size |
| `--device` | cuda | Device to use (cuda/cpu) |

## Common Use Cases

### Evaluate All Checkpoints in Directory
```bash
python evaluate.py \
  --checkpoint_dir results/acdc_unconditional/acdc_uncond_db4 \
  --real_data_dir data/acdc_wholeheart/images
```

### LPIPS Only (1000 samples)
```bash
python evaluate.py \
  --checkpoint_dir results/acdc_unconditional/acdc_uncond_db4 \
  --real_data_dir data/acdc_wholeheart/images \
  --num_lpips_samples 1000 \
  --skip_fid
```

### Custom Model Configuration
```bash
python evaluate.py \
  --checkpoint_dir results/acdc_unconditional/acdc_uncond_db4 \
  --real_data_dir data/acdc_wholeheart/images \
  --wave_type coif3 \
  --transform_levels 2 \
  --ch 256
```

### Reduce Memory Usage
```bash
python evaluate.py \
  --checkpoint_dir results/acdc_unconditional/acdc_uncond_db4 \
  --real_data_dir data/acdc_wholeheart/images \
  --batch_size 16  # Lower batch size
```

## Performance Expectations

### Speed (on NVIDIA V100)
- **2000 LPIPS samples**: ~30-60 minutes
- **50000 FID samples**: ~12-24 hours
- **LPIPS computation**: ~10-20 minutes
- **FID computation**: ~20-40 minutes

### Memory
- **Sample generation**: 8-12GB VRAM
- **LPIPS computation**: 6GB VRAM
- **FID computation**: 8GB VRAM

**Tips to reduce memory:**
- Decrease `--batch_size` (default 32)
- Decrease sample counts
- Run on CPU (much slower)

## Understanding the Metrics

### LPIPS (Learned Perceptual Image Patch Similarity)
- **Range**: 0 to 1 (lower = better)
- **Measures**: Perceptual similarity between generated and reference images
- **Typical values**: 0.1-0.3 for good models
- **What it means**: How much the generated images "look like" reference images

### FID (Fréchet Inception Distance)
- **Range**: 0 to ∞ (lower = better)
- **Measures**: Statistical difference between generated and real image distributions
- **Typical values**: 10-50 for good medical image models
- **What it means**: How well the generated distribution matches real distribution

## Troubleshooting

**Problem**: "No checkpoint files found"
- **Solution**: Verify checkpoint directory has `.pt` files
- Check path with: `ls results/acdc_unconditional/acdc_uncond_db4/*.pt`

**Problem**: "Real data directory not found"
- **Solution**: Verify real data path exists
- Check with: `ls data/acdc_wholeheart/images/ | head -5`

**Problem**: "Out of memory"
- **Solution**: Reduce batch size or sample counts
- Example: `--batch_size 16 --num_lpips_samples 500`

**Problem**: "Slow generation"
- **Solution**: Check GPU with `nvidia-smi`
- Increase batch size if memory available
- Run with multiple GPUs (requires code modification)

## Integration with Your Setup

The pipeline uses components already in your project:
- ✓ `models/wavelet.py` - WTUNet model
- ✓ `diffusion/model_factory.py` - Trainer and sampler factory
- ✓ `utils/is_lpips.py` - LPIPS computation
- ✓ `utils/fid_score.py` - FID computation
- ✓ `utils/inception.py` - NEW: Inception V3 model

All dependencies are in `requirements.txt` (already installed):
- torch, torchvision
- lpips
- numpy, Pillow
- scikit-image, scikit-learn

## Advanced Usage

### Batch Evaluate Multiple Runs
```bash
#!/bin/bash
for checkpoint_dir in results/acdc_unconditional/*; do
  python evaluate.py \
    --checkpoint_dir "$checkpoint_dir" \
    --real_data_dir data/acdc_wholeheart/images \
    --output_dir eval_results/$(basename "$checkpoint_dir")
done
```

### Generate Only (No Metrics)
If you just want to generate samples without computing metrics:
```bash
python evaluate.py \
  --checkpoint_dir results/acdc_unconditional/acdc_uncond_db4 \
  --skip_lpips \
  --skip_fid
```

### Inspect Results
```bash
# View summary
cat eval_results/evaluation_summary.json | python -m json.tool

# Count generated samples
ls eval_results/best_model_epoch_0001/lpips_samples/ | wc -l
ls eval_results/best_model_epoch_0001/fid_samples/ | wc -l
```

## Next Steps

1. **Test with single checkpoint**:
   ```bash
   python example_evaluation.py --example full --checkpoint [path_to_checkpoint]
   ```

2. **Evaluate all checkpoints**:
   ```bash
   python evaluate.py --checkpoint_dir results/acdc_unconditional/acdc_uncond_db4 \
     --real_data_dir data/acdc_wholeheart/images
   ```

3. **Analyze results**:
   - View `eval_results/evaluation_summary.json`
   - Compare LPIPS/FID scores across checkpoints
   - Inspect generated samples visually

4. **Optimize parameters** based on results:
   - Adjust model architecture if needed
   - Modify diffusion parameters
   - Train for more epochs

## Support

For detailed information, refer to:
- `EVALUATION_GUIDE.md` - Comprehensive parameter documentation
- `example_evaluation.py` - Detailed code examples
- `evaluate.py` - Source code with docstrings
