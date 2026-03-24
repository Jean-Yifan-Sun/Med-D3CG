# Evaluation Pipeline Guide

This guide explains how to use the evaluation pipeline to compute LPIPS and FID scores for trained D3CG model checkpoints.

## Overview

The evaluation pipeline (`evaluate.py`) allows you to:

1. **Generate 2000 images** from saved checkpoints and compute **LPIPS score**
2. **Generate 50,000 images** from saved checkpoints and compute **FID score**
3. **Evaluate multiple checkpoints** from a directory automatically
4. **Save results** to JSON format for analysis

## Prerequisites

Ensure you have the required packages installed:

```bash
pip install torch torchvision lpips
```

The pipeline requires:
- `torch` and `torchvision` for model loading and image generation
- `lpips` for LPIPS score computation
- Pre-trained Inception V3 model (automatically downloaded)

## Directory Structure

```
results/
├── acdc_unconditional/
│   ├── acdc_uncond_db4/
│   │   ├── best_model_epoch_0001.pt
│   │   ├── best_model_epoch_0002.pt
│   │   └── ...
│   └── acdc_uncond_db4_new/
│       └── ...
└── [other dataset results]
```

## Usage

### Basic Usage: Evaluate Default Checkpoint Directory

```bash
python evaluate.py \
  --checkpoint_dir results/acdc_unconditional/acdc_uncond_db4 \
  --real_data_dir data/acdc_wholeheart/images \
  --output_dir eval_results/acdc_uncond_db4
```

### Parameters Explained

#### Required Parameters

- `--checkpoint_dir`: Path to directory containing `.pt` checkpoint files
  - Example: `results/acdc_unconditional/acdc_uncond_db4`

#### Optional Parameters

**Data Paths:**
- `--output_dir`: Where to save evaluation results (default: `./eval_results/`)
- `--real_data_dir`: Directory with real images for FID computation
  - LPIPS and FID scores require real data for comparison
  - If not provided, only generates samples without scores

**Generation Configuration:**
- `--num_lpips_samples`: Number of samples for LPIPS (default: 2000)
- `--num_fid_samples`: Number of samples for FID (default: 50000)
- `--batch_size`: Generation batch size (default: 32)
- `--image_size`: Generated image size (default: 256)

**Model Configuration:**
- `--model_name`: Model type (default: `D3CG_uncond_db4`)
- `--wave_type`: Wavelet type (default: `db4`)
  - Options: `haar`, `db4`, `coif3`, `bior2.2`, `dmey`
- `--transform_levels`: Number of wavelet transform levels (default: 1)
- `--T`: Diffusion timesteps (default: 1000)
- `--beta_1`: Start of beta schedule (default: 1e-4)
- `--beta_T`: End of beta schedule (default: 0.02)

**Network Architecture:**
- `--ch`: Base channels (default: 128)
- `--ch_mult`: Channel multipliers (default: [1 2 3 4])
- `--attn`: Attention layer positions (default: [2])
- `--num_res_blocks`: Residual blocks per layer (default: 2)
- `--dropout`: Dropout rate (default: 0.3)

**Evaluation Control:**
- `--skip_lpips`: Skip LPIPS evaluation
- `--skip_fid`: Skip FID evaluation
- `--device`: Device to use (default: `cuda`)

### Example Commands

#### Evaluate ACDC Checkpoints for LPIPS Only (2000 samples)

```bash
python evaluate.py \
  --checkpoint_dir results/acdc_unconditional/acdc_uncond_db4 \
  --real_data_dir data/acdc_wholeheart/images \
  --skip_fid \
  --output_dir eval_results/acdc_lpips
```

#### Evaluate with Custom Configuration

```bash
python evaluate.py \
  --checkpoint_dir results/acdc_unconditional/acdc_uncond_db4 \
  --real_data_dir data/acdc_wholeheart/images \
  --wave_type coif3 \
  --batch_size 64 \
  --num_lpips_samples 1000 \
  --num_fid_samples 25000
```

#### Quick Test (100 LPIPS samples only)

```bash
python evaluate.py \
  --checkpoint_dir results/acdc_unconditional/acdc_uncond_db4 \
  --real_data_dir data/acdc_wholeheart/images \
  --num_lpips_samples 100 \
  --skip_fid
```

## Output Structure

After evaluation, your output directory will contain:

```
eval_results/
├── evaluation_summary.json          # Summary of all results (only persistent output)
└── [checkpoint_name]/
    # Note: Generated samples are stored in temporary directories and automatically deleted after evaluation
    # This saves disk space significantly
```

### Evaluation Workflow

For each checkpoint, the pipeline executes the following steps **sequentially**:

1. **LPIPS Evaluation** (if not skipped):
   - Generate 2000 samples in temporary directory
   - Compute LPIPS score comparing to real data
   - **Automatically delete all 2000 samples** (saves ~5-10 GB)
   
2. **FID Evaluation** (if not skipped):
   - Generate 50,000 samples in temporary directory
   - Compute FID score comparing to real data distribution
   - **Automatically delete all 50,000 samples** (saves ~100-200 GB)

This sequential approach ensures:
- ✅ **Space efficiency**: No sample accumulation on disk
- ✅ **LPIPS first**: Quality metrics computed before statistical distribution
- ✅ **Progress tracking**: Clear logging of what's happening
- ⏱️ **Plan for time**: Full evaluation takes many hours (LPIPS: 1-2h, FID: 6-12h per checkpoint)

### Results Summary (evaluation_summary.json)

```json
[
  {
    "checkpoint": "/path/to/best_model_epoch_0001.pt",
    "checkpoint_name": "best_model_epoch_0001",
    "lpips_score": 0.2456,
    "lpips_std": 0.0123,
    "fid_score": 15.3421,
    "sample_counts": {
      "lpips": 2000,
      "fid": 50000
    }
  },
  ...
]
```

## Performance Notes

### Memory Requirements
- **LPIPS Generation (2000 samples)**: ~8GB VRAM for batch_size=32
- **FID Generation (50000 samples)**: ~12GB VRAM for batch_size=32
- **LPIPS Computation**: ~6GB VRAM (depends on LPIPS model)
- **FID Computation**: ~8GB VRAM (depends on Inception model)

### Timing Estimates
- **2000 samples generation**: ~30-60 minutes on V100 GPU
- **50000 samples generation**: ~12-24 hours on V100 GPU
## Performance Notes

### Memory Requirements
- **Model loading**: ~4GB VRAM
- **Sample generation**: ~8-12GB VRAM for batch_size=32
- **LPIPS computation**: ~8GB VRAM (depends on LPIPS model)
- **FID computation**: ~10GB VRAM (depends on Inception model)

**Peak memory**: ~12-15GB total (samples are generated sequentially, not all at once)

### Storage Requirements
- **LPIPS generation**: ~100-150 MB (temporary, auto-deleted)
- **FID generation**: ~2-3 GB (temporary, auto-deleted)
- **Final results**: ~1-2 MB (evaluation_summary.json only)

**Total disk needed**: Minimal! ~5-10GB working space (temporary files auto-cleaned)

### Timing Estimates (per checkpoint)

| Phase | Time (V100) | Samples | Notes |
|-------|------------|---------|-------|
| LPIPS generation | 1-2 hours | 2,000 | Single stream |
| LPIPS computation | 10-20 min | 2,000 | After generation, before deletion |
| FID generation | 6-12 hours | 50,000 | Single stream |
| FID computation | 20-40 min | 50,000 | After generation, before deletion |
| **Total per checkpoint** | **~8-15 hours** | **52,000** | Samples auto-deleted |

**For 116 checkpoints**: ~330-600 days (you'd want to run in parallel on multiple GPUs)

### To speed up evaluation

**Reduce sample counts:**
```bash
python evaluate.py \
  --num_lpips_samples 500 \
  --num_fid_samples 5000 \
  ...
```

**Accelerated sampling (256 instead of 1000 timesteps):**
```bash
python evaluate.py \
  --sampling_timesteps 256 \
  --num_lpips_samples 2000 \
  --num_fid_samples 50000 \
  ...
```
(~4x faster generation with slightly lower quality)

**Skip one metric:**
```bash
python evaluate.py ... --skip_fid  # Only LPIPS
# OR
python evaluate.py ... --skip_lpips  # Only FID
```

To reduce VRAM usage, decrease `--batch_size`:
```bash
python evaluate.py ... --batch_size 16
```

## Troubleshooting

### "Real data directory not found"
- Ensure `--real_data_dir` points to correct directory with PNG/JPG images
- LPIPS and FID require real images for comparison
- Without real images, only generates samples (no scores computed)

### Out of Memory (OOM)
- Reduce `--batch_size` (default: 32, try: 16, 8, or 4)
- Reduce `--num_lpips_samples` or `--num_fid_samples`
- Generate samples separately and compute scores with other tools

### Checkpoint not found
- Verify checkpoint path exists and contains `.pt` files
- Check the exact filename and directory structure

### Slow generation
- Check GPU usage with `nvidia-smi`
- Consider using multiple GPUs with modified code
- Increase `--batch_size` if GPU has enough memory

## Understanding Metrics

### LPIPS (Learned Perceptual Image Patch Similarity)
- **Range**: 0 to 1 (lower is better)
- **Interpretation**: Perceptual distance between generated and reference images
- **0**: Identical images (perceptually)
- **1**: Very different images
- **Typical for good models**: 0.1-0.3

### FID (Fréchet Inception Distance)
- **Range**: 0 to ∞ (lower is better)
- **Interpretation**: Distance between distribution of generated vs real images
- **0**: Distributions are identical
- **Higher values**: More distributional mismatch
- **Typical for good medical image models**: 10-50

## Advanced Usage

### Evaluate Multiple Directories Sequentially

```bash
#!/bin/bash
for dir in results/acdc_unconditional/*/; do
  python evaluate.py \
    --checkpoint_dir "$dir" \
    --real_data_dir data/acdc_wholeheart/images \
    --output_dir eval_results/$(basename "$dir")
done
```

### Skip Checkpoint if Samples Already Generated

To reuse previously generated samples, you can modify the pipeline to check for existing samples before regenerating.

### Batch Processing Script

Create `batch_evaluate.sh`:

```bash
#!/bin/bash

DATASETS=("acdc_uncond_db4" "acdc_uncond_db4_new")
REAL_DATA="data/acdc_wholeheart/images"

for dataset in "${DATASETS[@]}"; do
  echo "Evaluating $dataset..."
  python evaluate.py \
    --checkpoint_dir "results/acdc_unconditional/$dataset" \
    --real_data_dir "$REAL_DATA" \
    --output_dir "eval_results/$dataset" \
    --batch_size 64
done
```

Then run:
```bash
chmod +x batch_evaluate.sh
./batch_evaluate.sh
```

## Integration with Existing Code

The evaluation pipeline uses the same components as your training pipeline:

- **Models**: Uses `models/wavelet.py` WTUNet
- **Diffusion**: Uses `diffusion/model_factory.py` get_trainer_sampler
- **Utilities**: Uses existing `utils/is_lpips.py` and `utils/fid_score.py`

This ensures consistency between training and evaluation.

## FAQ

**Q: Can I evaluate without real images?**
A: Yes, the pipeline will generate samples but won't compute LPIPS/FID scores without real images for comparison.

**Q: Can I evaluate custom model architectures?**
A: Yes, adjust the model parameters (`--ch`, `--ch_mult`, `--num_res_blocks`, etc.) to match your training configuration.

**Q: How do I compare results across different checkpoints?**
A: Results are saved in `evaluation_summary.json`. You can parse this file to create comparison tables and charts.

**Q: Can I resume evaluation if it fails?**
A: The pipeline checks for existing samples but currently regenerates all. To skip, manually remove the output directory or checkpoint subdirectories.

## Support

For issues or questions:
1. Check the logs for error messages
2. Verify all paths and files exist
3. Ensure checkpoint format is compatible
4. Check GPU memory with `nvidia-smi`
