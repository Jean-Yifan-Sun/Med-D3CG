# Accelerated Sampling Guide

## Overview

Your models were trained with 1000 diffusion timesteps, but you can use **accelerated sampling** to generate images faster using fewer timesteps (e.g., 256, 100, or 50) during inference.

This doesn't require retraining - the model automatically adapts to fewer sampling steps while maintaining decent quality!

## How It Works

- **Training**: Model learned to denoise over 1000 steps
- **Inference**: You can skip some denoising steps and sample faster
- **Trade-off**: Fewer steps = faster generation, but potentially lower quality
- **Method**: The sampler linearly interpolates between timesteps

## Usage

### Option 1: Using evaluate.py

Generate samples with 256 timesteps instead of 1000:

```bash
python evaluate.py \
  --checkpoint_dir results/acdc_unconditional/acdc_uncond_db4 \
  --real_data_dir data/acdc_wholeheart/images \
  --sampling_timesteps 256 \
  --num_lpips_samples 100 \
  --num_fid_samples 100
```

### Option 2: Using example_evaluation.py

Edit the script and change:

```python
SAMPLING_TIMESTEPS = 256  # Change from None to use accelerated sampling
```

Then run:

```bash
python example_evaluation.py --example full
```

### Option 3: Using unconditional_generation.py

```bash
python unconditional_generation.py \
  --checkpoint_dir results/acdc_unconditional/acdc_uncond_db4 \
  --sampling_timesteps 256 \
  --num_samples 100
```

## Recommended Timestep Values

| Timesteps | Speed    | Quality | Use Case |
|-----------|----------|---------|----------|
| 1000      | Baseline | Highest | Final evaluation, high quality |
| 500       | 2x faster| Excellent | Good balance |
| 256       | 4x faster| Very good | Default accelerated sampling |
| 100       | 10x faster| Good | Quick testing |
| 50        | 20x faster| Fair | Fast prototyping |

## Comparing Quality

To compare quality at different timesteps:

```bash
# Full 1000 steps
python evaluate.py \
  --checkpoint_dir results/acdc_unconditional/acdc_uncond_db4 \
  --real_data_dir data/acdc_wholeheart/images \
  --num_lpips_samples 100 \
  --output_dir eval_results/T1000

# Accelerated 256 steps
python evaluate.py \
  --checkpoint_dir results/acdc_unconditional/acdc_uncond_db4 \
  --real_data_dir data/acdc_wholeheart/images \
  --sampling_timesteps 256 \
  --num_lpips_samples 100 \
  --output_dir eval_results/T256
```

Then compare the LPIPS and FID scores in:
- `eval_results/T1000/evaluation_summary.json`
- `eval_results/T256/evaluation_summary.json`

## Important Notes

1. **Always use T=1000** - The model was trained with 1000 steps, so the `--T` parameter must always be 1000
2. **Only change sampling_timesteps** - This tells the sampler to skip steps during generation
3. **Checkpoint loading is unchanged** - The model architecture stays the same, only inference sampling changes
4. **No retraining needed** - This works with existing checkpoints without modification

## Troubleshooting

**Q: My samples look weird with 256 steps**
- A: Lower timesteps can produce lower quality. Try 500 steps instead.

**Q: What's the speed improvement?**
- A: ~4x faster with 256 steps (1000/256 ≈ 4), but actual timing depends on GPU.

**Q: Can I use less than 100 timesteps?**
- A: Yes, but quality may degrade. Test on a few samples first.

**Q: How do I know the best timestep value?**
- A: Use the evaluation pipeline to compute LPIPS/FID at different values and find the sweet spot for your use case.

## References

- DDPM: Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- DDIM: Denoising Diffusion Implicit Models (Song et al., 2021) - accelerated sampling
