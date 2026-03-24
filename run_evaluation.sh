#!/bin/bash
# Quick evaluation script - adjust paths and parameters as needed

# Example 1: Evaluate ACDC checkpoints (2000 LPIPS + 50000 FID)
echo "Starting evaluation pipeline..."

python evaluate.py \
  --checkpoint_dir results/acdc_unconditional/acdc_uncond_db4 \
  --real_data_dir data/acdc_wholeheart/images \
  --output_dir eval_results/acdc_uncond_db4 \
  --num_lpips_samples 2000 \
  --num_fid_samples 50000 \
  --batch_size 32 \
  --device cuda

echo "Evaluation complete! Results saved to eval_results/"
echo "Check eval_results/evaluation_summary.json for summary"
