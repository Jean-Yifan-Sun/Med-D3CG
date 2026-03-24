"""
Evaluation pipeline for D3CG models

This script evaluates trained checkpoints by computing:
- LPIPS score (on 2000 generated images)
- FID score (on 50000 generated images)
"""

import torch
import torch.nn as nn
import numpy as np
import os
import glob
import shutil
import tempfile
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
import argparse
import logging
import warnings

from models.wavelet import WTUNet
from diffusion.model_factory import get_trainer_sampler
from utils.is_lpips import calculate_lpips_score
from utils.fid_score import calculate_fid_given_paths, save_statistics_of_path
from utils.inception import InceptionV3

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate D3CG models")
    
    # Checkpoint and output directories
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing model checkpoints")
    parser.add_argument("--output_dir", type=str, default="./eval_results/",
                        help="Directory to save evaluation results")
    parser.add_argument("--real_data_dir", type=str, default=None,
                        help="Directory with real images for FID (optional)")
    
    # Generation parameters
    parser.add_argument("--num_lpips_samples", type=int, default=2000,
                        help="Number of samples for LPIPS evaluation")
    parser.add_argument("--num_fid_samples", type=int, default=50000,
                        help="Number of samples for FID evaluation")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for generation")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Generated image size")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="D3CG_uncond_db4",
                        help="Model name")
    parser.add_argument("--eval_interval", type=int, default=5,
                        help="Evaluate every N epochs (e.g., 5, 10)")
    parser.add_argument("--wave_type", type=str, default="db4",
                        choices=["haar", "db4", "coif3", "bior2.2", "dmey"])
    parser.add_argument("--transform_levels", type=int, default=1)
    
    # Diffusion parameters
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--sampling_timesteps", type=int, default=None,
                        help="Number of timesteps for accelerated sampling (None = use all T steps)")
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    
    # Network architecture
    parser.add_argument("--ch", type=int, default=128)
    parser.add_argument("--ch_mult", nargs='+', type=int, default=[1, 2, 3, 4])
    parser.add_argument("--attn", nargs='+', type=int, default=[2])
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    
    # Evaluation flags
    parser.add_argument("--skip_lpips", action="store_true",
                        help="Skip LPIPS evaluation")
    parser.add_argument("--skip_fid", action="store_true",
                        help="Skip FID evaluation")
    parser.add_argument("--no_temp_dir_lpips", action="store_true",
                        help="Do not use temporary directory for samples (keep them in output_dir)")
    parser.add_argument("--no_temp_dir_fid", action="store_true",
                        help="Do not use temporary directory for samples (keep them in output_dir)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--fid_stats", type=str, default=None,
                        help="Path to precomputed FID statistics (optional)")
    
    return parser.parse_args()


def tensor_to_pil(tensor, img_sz=96):
    """Convert a batch of tensors to a list of PIL Images (grayscale)"""
    # Denormalize: [-1, 1] -> [0, 1]
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    
    # Handle batch dimension
    if tensor.dim() == 4:
        assert tensor.size(-1) == img_sz and tensor.size(-2) == img_sz, f"Expected tensor shape [B, C, {img_sz}, {img_sz}], got {tensor.shape}"
        # Expecting [B, 1, H, W] or [B, C, H, W]
        # If it's wavelet coefficients with 4 channels, we might only want the first channel (LL)
        # or it should have been reconstructed already. Assuming it's reconstructed [B, 1, H, W]
        if tensor.size(1) > 1:
            tensor = tensor[:, :1, :, :] # Take only first channel if multiple
        
        images = []
        for i in range(tensor.size(0)):
            img_tensor = tensor[i].squeeze()
            img_tensor = (img_tensor * 255).byte()
            images.append(Image.fromarray(img_tensor.cpu().numpy(), mode='L'))
        return images
    else:
        # Single image case
        assert tensor.size(-1) == img_sz and tensor.size(-2) == img_sz, f"Expected tensor shape [C, {img_sz}, {img_sz}], got {tensor.shape}"
        tensor = tensor.squeeze()
        tensor = (tensor * 255).byte()
        return Image.fromarray(tensor.cpu().numpy(), mode='L')


def filter_checkpoint_keys(checkpoint):
    """Remove profiling keys from checkpoint dict"""
    return {
        k: v for k, v in checkpoint.items() 
        if 'total_ops' not in k and 'total_params' not in k
    }


def generate_samples(
    checkpoint_path,
    output_dir,
    num_samples,
    batch_size,
    args,
    device
):
    """Generate samples from a checkpoint using batches for speed"""
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    # Initialize model
    net_model = WTUNet(
        T=args.T,
        ch=args.ch,
        ch_mult=args.ch_mult,
        attn=args.attn,
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        in_channels=1,
        out_channels=1
    ).to(device)
    
    # Load weights
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    filtered_checkpoint = filter_checkpoint_keys(checkpoint)
    
    net_model.load_state_dict(filtered_checkpoint, strict=False)
    net_model.eval()
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    # Get sampler
    _, sampler = get_trainer_sampler(
        model_name=args.model_name,
        net_model=net_model,
        beta_1=args.beta_1,
        beta_T=args.beta_T,
        T=args.T,
        device=device,
        wave_type=args.wave_type,
        transform_levels=args.transform_levels,
        sampling_timesteps=args.sampling_timesteps
    )
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate samples
    logger.info(f"Generating {num_samples} samples with batch_size {batch_size}...")
    sample_paths = []
    generated_count = 0
    
    with torch.no_grad():
        pbar = tqdm(total=num_samples, desc="Generating samples")
        while generated_count < num_samples:
            # Calculate current batch size
            current_batch_size = min(batch_size, num_samples - generated_count)
            
            # Generate batch
            generated = sampler(
                batch_size=current_batch_size,
                channels=4,  # Wavelet coefficients for grayscale (LL, LH, HL, HH)
                image_size=args.image_size,
                device=device
            )
            
            # Convert batch to PIL images
            imgs = tensor_to_pil(generated, img_sz=args.image_size)
            
            # Handle both list and single image return from tensor_to_pil
            if not isinstance(imgs, list):
                imgs = [imgs]
                
            # Save images in batch
            for img in imgs:
                output_path = os.path.join(output_dir, f"sample_{generated_count:05d}.png")
                img.save(output_path)
                sample_paths.append(output_path)
                generated_count += 1
                pbar.update(1)
        pbar.close()
    
    logger.info(f"Generated {len(sample_paths)} samples in {output_dir}")
    return sample_paths


def evaluate_checkpoint(checkpoint_path, args, device):
    """Evaluate a single checkpoint"""
    
    checkpoint_name = Path(checkpoint_path).stem
    eval_output_dir = os.path.join(args.output_dir, checkpoint_name)
    
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
    
    results = {
        'checkpoint': checkpoint_path,
        'checkpoint_name': checkpoint_name,
        'lpips_score': None,
        'lpips_std': None,
        'fid_score': None,
        'sample_counts': {
            'lpips': args.num_lpips_samples,
            'fid': args.num_fid_samples
        }
    }
    
    # Generate and evaluate samples for LPIPS first
    if not args.skip_lpips:
        logger.info(f"\n=== LPIPS Evaluation for {checkpoint_name} ===")
        
        lpips_sample_dir = os.path.join(eval_output_dir, "lpips_samples")
        
        # Context manager for directory handling
        class DirHandler:
            def __init__(self, path, use_temp):
                self.path = path
                self.use_temp = use_temp
                self.tmp_dir_obj = None

            def __enter__(self):
                if self.use_temp:
                    self.tmp_dir_obj = tempfile.TemporaryDirectory(prefix="lpips_")
                    return self.tmp_dir_obj.name
                else:
                    os.makedirs(self.path, exist_ok=True)
                    return self.path

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.tmp_dir_obj:
                    self.tmp_dir_obj.cleanup()

        with DirHandler(lpips_sample_dir, not args.no_temp_dir_lpips) as lpips_dir:
            logger.info(f"Using directory: {lpips_dir}")
            
            lpips_samples = generate_samples(
                checkpoint_path,
                lpips_dir,
                args.num_lpips_samples,
                args.batch_size,
                args,
                device
            )
            
            # Compute LPIPS
            if args.real_data_dir and os.path.exists(args.real_data_dir):
                logger.info(f"Computing LPIPS score...")
                try:
                    lpips_score = calculate_lpips_score(
                        lpips_dir,
                        args.real_data_dir,
                        device,
                        batch_size=args.batch_size
                    )
                    results['lpips_score'] = float(lpips_score)
                    logger.info(f"LPIPS Score: {lpips_score:.4f}")
                except Exception as e:
                    logger.warning(f"Failed to compute LPIPS: {e}")
            else:
                logger.warning(f"Real data directory not provided or not found. Skipping LPIPS computation.")
            
            if not args.no_temp_dir_lpips:
                logger.info(f"LPIPS samples deleted (temp dir: {lpips_dir})")
            else:
                logger.info(f"LPIPS samples kept in: {lpips_dir}")
    
    # Generate and evaluate samples for FID
    if not args.skip_fid:
        logger.info(f"\n=== FID Evaluation for {checkpoint_name} ===")
        
        fid_sample_dir = os.path.join(eval_output_dir, "fid_samples")
        
        # Context manager for directory handling (reused logic)
        class FidDirHandler:
            def __init__(self, path, use_temp):
                self.path = path
                self.use_temp = use_temp
                self.tmp_dir_obj = None

            def __enter__(self):
                if self.use_temp:
                    self.tmp_dir_obj = tempfile.TemporaryDirectory(prefix="fid_")
                    return self.tmp_dir_obj.name
                else:
                    os.makedirs(self.path, exist_ok=True)
                    return self.path

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.tmp_dir_obj:
                    self.tmp_dir_obj.cleanup()

        with FidDirHandler(fid_sample_dir, not args.no_temp_dir_fid) as fid_dir:
            logger.info(f"Using directory: {fid_dir}")
            
            fid_samples = generate_samples(
                checkpoint_path,
                fid_dir,
                args.num_fid_samples,
                args.batch_size,
                args,
                device
            )
            
            # Compute FID
            if args.real_data_dir and os.path.exists(args.real_data_dir):
                logger.info(f"Computing FID score...")
                try:
                    if args.fid_stats is not None:
                        fid_score = calculate_fid_given_paths(
                            [args.fid_stats , fid_dir],
                            device=device,
                            batch_size=args.batch_size
                        )
                    else:
                        fid_score = calculate_fid_given_paths(
                            [args.real_data_dir , fid_dir],
                            device=device,
                            batch_size=args.batch_size
                        )
                    results['fid_score'] = float(fid_score)
                    logger.info(f"FID Score: {fid_score:.4f}")
                except Exception as e:
                    logger.warning(f"Failed to compute FID: {e}")
            else:
                logger.warning(f"Real data directory not provided or not found. Skipping FID computation.")
            
            if not args.no_temp_dir_fid:
                logger.info(f"FID samples deleted (temp dir: {fid_dir})")
            else:
                logger.info(f"FID samples kept in: {fid_dir}")
    
    return results


def find_checkpoints(checkpoint_dir, args):
    """Find specific epoch checkpoints (e.g., model_epoch_0005.pt) and filter by interval"""
    import re
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Pattern to match model_epoch_XXXX.pt
    pattern = re.compile(r"model_epoch_(\d+)\.pt")
    
    all_files = sorted(checkpoint_dir.glob("*.pt"))
    filtered_checkpoints = []
    
    for ckpt in all_files:
        filename = ckpt.name
        # Skip "best" models
        if "best" in filename.lower():
            continue
            
        match = pattern.search(filename)
        if match:
            epoch_num = int(match.group(1))
            if epoch_num % args.eval_interval == 0:
                filtered_checkpoints.append(str(ckpt))
    
    if not filtered_checkpoints:
        logger.warning(f"No checkpoints matching 'model_epoch_XXXX.pt' with interval {args.eval_interval} found in {checkpoint_dir}")
    
    return filtered_checkpoints


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Find checkpoints
    checkpoints = find_checkpoints(args.checkpoint_dir, args)
    
    if not checkpoints:
        logger.error(f"No checkpoints found in {args.checkpoint_dir}")
        return
    
    logger.info(f"Found {len(checkpoints)} checkpoint(s)")
    
    # Evaluate each checkpoint
    all_results = []
    
    for checkpoint_path in checkpoints:
        try:
            logger.info(f"\nEvaluating checkpoint: {checkpoint_path}")
            results = evaluate_checkpoint(checkpoint_path, args, device)
            all_results.append(results)
        except Exception as e:
            logger.error(f"Failed to evaluate {checkpoint_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results summary
    summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nEvaluation summary saved to {summary_path}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    for result in all_results:
        logger.info(f"\nCheckpoint: {result['checkpoint_name']}")
        if result['lpips_score'] is not None:
            logger.info(f"  LPIPS: {result['lpips_score']:.4f}")
        if result['fid_score'] is not None:
            logger.info(f"  FID:   {result['fid_score']:.4f}")


if __name__ == '__main__':
    main()
