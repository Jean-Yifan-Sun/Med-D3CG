"""
无条件生成脚本

该脚本使用训练好的无条件D3CG模型生成医学图像。
可以生成单个样本或批量样本。
"""

import torch
import numpy as np
import os
from models.base import BaseUNet
from models.attention import MidAttnUNet, UpAttnUNet
from models.wavelet import WTUNet
from diffusion.model_factory import get_trainer_sampler, get_available_models
from PIL import Image
import warnings
import argparse

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Unconditional D3CG Generation")
    
    # Model configuration
    available_models = get_available_models()["d3cg_uncond_models"] + ["D3CG_custom_uncond"]
    parser.add_argument("--model_name", type=str, default="D3CG_uncond_db4",
                        choices=available_models,
                        help="Model name for generation")
    
    # Unconditional D3CG configuration
    parser.add_argument("--wave_type", type=str, default="db4",
                        choices=["haar", "db4", "coif3", "bior2.2", "dmey"],
                        help="Wavelet type for D3CG models")
    parser.add_argument("--transform_levels", type=int, default=1,
                        help="Number of wavelet transform levels")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./results/generated_images/")
    parser.add_argument("--model_weight_path", type=str, 
                        default="./results/unconditional/D3CG_uncond/best_model_epoch_0100.pt",
                        help="Path to trained model weights")
    
    # Model parameters
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    
    # Channel configuration
    parser.add_argument("--is_rgb", action="store_true", default=False,
                        help="Whether to generate RGB images (default: grayscale)")
    
    # Generation parameters
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to generate")
    parser.add_argument("--ch", type=int, default=128)
    parser.add_argument("--ch_mult", nargs='+', type=int, default=[1, 2, 3, 4])
    parser.add_argument("--attn", nargs='+', type=int, default=[2])
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--sampling_timesteps", type=int, default=None,
                        help="Number of timesteps to use for sampling (default: use all timesteps)")
    
    return parser.parse_args()


def tensor_to_image(tensor, is_rgb=False):
    """Convert tensor to PIL Image"""
    # Denormalize: [-1, 1] -> [0, 1]
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    
    if is_rgb:
        # RGB image
        tensor = tensor.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        tensor = (tensor[0] * 255).byte()
        return Image.fromarray(tensor.cpu().numpy(), mode='RGB')
    else:
        # Grayscale image
        tensor = tensor.squeeze()  # [B, 1, H, W] -> [H, W]
        tensor = (tensor * 255).byte()
        return Image.fromarray(tensor.cpu().numpy(), mode='L')


def main():
    args = parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine channels
    if args.is_rgb:
        in_channels = 3
        out_channels = 3
    else:
        in_channels = 1
        out_channels = 1

    # Initialize model
    print(f"Loading model: {args.model_name}")
    net_model = WTUNet(
        T=args.T,
        ch=args.ch,
        ch_mult=args.ch_mult,
        attn=args.attn,
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        in_channels=in_channels,
        out_channels=out_channels
    ).to(device)

    # Load weights
    if not os.path.exists(args.model_weight_path):
        raise FileNotFoundError(f"Model weights not found at {args.model_weight_path}")
    
    net_model.load_state_dict(
        torch.load(args.model_weight_path, map_location=device, weights_only=True)
    )
    net_model.eval()
    print(f"Loaded model weights from {args.model_weight_path}")

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
        sampling_timesteps=args.sampling_timesteps  # Pass the new argument
    )

    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    
    with torch.no_grad():
        for i in range(args.num_samples):
            print(f"Generating sample {i+1}/{args.num_samples}...")
            
            # Generate image
            channels = 3 if args.is_rgb else 1
            generated = sampler(
                batch_size=1,
                channels=channels * 4,  # Wavelet coefficients (4 for each channel)
                image_size=args.image_size,
                device=device
            )
            
            # Convert to image and save
            img = tensor_to_image(generated, is_rgb=args.is_rgb)
            
            # Save image
            output_path = os.path.join(
                args.output_dir,
                f"generated_sample_{i:04d}.png"
            )
            img.save(output_path)
            print(f"Saved: {output_path}")

    print(f"Generation completed! Images saved to {args.output_dir}")


if __name__ == '__main__':
    args = parse_args()
    main()
