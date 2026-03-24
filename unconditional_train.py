"""
无条件生成训练脚本

该脚本基于小波变换和扩散模型进行无条件图像生成训练。
与条件生成不同，该方案不依赖于输入条件，而是直接学习生成特定医学图像。
"""

import os
import time
import datetime
import torch
from torch.utils.data import DataLoader, Dataset
import logging
import numpy as np
import argparse
from PIL import Image
from torchvision import transforms

from models.base import BaseUNet
from models.attention import MidAttnUNet, UpAttnUNet
from models.wavelet import WTUNet
from utils.profiling import profile_model, log_profiling_results, get_model_input_shape
from diffusion.model_factory import get_trainer_sampler, get_available_models, print_model_info
from utils.metrics import calculate_metrics, calculate_metrics_rgb


class UnconditionalDataset(Dataset):
    """无条件数据集 - 仅包含目标图像"""
    
    def __init__(self, root_dir, image_size=256, transforms_=None, is_rgb=False):
        """
        Args:
            root_dir: 图像目录
            image_size: 图像大小
            transforms_: 图像变换
            is_rgb: 是否为RGB图像
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.is_rgb = is_rgb
        
        # 收集所有图像文件
        self.image_paths = []
        if os.path.isdir(root_dir):
            # 如果是文件夹结构 (样本/文件)
            for item in os.listdir(root_dir):
                item_path = os.path.join(root_dir, item)
                if os.path.isdir(item_path):
                    # 递归查找子目录中的图像文件
                    for file in os.listdir(item_path):
                        if file.endswith(('.png', '.jpg', '.jpeg')):
                            self.image_paths.append(os.path.join(item_path, file))
                else:
                    # 直接是图像文件
                    if item.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(item_path)
        
        # 添加调试日志
        import logging
        logging.info(f"Dataset initialized with root_dir: {root_dir}")
        logging.info(f"Found {len(self.image_paths)} image files")
        if len(self.image_paths) == 0:
            logging.warning(f"No images found in {root_dir}. Checking contents...")
            logging.warning(f"Directory exists: {os.path.isdir(root_dir)}")
            if os.path.isdir(root_dir):
                logging.warning(f"Contents: {os.listdir(root_dir)[:10]}")  # First 10 items
        
        # 定义变换
        if transforms_ is None:
            if is_rgb:
                self.transforms = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
            else:
                self.transforms = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
        else:
            self.transforms = transforms_

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB' if self.is_rgb else 'L')
        image = self.transforms(image)
        return {'image': image}


def parse_args():
    parser = argparse.ArgumentParser(description="Unconditional D3CG Training")
    
    # Model configuration
    all_models = get_available_models()
    available_models = (all_models.get("base_models", []) + 
                       all_models.get("d3cg_models", []) + 
                       all_models.get("d3cg_uncond_models", []) + 
                       all_models.get("custom_models", []))
    parser.add_argument("--model_name", type=str, default="D3CG_uncond_db4",
                        choices=available_models)
    
    # D3CG configuration for unconditional models
    parser.add_argument("--wave_type", type=str, default="db4",
                        choices=["haar", "db4", "coif3", "bior2.2", "dmey"],
                        help="Wavelet type for unconditional D3CG models")
    parser.add_argument("--transform_levels", type=int, default=1,
                        help="Number of wavelet transform levels")
    
    # Dataset configuration
    parser.add_argument("--dataset_dir", type=str, default="./data/medical_images",
                        help="Path to training dataset")
    parser.add_argument("--val_dataset_dir", type=str, default="./data/medical_images_val",
                        help="Path to validation dataset")
    parser.add_argument("--is_rgb", action="store_true", default=False,
                        help="Whether images are RGB (default: grayscale)")
    
    # Training configuration
    parser.add_argument("--out_name", type=str, default="D3CG_uncond",
                        help="Output model name")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--T", type=int, default=1000,
                        help="Number of diffusion steps")
    parser.add_argument("--ch", type=int, default=128,
                        help="Number of channels in base model")
    parser.add_argument("--ch_mult", nargs='+', type=int, default=[1, 2, 3, 4],
                        help="Channel multipliers")
    parser.add_argument("--attn", nargs='+', type=int, default=[2],
                        help="Attention layers")
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--grad_clip", type=float, default=1.)
    parser.add_argument("--image_size", type=int, default=256)
    
    # Training control
    parser.add_argument("--save_weight_dir", type=str, default="./results/unconditional")
    parser.add_argument("--resume_ckpt", type=str, default="",
                        help="Resume from checkpoint")
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--val_start_epoch", type=int, default=100,
                        help="Start validation from this epoch")
    parser.add_argument("--val_num", type=int, default=5,
                        help="Number of validation samples to generate")
    parser.add_argument("--save_interval", type=int, default=50,
                        help="Save model every N epochs")
    
    # Model info
    parser.add_argument("--list_models", action="store_true", 
                        help="List all available models")
    
    return parser.parse_args()


def should_save_model(current_loss, best_loss):
    """判断是否需要保存模型"""
    return current_loss < best_loss


def main():
    args = parse_args()
    
    # 显示可用模型信息
    if args.list_models:
        print_model_info()
        return
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup logging
    save_weight_dir = os.path.join(args.save_weight_dir, args.out_name)
    os.makedirs(save_weight_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(save_weight_dir, "training_log.log")),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Args: {args}")
    logging.info(f"Model: {args.model_name}")
    logging.info(f"Device: {device}")

    # Setup data
    logging.info("Loading datasets...")
    train_dataset = UnconditionalDataset(
        args.dataset_dir, 
        image_size=args.image_size,
        is_rgb=args.is_rgb
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4
    )
    logging.info(f"Training dataset size: {len(train_dataset)}")

    # Validation dataset (optional)
    val_dataloader = None
    if os.path.exists(args.val_dataset_dir):
        val_dataset = UnconditionalDataset(
            args.val_dataset_dir,
            image_size=args.image_size,
            is_rgb=args.is_rgb
        )
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
        logging.info(f"Validation dataset size: {len(val_dataset)}")

    # Determine channels based on RGB or grayscale
    if args.is_rgb:
        in_channels = 3  # RGB
        out_channels = 3
    else:
        in_channels = 1  # Grayscale
        out_channels = 1

    # Initialize model
    logging.info(f"Initializing model with in_channels={in_channels}, out_channels={out_channels}")
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

    optimizer = torch.optim.AdamW(
        net_model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0
    )

    # Initialize trainer and sampler using the factory
    logging.info("Initializing trainer and sampler...")
    trainer, sampler = get_trainer_sampler(
        model_name=args.model_name,
        net_model=net_model,
        beta_1=args.beta_1,
        beta_T=args.beta_T,
        T=args.T,
        device=device,
        # Unconditional D3CG parameters
        wave_type=args.wave_type,
        transform_levels=args.transform_levels
    )

    # Model profiling
    logging.info("Starting model profiling...")
    try:
        # For wavelet-based models, input should be wavelet-transformed (in_channels * 4)
        # For standard models, input is just the raw channels
        if "D3CG" in args.model_name or args.model_name == "WTDDPM":
            wavelet_channels = in_channels * 4
            # Wavelet transform reduces spatial dimensions by 2
            wavelet_size = args.image_size // 2
            input_shape = (args.batch_size, wavelet_channels, wavelet_size, wavelet_size)
        else:
            input_shape = (args.batch_size, in_channels, args.image_size, args.image_size)
        
        profiling_results = profile_model(net_model, input_shape, device, args.model_name)
        
        logging.info(f"Model profiling completed:")
        logging.info(f"  Parameters: {profiling_results['total_params_M']:.2f}M")
        logging.info(f"  FLOPs: {profiling_results['flops_G']:.3f}G")
        logging.info(f"  Memory: {profiling_results['model_memory_MB']:.1f}MB")
        logging.info(f"  Inference Time: {profiling_results['inference_time_ms']:.2f}ms")
    except Exception as e:
        logging.warning(f"Model profiling failed: {e}")

    # Resume from checkpoint if specified
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        net_model.load_state_dict(torch.load(args.resume_ckpt, map_location=device, weights_only=True))
        logging.info(f"Loaded checkpoint from {args.resume_ckpt}")

    # Training loop
    prev_time = time.time()
    best_loss = float('inf')

    for epoch in range(args.start_epoch, args.n_epochs + 1):
        net_model.train()
        losses = []

        # Training step
        for batch in train_dataloader:
            optimizer.zero_grad()
            x_0 = batch['image'].to(device)

            loss = trainer(x_0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        elapsed_time = time.time() - prev_time
        prev_time = time.time()

        logging.info(
            f'epoch {epoch:04d}: '
            f'loss = {avg_loss:.6f}, '
            f'elapsed = {datetime.timedelta(seconds=elapsed_time)}'
        )

        # Validation and sample generation
        if epoch >= args.val_start_epoch and epoch % args.save_interval == 0:
            logging.info(f"Generating samples for epoch {epoch}...")
            net_model.eval()
            
            # Generate unconditional samples
            sample_dir = os.path.join(save_weight_dir, f"epoch_{epoch:04d}_samples")
            os.makedirs(sample_dir, exist_ok=True)
            
            try:
                with torch.no_grad():
                    for i in range(args.val_num):
                        # Generate samples
                        channels = 3 if args.is_rgb else 1
                        generated = sampler(
                            batch_size=1,
                            channels=channels * 4,  # Wavelet coefficients
                            image_size=args.image_size,
                            device=device
                        )
                        
                        # Save generated samples
                        for j in range(generated.shape[0]):
                            img = generated[j].cpu()
                            if args.is_rgb:
                                img = (img * 0.5 + 0.5).clamp(0, 1)
                                from torchvision.transforms import ToPILImage
                                img_pil = ToPILImage()(img)
                            else:
                                img = (img.squeeze() * 0.5 + 0.5).clamp(0, 1)
                                from torchvision.transforms import ToPILImage
                                img_pil = ToPILImage()((img * 255).byte())
                            
                            img_pil.save(os.path.join(sample_dir, f"sample_{i:03d}.png"))
                
                logging.info(f"Samples saved to {sample_dir}")
            except Exception as e:
                logging.warning(f"Sample generation failed: {e}")

        # Save checkpoint
        if should_save_model(avg_loss, best_loss):
            best_loss = avg_loss
            ckpt_path = os.path.join(
                save_weight_dir,
                f'best_model_epoch_{epoch:04d}.pt'
            )
            torch.save(net_model.state_dict(), ckpt_path)
            logging.info(f'Saved checkpoint: {ckpt_path}')
        
        # Periodic checkpoint saving
        if epoch % args.save_interval == 0:
            ckpt_path = os.path.join(
                save_weight_dir,
                f'model_epoch_{epoch:04d}.pt'
            )
            torch.save(net_model.state_dict(), ckpt_path)
            logging.info(f'Saved periodic checkpoint: {ckpt_path}')

    logging.info("Training completed!")


if __name__ == '__main__':
    main()
