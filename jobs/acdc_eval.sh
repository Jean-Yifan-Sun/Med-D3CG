#!/bin/bash
#SBATCH --account=qingjiem-heart-tte
#SBATCH --qos=bham
#SBATCH --time=128:00:00
#SBATCH --nodes 1
#SBATCH --gres gpu:1
#SBATCH --gpus-per-task 1
#SBATCH --tasks-per-node 1
#SBATCH --constraint=a100_80
#SBATCH --mem=256G  # 请求内存
set -e
module purge
module load baskerville
module load bask-apps/live/live
# module load CUDA/11.3.1

# 运行 Python 命令
source /bask/projects/q/qingjiem-heart-tte/yifansun/conda/miniconda/etc/profile.d/conda.sh
conda init
conda activate medd3cg
conda info --envs
cd /bask/projects/c/chenhp-data-gen/yifansun/project/Med-D3CG/
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_LAUNCH_BLOCKING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200

# ============================================================
# ACDC 无条件生成 Pipeline
# ============================================================
# 数据集特性: 96x96 单通道心脏图像
# ============================================================

# 配置参数
DATA_ROOT="/bask/projects/c/chenhp-data-gen/yifansun/project/Med-D3CG/data/acdc_wholeheart/25022_JPGs"
OUTPUT_DIR="./results/acdc_unconditional"
MODEL_NAME="D3CG_uncond_db4"
OUT_NAME="acdc_uncond_db4"
FID_STATS="/bask/projects/c/chenhp-data-gen/yifansun/project/DCTdiff/data/scratch/U-ViT2/assets/fid_stats/acdc_unlabel_wholeheart_greyscale.npz"

echo "=========================================="
echo "开始评估无条件D3CG模型..."
echo "=========================================="

python evaluate.py \
    --checkpoint_dir "$OUTPUT_DIR/${OUT_NAME}" \
    --output_dir "$OUTPUT_DIR/${OUT_NAME}/eval" \
    --real_data_dir "$DATA_ROOT" \
    --eval_interval 50 \
    --batch_size 512 \
    --image_size 96 \
    --model_name "$MODEL_NAME" \
    --wave_type "db4" \
    --transform_levels 1 \
    --T 1000 \
    --sampling_timesteps 512 \
    --ch 64 \
    --ch_mult 1 2 3 4 \
    --num_res_blocks 2 \
    --dropout 0.1 \
    --fid_stats "$FID_STATS" \
    --no_temp_dir_lpips 
echo "=========================================="
echo "评估完成！结果保存在: $OUTPUT_DIR/${OUT_NAME}/eval"
echo "=========================================="


