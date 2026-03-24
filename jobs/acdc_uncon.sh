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

# 如果目录不存在，创建它们
mkdir -p "${OUTPUT_DIR}"

# ============================================================
# 步骤1: 数据准备
# ============================================================
# echo "=========================================="
# echo "数据准备..."
# echo "=========================================="

# echo "使用数据目录: ${DATA_ROOT}"
# IMAGE_COUNT=$(find "${DATA_ROOT}" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
# echo "✓ 检测到 ${IMAGE_COUNT} 个图像文件"

# ============================================================
# 步骤2: 训练无条件模型
# ============================================================
echo "=========================================="
echo "开始训练无条件D3CG模型..."
echo "=========================================="

python unconditional_train.py \
    --model_name "${MODEL_NAME}" \
    --dataset_dir "${DATA_ROOT}" \
    --out_name "${OUT_NAME}" \
    --batch_size 512 \
    --n_epochs 1000\
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
    --save_weight_dir "${OUTPUT_DIR}" \
    --save_interval 50 \
    --val_start_epoch 100 \
    --val_num 8 \
    --wave_type db4 \
    --transform_levels 1

echo "训练完成！"

# ============================================================
# 步骤3: 获取最优模型权重路径
# ============================================================
echo "=========================================="
echo "查找最优模型权重..."
echo "=========================================="

BEST_MODEL=$(find "${OUTPUT_DIR}/${OUT_NAME}" -name "best_model_epoch_*.pt" -printf '%T@ %p\n' | sort -rn | cut -d' ' -f2- | head -1)

if [ -z "${BEST_MODEL}" ]; then
    echo "警告: 未找到最优模型，尝试寻找最后保存的模型..."
    BEST_MODEL=$(find "${OUTPUT_DIR}/${OUT_NAME}" -name "model_epoch_*.pt" -printf '%T@ %p\n' | sort -rn | cut -d' ' -f2- | head -1)
fi

echo "使用模型: ${BEST_MODEL}"

# ============================================================
# 步骤4: 生成新的心脏图像
# ============================================================
echo "=========================================="
echo "开始生成无条件样本..."
echo "=========================================="

python unconditional_generation.py \
    --model_name "${MODEL_NAME}" \
    --model_weight_path "${BEST_MODEL}" \
    --num_samples 100 \
    --batch_size 2 \
    --image_size 96 \
    --ch 64 \
    --ch_mult 1 2 3 4 \
    --num_res_blocks 2 \
    --dropout 0.1 \
    --output_dir "${OUTPUT_DIR}/generated_samples/" \
    --wave_type db4 \
    --transform_levels 1 \
    --sampling_timesteps 128

echo "生成完成！"

# ============================================================
# 步骤5: 统计和报告
# ============================================================
echo "=========================================="
echo "生成结果统计..."
echo "=========================================="

GENERATED_COUNT=$(find "${OUTPUT_DIR}/generated_samples/" -name "generated_sample_*.png" 2>/dev/null | wc -l)
echo "✓ 生成了 ${GENERATED_COUNT} 个图像样本"

echo ""
echo "=========================================="
echo "ACDC 无条件生成 Pipeline 完成！"
echo "=========================================="
echo "训练日志: ${OUTPUT_DIR}/acdc_uncond_db4/training_log.log"
echo "生成样本: ${OUTPUT_DIR}/generated_samples/"
echo "最优模型: ${BEST_MODEL}"
echo "=========================================="
