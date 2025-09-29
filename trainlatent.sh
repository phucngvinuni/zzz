#!/bin/bash
export DATA_DIR="/home/trungphuc/jscc/Kodak"
export OUTPUT_DIR="./output_stage1_recon_loss"
export CUDA_VISIBLE_DEVICES=0

python3 ./code/cifar10/train_latent_space.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 16 \
  --num_workers 4 \
  --lr 5e-4 \
  --total_steps 300001 \
  --use_recon_loss=True \
  --recon_loss_weight 0.1 \
  --recon_ode_steps 5 \
  --vis_step 2500 \
  --save_step 10000 \
  --parallel=False