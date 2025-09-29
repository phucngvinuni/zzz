#!/bin/bash
export DATA_DIR="/home/trungphuc/jscc/Kodak"
export OUTPUT_DIR="./results_openimages_single_gpu"
export CUDA_VISIBLE_DEVICES=0

# # --- CẤU HÌNH RESUME ---
# export CHECKPOINT_PATH="results_openimages_single_gpu/reconstruction_otcfm/weights_step_90000.pt"

# RESTART_ARG=""
# if [ -n "$CHECKPOINT_PATH" ]; then
#   RESTART_ARG="--restart_dir $CHECKPOINT_PATH"
# fi  # <--- THÊM `fi` VÀO ĐÂY ĐỂ ĐÓNG KHỐI if

# Các tham số khác
BATCH_SIZE=1
LEARNING_RATE=1e-4
# ... (các echo)

# Chú ý: ${RESTART_ARG} sẽ tự động mở rộng thành --restart_dir ... nếu có
python3 ./code/cifar10/train_openimages_ddp_latent_cfm.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --mode "reconstruction" \
  --model "otcfm" \
  --lr $LEARNING_RATE \
  --batch_size $BATCH_SIZE \
  --parallel=False \
  # ${RESTART_ARG}