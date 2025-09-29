#!/bin/bash

# =================================================================
# Shell Script để đánh giá ảnh hưởng của số bước ODE (NFE)
# =================================================================

# --- 1. Cấu hình ---

# Thư mục chứa các file checkpoint
export RUN_DIR="./results_openimages_single_gpu/reconstruction_otcfm/"

# Số step của checkpoint bạn muốn đánh giá
export CHECKPOINT_STEP=130000

# Thư mục chứa các ảnh gốc để đánh giá (ví dụ: toàn bộ bộ Kodak)
export IMAGE_DIR="/home/trungphuc/jscc/Kodak/"

# Thư mục để lưu kết quả (file CSV và các ảnh tái tạo)
export OUTPUT_DIR="./nfe_evaluation/step_${CHECKPOINT_STEP}"

# GPU để chạy
export CUDA_VISIBLE_DEVICES=0

# Danh sách các giá trị NFE cần kiểm tra, cách nhau bởi dấu phẩy
export NFE_LIST="1,2,3,4,5,10,15,20,30,50,100"


# --- 2. Lệnh thực thi ---
echo "Bắt đầu đánh giá NFE cho checkpoint step ${CHECKPOINT_STEP}..."
echo "Dữ liệu ảnh từ: ${IMAGE_DIR}"
echo "Lưu kết quả tại: ${OUTPUT_DIR}"
echo "Các giá trị NFE sẽ được kiểm tra: ${NFE_LIST}"
echo "================================================="

python3 ./code/cifar10/evalode.py \
  --run_dir "$RUN_DIR" \
  --checkpoint_step "$CHECKPOINT_STEP" \
  --image_dir "$IMAGE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --nfe_list "$NFE_LIST" \
  --image_size 128 \
  --num_channel 128

echo "================================================="
echo "Hoàn tất đánh giá!"
echo "File CSV kết quả: ${OUTPUT_DIR}/results_step_${CHECKPOINT_STEP}.csv"
echo "================================================="