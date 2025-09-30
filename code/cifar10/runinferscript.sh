#!/bin/bash

# =================================================================
# Shell Script để Trực quan hóa tiến trình huấn luyện
# =================================================================

# --- 1. Cấu hình ---

# Đường dẫn đến thư mục chứa các checkpoint của lần chạy
# Ví dụ: ./results_reconstruction/otcfm/
export RUN_DIR="./results_reconstruction/otcfm/"

# Đường dẫn đến MỘT ảnh cố định mà bạn muốn dùng để tái tạo
# Hãy chọn một ảnh đẹp, rõ nét từ bộ dữ liệu của bạn
export CONDITION_IMAGE="/home/trungphuc/jscc/Kodak/kodim03.png"

# Thư mục để lưu tất cả các ảnh kết quả
export OUTPUT_DIR="./visualizations/kodim03_reconstruction_progress"

# GPU để chạy inference
export CUDA_VISIBLE_DEVICES=0

# --- 2. Danh sách các checkpoint cần visualize ---

# Liệt kê các số step của các checkpoint bạn muốn kiểm tra
# Ví dụ, nếu bạn đã lưu checkpoint ở step 1, 8000, 10000, 20000
CHECKPOINT_STEPS="1 8000 10000 20000 50000"

# --- 3. Vòng lặp thực thi ---

echo "Bắt đầu quá trình trực quan hóa..."
echo "Ảnh điều kiện: $CONDITION_IMAGE"
echo "Lưu kết quả tại: $OUTPUT_DIR"
echo "================================================="

for step in $CHECKPOINT_STEPS
do
    echo "--- Đang xử lý checkpoint step: $step ---"
    python3 ./code/cifar10/visualize.py \
      --run_dir "$RUN_DIR" \
      --step "$step" \
      --condition_image "$CONDITION_IMAGE" \
      --output_dir "$OUTPUT_DIR"
    echo "----------------------------------------"
done

echo "================================================="
echo "Hoàn tất! Hãy kiểm tra các ảnh trong thư mục $OUTPUT_DIR"
echo "================================================="