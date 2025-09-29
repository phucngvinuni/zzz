#!/bin/bash

# =================================================================
# Shell Script để Trực quan hóa tiến trình huấn luyện
# Chạy file visualize.py cho một danh sách các checkpoints
# =================================================================

# --- 1. Cấu hình ---

# Đường dẫn đến thư mục chứa các file checkpoint
# QUAN TRỌNG: Đây là thư mục con, ví dụ: ./results_reconstruction/otcfm/
# Hãy thay đổi cho đúng với cấu trúc thư mục của bạn.
export RUN_DIR="/home/trungphuc/Latent_CFM-66CF new/results_openimages_single_gpu/reconstruction_otcfm/"

# Đường dẫn đến MỘT ảnh cố định mà bạn muốn dùng để tái tạo
# Hãy chọn một ảnh đẹp, rõ nét từ bộ dữ liệu của bạn
export CONDITION_IMAGE="/home/trungphuc/jscc/Kodak/5.png"

# Thư mục để lưu tất cả các ảnh kết quả
# Tên thư mục sẽ được tạo dựa trên tên của ảnh điều kiện
BASENAME=$(basename "$CONDITION_IMAGE")
FILENAME_NO_EXT="${BASENAME%.*}"
export OUTPUT_DIR="./visualizations/${FILENAME_NO_EXT}_progress"

# GPU để chạy inference (ví dụ: GPU 0)
export CUDA_VISIBLE_DEVICES=0

# --- 2. Danh sách các checkpoint cần visualize ---

# Liệt kê các số step của các checkpoint bạn muốn kiểm tra, cách nhau bởi dấu cách.
# Ví dụ: nếu bạn đã lưu checkpoint ở step 1, 10000, 20000, ...
# Script sẽ tự động tìm file `weights_step_XXXX.pt`.
# LƯU Ý: Tên file checkpoint trong script visualize.py của bạn đang là
# `OpenImages_weights_step_{args.step}_Lcfm.pt`. Hãy đảm bảo tên này khớp
# hoặc sửa lại tên trong script visualize.py cho đơn giản hơn.
# Giả sử tên file là `weights_step_...pt`
CHECKPOINT_STEPS="130000"

# --- 3. Vòng lặp thực thi ---

echo "Bắt đầu quá trình trực quan hóa..."
echo "Thư mục checkpoints: $RUN_DIR"
echo "Ảnh điều kiện: $CONDITION_IMAGE"
echo "Lưu kết quả tại: $OUTPUT_DIR"
echo "================================================="

# Tạo thư mục output nếu chưa có
mkdir -p "$OUTPUT_DIR"

for step in $CHECKPOINT_STEPS
do
    # Kiểm tra xem file checkpoint có tồn tại không trước khi chạy
    # Cần sửa lại tên file cho khớp với script visualize.py
    CHECKPOINT_FILE="${RUN_DIR}weights_step_${step}.pt"
    if [ -f "$CHECKPOINT_FILE" ]; then
        echo "--- Đang xử lý checkpoint step: $step ---"
        python3 ./code/cifar10/visualize.py \
          --run_dir "$RUN_DIR" \
          --step "$step" \
          --condition_image "$CONDITION_IMAGE" \
          --output_dir "$OUTPUT_DIR"
        echo "----------------------------------------"
    else
        echo "--- Bỏ qua step: $step (không tìm thấy checkpoint) ---"
    fi
done

echo "================================================="
echo "Hoàn tất! Hãy kiểm tra các ảnh trong thư mục $OUTPUT_DIR"
echo "================================================="