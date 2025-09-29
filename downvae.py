# File: save_vae_locally.py
from diffusers.models import AutoencoderKL
import os

# Tên model trên Hugging Face
model_id = "stabilityai/sd-vae-ft-mse"
# Thư mục để lưu VAE trên máy của bạn
local_save_directory = "./models/vae-ft-mse"

# Tạo thư mục nếu chưa có
os.makedirs(local_save_directory, exist_ok=True)

print(f"Bắt đầu tải VAE '{model_id}'...")
# Tải và lưu mô hình
vae = AutoencoderKL.from_pretrained(model_id)
vae.save_pretrained(local_save_directory)

print(f"Đã tải và lưu VAE thành công vào thư mục: '{local_save_directory}'")