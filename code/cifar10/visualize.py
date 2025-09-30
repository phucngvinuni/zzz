# File: visualize.py
# Mô tả: Tải một checkpoint cụ thể và tái tạo một ảnh điều kiện để trực quan hóa.
# Phiên bản này đã sửa lỗi kích thước tensor.

import os
import sys
import torch
import argparse
from PIL import Image
from torchvision import transforms
from collections import OrderedDict
from torchvision.utils import save_image

# Thêm các thư mục cần thiết vào PYTHONPATH
sys.path.append('./code/cifar10/')
sys.path.append('./code/torchcfm/models/unet/')

# Imports từ dự án
from unet_resnetVAE import UNetModelWrapper
from diffusers.models import AutoencoderKL
from torchdyn.core import NeuralODE

# --- 1. Cấu hình Tham số Dòng lệnh ---
parser = argparse.ArgumentParser(description="Visualization script for Latent-CFM Reconstruction")
parser.add_argument("--run_dir", type=str, required=True, help="Thư mục của lần chạy (ví dụ: ./results_reconstruction/otcfm/)")
parser.add_argument("--step", type=int, required=True, help="Số bước của checkpoint cần tải")
parser.add_argument("--condition_image", type=str, required=True, help="Đường dẫn đến ảnh điều kiện cố định")
parser.add_argument("--output_dir", type=str, default="./visualizations/", help="Thư mục để lưu tất cả các ảnh kết quả")
parser.add_argument("--image_size", type=int, default=128, help="Kích thước ảnh")
parser.add_argument("--num_channel", type=int, default=128, help="Số kênh cơ sở của U-Net")
parser.add_argument("--seed", type=int, default=42, help="Seed để đảm bảo nhiễu ban đầu là như nhau")
parser.add_argument("--integration_steps", type=int, default=2, help="Số bước tích phân cho Euler solver")
args = parser.parse_args()

# --- 2. Lớp Wrapper cho ODE Solver (giữ nguyên) ---
class torch_wrapper(torch.nn.Module):
    def __init__(self, model, y=None):
        super().__init__()
        self.model = model
        self.y = y # y ở đây phải là latent_features (4096D)
    def forward(self, t, x, **kwargs):
        output = self.model(t, x, y=self.y)
        return output[0] # Chỉ lấy vt

def main():
    # --- 3. Chuẩn bị ---
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.step}] Sử dụng thiết bị: {device}")

    # --- 4. Tải các mô hình ---
    if not os.path.exists("./models/vae-ft-mse"):
        raise FileNotFoundError("Thư mục VAE cục bộ không tồn tại. Vui lòng chạy script save_vae_locally.py trước.")
    vae = AutoencoderKL.from_pretrained("./models/vae-ft-mse", torch_dtype=torch.float32).to(device)
    vae.eval()

    LATENT_DIM = 4 * (args.image_size // 8) * (args.image_size // 8)
    # Khởi tạo U-Net ở chế độ tất định (reconstruction)
    net_model = UNetModelWrapper(
        dim=(3, args.image_size, args.image_size), num_res_blocks=2, num_channels=args.num_channel,
        channel_mult=(1, 2, 2) if args.image_size == 128 else (1, 1, 2, 2, 4, 4), num_heads=4, num_head_channels=64,
        attention_resolutions="16" if args.image_size == 128 else "32,16,8", num_latents=4*(args.image_size//8)**2,
        deterministic=True, dropout=0.2
    ).to(device)
    checkpoint_path = os.path.join(args.run_dir, f"weights_step_{args.step}.pt")
    print(f"[{args.step}] Đang tải checkpoint: '{checkpoint_path}'...")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"LỖI: Không tìm thấy checkpoint tại '{checkpoint_path}'.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["ema_model"]
    
    # Xử lý prefix 'module.' nếu có
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    net_model.load_state_dict(new_state_dict)
    net_model.eval() # Chuyển sang chế độ inference

    # --- 5. Xử lý ảnh điều kiện và tạo Latent ---
    preprocess = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    image = Image.open(args.condition_image).convert("RGB")
    x1 = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Trích xuất latent_features (4096D) từ VAE. Đây sẽ là điều kiện cho ODE.
        latent_dist = vae.encode(x1).latent_dist
        latent_features = latent_dist.mean * vae.config.scaling_factor
        latent_features = latent_features.view(1, -1)
        
    # --- 6. Chạy ODE Solver để tạo ảnh ---
    print(f"[{args.step}] Bắt đầu quá trình tái tạo ảnh...")
    with torch.no_grad():
        # Nhiễu ban đầu, cố định nhờ seed
        x0 = torch.randn(1, 3, args.image_size, args.image_size, device=device)
        
        # Khởi tạo ODE solver với `latent_features` làm điều kiện
        node = NeuralODE(torch_wrapper(net_model, y=latent_features), solver="euler")
        t_span = torch.linspace(0, 1, args.integration_steps + 1, device=device)
        
        # Chạy quỹ đạo
        traj = node.trajectory(x0, t_span=t_span)
        generated_tensor = traj[-1]

    # --- 7. Hậu xử lý và Lưu ảnh ---
    # Ghép ảnh gốc và ảnh tái tạo để so sánh
    original_img_tensor = (x1.squeeze() / 2 + 0.5).clamp(0, 1)
    reconstructed_img_tensor = (generated_tensor.squeeze() / 2 + 0.5).clamp(0, 1)
    
    comparison_grid = torch.stack([original_img_tensor, reconstructed_img_tensor])
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = f"reconstruction_step_{args.step}.png"
    output_path = os.path.join(args.output_dir, output_filename)
    
    save_image(comparison_grid, output_path, nrow=2)
    print(f"[{args.step}] Tái tạo thành công! Ảnh so sánh đã được lưu tại: '{output_path}'")

if __name__ == "__main__":
    main()