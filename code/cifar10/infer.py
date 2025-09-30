# File: infer.py
# Mô tả: Script để tạo ảnh từ một checkpoint Latent-CFM đã huấn luyện.

import os
import sys
import torch
import argparse
from PIL import Image
from torchvision import transforms
from collections import OrderedDict

# Thêm các thư mục cần thiết vào PYTHONPATH
sys.path.append('./code/cifar10/')
sys.path.append('./code/torchcfm/models/unet/')

# Imports từ dự án
from unet_resnetVAE import UNetModelWrapper
from diffusers.models import AutoencoderKL
from torchdyn.core import NeuralODE

# --- 1. Cấu hình Tham số Dòng lệnh ---
parser = argparse.ArgumentParser(description="Inference script for Latent-CFM")
parser.add_argument("--input_dir", type=str, default="./results_openimages_single_gpu/", help="Thư mục chứa checkpoints")
parser.add_argument("--model_name", type=str, default="icfm", help="Tên model (dùng để xây dựng đường dẫn)")
parser.add_argument("--step", type=int, required=True, help="Số bước của checkpoint cần tải (ví dụ: 8000)")
parser.add_argument("--condition_image", type=str, required=True, help="Đường dẫn đến ảnh dùng để làm điều kiện")
parser.add_argument("--output_image", type=str, default="generated_image.png", help="Đường dẫn để lưu ảnh kết quả")
parser.add_argument("--image_size", type=int, default=256, help="Kích thước ảnh")
parser.add_argument("--num_channel", type=int, default=128, help="Số kênh cơ sở của U-Net")
parser.add_argument("--seed", type=int, default=42, help="Seed để tái tạo kết quả")
parser.add_argument("--use_mu", action='store_true', help="Dùng mu thay vì sample latent_cond (cho kết quả tất định hơn)")
parser.add_argument("--integration_steps", type=int, default=100, help="Số bước tích phân cho Euler solver")
args = parser.parse_args()

# --- 2. Chuẩn bị ---

# Thiết lập seed
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# Thiết lập thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")

# Lớp Wrapper cho ODE Solver
class torch_wrapper(torch.nn.Module):
    def __init__(self, model, y=None):
        super().__init__()
        self.model = model
        self.y = y

    def forward(self, t, x, *args, **kwargs):
        output = self.model(t, x, y=self.y)
        return output[0] if isinstance(output, tuple) else output

# --- 3. Tải các mô hình ---

# Tải VAE từ thư mục cục bộ
vae_local_path = "./models/vae-ft-mse"
print(f"Đang tải VAE từ '{vae_local_path}'...")
vae = AutoencoderKL.from_pretrained(vae_local_path, torch_dtype=torch.float32).to(device)
vae.eval()

# Khởi tạo kiến trúc U-Net
print("Đang khởi tạo kiến trúc U-Net...")
LATENT_DIM = 4 * (args.image_size // 8) * (args.image_size // 8)
net_model = UNetModelWrapper(
    dim=(3, args.image_size, args.image_size), num_res_blocks=2,
    num_channels=args.num_channel, channel_mult=(1, 1, 2, 2, 4, 4),
    num_heads=4, num_head_channels=64, attention_resolutions="32,16,8",
    dropout=0.1, num_latents=LATENT_DIM
).to(device)

# Tải checkpoint đã huấn luyện
checkpoint_path = os.path.join(args.input_dir, args.model_name, f"OpenImages_weights_step_{args.step}_Lcfm.pt")
print(f"Đang tải checkpoint từ '{checkpoint_path}'...")
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Không tìm thấy file checkpoint! Kiểm tra lại đường dẫn và số step.")

checkpoint = torch.load(checkpoint_path, map_location=device)
# Sử dụng EMA model cho kết quả tốt hơn
state_dict = checkpoint["ema_model"]

# Xử lý prefix 'module.' nếu checkpoint được lưu từ DDP
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k
    new_state_dict[name] = v
net_model.load_state_dict(new_state_dict)
net_model.eval()
net_model.training = False # Rất quan trọng!

print("Tải mô hình thành công!")

# --- 4. Xử lý ảnh điều kiện và tạo Latent ---
print(f"Đang xử lý ảnh điều kiện: '{args.condition_image}'...")
preprocess = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

image = Image.open(args.condition_image).convert("RGB")
# Thêm một chiều batch (1, C, H, W)
x1 = preprocess(image).unsqueeze(0).to(device)

with torch.no_grad():
    # 1. Encode ảnh điều kiện
    latent_dist = vae.encode(x1).latent_dist
    latent_features = latent_dist.mean * vae.config.scaling_factor
    latent_features = latent_features.view(1, -1) # Batch size là 1

    # 2. Lấy mu, logvar từ U-Net
    latent_params = net_model.latent_encodings(latent_features)
    mu, logvar = torch.chunk(latent_params, 2, dim=1)

    # 3. Tạo latent condition cuối cùng
    if args.use_mu:
        print("Sử dụng mu làm latent condition (tất định).")
        latent_cond = mu
    else:
        print("Sample latent condition từ phân phối (ngẫu nhiên).")
        # Dùng seed đã set để đảm bảo tính tái lập
        latent_cond = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

# --- 5. Chạy ODE Solver để tạo ảnh ---
print("Bắt đầu quá trình tạo ảnh với ODE solver...")
with torch.no_grad():
    # Điểm bắt đầu là nhiễu Gauss
    x0 = torch.randn(1, 3, args.image_size, args.image_size, device=device)

    # Khởi tạo ODE solver
    node = NeuralODE(torch_wrapper(net_model, y=latent_cond), solver="euler", sensitivity="adjoint")
    t_span = torch.linspace(0, 1, args.integration_steps + 1, device=device)
    
    # Chạy quỹ đạo
    traj = node.trajectory(x0, t_span=t_span)
    
    # Lấy kết quả cuối cùng tại t=1
    generated_tensor = traj[-1]

print("Tạo ảnh thành công!")

# --- 6. Hậu xử lý và Lưu ảnh ---
# Chuyển đổi tensor từ [-1, 1] về [0, 1]
generated_tensor = (generated_tensor.squeeze() / 2 + 0.5).clamp(0, 1)

# Chuyển đổi sang ảnh PIL để lưu
to_pil = transforms.ToPILImage()
generated_image = to_pil(generated_tensor.cpu())

# Lưu ảnh
generated_image.save(args.output_image)
print(f"Ảnh đã được lưu tại: '{args.output_image}'")