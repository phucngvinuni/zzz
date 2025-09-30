# File: evaluate_nfe.py
# Mô tả: Đánh giá chất lượng tái tạo (PSNR, MS-SSIM)
#        với các số bước ODE (NFE) khác nhau.

import os, sys, torch, argparse, csv
from PIL import Image
from torchvision import transforms
from collections import OrderedDict
from torchvision.utils import save_image
sys.path.append('./code/cifar10/'); sys.path.append('./code/torchcfm/models/unet/')

from unet_resnetVAE import UNetModelWrapper
from diffusers.models import AutoencoderKL
from torchdyn.core import NeuralODE
from torchmetrics.image import PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure

# --- Lớp Wrapper (giữ nguyên) ---
class torch_wrapper(torch.nn.Module):
    def __init__(self, model, y): super().__init__(); self.model, self.y = model, y
    def forward(self, t, x, **kwargs): return self.model(t, x, y=self.y)[0]

def main():
    # --- 1. Cấu hình Tham số ---
    parser = argparse.ArgumentParser(description="Evaluate NFE for Latent-CFM")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--checkpoint_step", type=int, required=True)
    parser.add_argument("--image_dir", type=str, required=True, help="Thư mục chứa các ảnh để đánh giá")
    parser.add_argument("--output_dir", type=str, default="./nfe_evaluation/")
    parser.add_argument("--nfe_list", type=str, default="1,2,5,10,20,50,100", help="Danh sách các NFE cần kiểm tra")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--num_channel", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    nfe_values = [int(nfe) for nfe in args.nfe_list.split(',')]

    # --- 2. Chuẩn bị ---
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # --- 3. Tải các mô hình (chỉ 1 lần) ---
    vae = AutoencoderKL.from_pretrained("./models/vae-ft-mse", torch_dtype=torch.float32).to(device)
    vae.eval()

    channel_mult = (1, 2, 2) if args.image_size == 128 else (1, 1, 2, 2, 4, 4)
    attention_res = "16" if args.image_size == 128 else "32,16,8"
    net_model = UNetModelWrapper(
        dim=(3, args.image_size, args.image_size), num_res_blocks=2, num_channels=args.num_channel,
        channel_mult=channel_mult, attention_resolutions=attention_res,
        num_latents=4*(args.image_size//8)**2, deterministic=True, dropout=0.2
    ).to(device)

    checkpoint_path = os.path.join(args.run_dir, f"weights_step_{args.checkpoint_step}.pt")
    print(f"Đang tải checkpoint: '{checkpoint_path}'...")
    state_dict = torch.load(checkpoint_path, map_location=device)["ema_model"]
    new_state_dict = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in state_dict.items())
    net_model.load_state_dict(new_state_dict)
    net_model.eval()

    # --- 4. Chuẩn bị tính toán metric ---
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(
    data_range=1.0, 
    betas=(0.0448, 0.2856, 0.3001, 0.2363) # Chỉ sử dụng 4 scales
    ).to(device)
    
    # Chuẩn bị file CSV để lưu kết quả
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"results_step_{args.checkpoint_step}.csv")
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Image", "NFE", "PSNR", "MS_SSIM"])

    # --- 5. Vòng lặp đánh giá ---
    image_paths = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    preprocess = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    for img_path in image_paths:
        print(f"\n--- Đang xử lý ảnh: {os.path.basename(img_path)} ---")
        original_pil = Image.open(img_path).convert("RGB")
        x1 = preprocess(original_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            latent_features = vae.encode(x1).latent_dist.mean * vae.config.scaling_factor
            latent_features = latent_features.view(1, -1)
            node = NeuralODE(torch_wrapper(net_model, y=latent_features), solver="euler")
            
            for nfe in nfe_values:
                torch.manual_seed(args.seed) # Reset seed để nhiễu x0 là như nhau
                x0 = torch.randn_like(x1)
                t_span = torch.linspace(0, 1, nfe + 1, device=device)
                
                traj = node.trajectory(x0, t_span=t_span)
                generated_tensor = traj[-1]
                
                # Chuyển về thang đo [0, 1] để tính metric
                original_norm = (x1 + 1) / 2
                generated_norm = (generated_tensor + 1) / 2
                
                psnr = psnr_metric(generated_norm, original_norm).item()
                ms_ssim = ms_ssim_metric(generated_norm, original_norm).item()
                
                print(f"NFE: {nfe:3d} | PSNR: {psnr:.4f} | MS-SSIM: {ms_ssim:.4f}")
                csv_writer.writerow([os.path.basename(img_path), nfe, f"{psnr:.4f}", f"{ms_ssim:.4f}"])

                # Lưu ảnh ví dụ
                img_dir = os.path.join(args.output_dir, os.path.basename(img_path).split('.')[0])
                os.makedirs(img_dir, exist_ok=True)
                save_image(generated_norm, os.path.join(img_dir, f"nfe_{nfe}.png"))

    csv_file.close()
    print(f"\nHoàn tất! Kết quả đã được lưu tại: '{csv_path}'")

if __name__ == "__main__":
    main()