# File: compute_fid_openimages.py
# Đã cập nhật để làm việc với mô hình huấn luyện trên OpenImages và VAE của Stability AI.

import os
import sys
sys.path.append('./code/cifar10/')
sys.path.append('./code/torchcfm/models/unet/')

import matplotlib.pyplot as plt
import torch
from absl import app, flags
from cleanfid import fid
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

# --- Imports đã cập nhật ---
from diffusers.models import AutoencoderKL
from datasets import load_dataset
# -------------------------

from unet_resnetVAE import UNetModelWrapper
from utils_cifar import infiniteloop

# --- Cấu hình Flags ---
FLAGS = flags.FLAGS
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_string("input_dir", "./results_openimages/", help="output_directory")
flags.DEFINE_string("model", "icfm", help="flow matching model type")
flags.DEFINE_integer("step", 50000, help="training steps of the checkpoint to load")

# Inference
flags.DEFINE_integer("integration_steps", 100, help="number of inference steps for Euler")
flags.DEFINE_string("integration_method", "euler", help="integration method to use (e.g., euler, dopri5)")
flags.DEFINE_integer("num_gen", 10000, help="number of samples to generate for FID")
flags.DEFINE_float("tol", 1e-5, help="Integrator tolerance for adaptive solvers")
flags.DEFINE_integer("batch_size_fid", 16, help="Batch size to compute FID (giảm xuống cho ảnh lớn)")
flags.DEFINE_bool('ema', True, help='Use EMA model')
flags.DEFINE_integer("class_cond", 1, help="Conditioning type - 0: none, 1: Latent-CFM")
flags.DEFINE_integer("image_size", 256, "Size of the generated images")

# Chạy parser cho flags
FLAGS(sys.argv)

# --- Các hằng số và cấu hình thiết bị ---
IMAGE_SIZE = FLAGS.image_size
LATENT_DIM = 4 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8) # 4 * 32 * 32 = 4096

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# --- Lớp Wrapper cho ODE Solver ---
class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""
    def __init__(self, model, y=None):
        super().__init__()
        self.model = model
        if y is not None:
            self.y = y

    def forward(self, t, x, *args, **kwargs):
        # Lúc inference, model.training=False nên sẽ không trả về mu, logvar
        output = self.model(t, x, y=self.y)
        # Chỉ lấy vt cho ODE solver
        return output[0] if isinstance(output, tuple) else output

# --- Hàm tải dữ liệu OpenImages ---
def get_openimages_dataloader_for_fid(batch_size, num_workers, image_size, cache_dir="./data_cache"):
    print("Đang chuẩn bị dataset OpenImages cho conditioning...")
    dataset = load_dataset("openimages", split='train', cache_dir=cache_dir)
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    def transform_images(examples):
        images = [img.convert("RGB") for img in examples['image'] if img is not None]
        if not images: return {'pixel_values': []}
        examples['pixel_values'] = [preprocess(img) for img in images]
        return examples

    dataset = dataset.with_transform(transform_images)
    # Giữ lại chỉ cột pixel_values
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != 'pixel_values'])
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    return dataloader

# --- Khởi tạo Dataloader ---
dataloader = get_openimages_dataloader_for_fid(FLAGS.batch_size_fid, 4, IMAGE_SIZE)
datalooper = infiniteloop(dataloader)

# --- Khởi tạo mô hình U-Net ---
print("Đang khởi tạo mô hình U-Net...")
new_net = UNetModelWrapper(
    dim=(3, IMAGE_SIZE, IMAGE_SIZE),
    num_res_blocks=2,
    num_channels=FLAGS.num_channel,
    channel_mult=(1, 1, 2, 2, 4, 4),
    num_heads=4,
    num_head_channels=64,
    attention_resolutions="32,16,8",
    dropout=0.1,
    num_classes=None,
    num_latents=LATENT_DIM,
    class_cond=False,
).to(device)
    
# --- Khởi tạo VAE ---
print("Đang tải VAE của Stability AI...")
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=torch.float32
).to(device)
vae.eval()
print("VAE đã được tải xong.")

# --- Tải Checkpoint ---
if FLAGS.class_cond == 0:
    PATH = f"{FLAGS.input_dir}/{FLAGS.model}/OpenImages_weights_step_{FLAGS.step}.pt" 
elif FLAGS.class_cond == 1:
    PATH = f"{FLAGS.input_dir}/{FLAGS.model}/OpenImages_weights_step_{FLAGS.step}_Lcfm.pt"

print("Đang tải checkpoint từ đường dẫn: ", PATH)
checkpoint = torch.load(PATH, map_location=device)
state_dict = checkpoint["ema_model"] if FLAGS.ema else checkpoint["net_model"]

# Xử lý DDP prefix 'module.' nếu có
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k
    new_state_dict[name] = v

new_net.load_state_dict(new_state_dict)
new_net.eval()
new_net.training = False # Quan trọng: Chuyển sang chế độ inference
print("Checkpoint đã được tải thành công.")

# --- Hàm tạo ảnh để tính FID ---
def gen_1_img(unused_arg):
    with torch.no_grad():
        if FLAGS.class_cond == 0:
            # Trường hợp không điều kiện
            x = torch.randn(FLAGS.batch_size_fid, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
            latent_cond = None
        
        elif FLAGS.class_cond == 1:
            # Trường hợp Latent-CFM
            batch = next(datalooper)
            x1 = batch['pixel_values'].to(device)

            # 1. Encode ảnh điều kiện để lấy latent features
            latent_dist = vae.encode(x1).latent_dist
            latent_features = latent_dist.mean * vae.config.scaling_factor
            latent_features = latent_features.view(FLAGS.batch_size_fid, -1)
            
            # 2. Dùng U-Net để dự đoán tham số của phân phối latent
            latent_params = new_net.latent_encodings(latent_features)
            mu, logvar = torch.chunk(latent_params, 2, dim=1)
            
            # 3. Sample từ phân phối latent để tạo điều kiện cuối cùng (Reparameterization Trick)
            latent_cond = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            
            # 4. Tạo nhiễu Gauss làm điểm bắt đầu cho flow
            x = torch.randn(FLAGS.batch_size_fid, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        
        # Tích hợp ODE
        if FLAGS.integration_method == "euler":
            node = NeuralODE(torch_wrapper(new_net, y=latent_cond), solver=FLAGS.integration_method)
            t_span = torch.linspace(0, 1, FLAGS.integration_steps + 1, device=device)
            traj = node.trajectory(x, t_span=t_span)
        else:
            t_span = torch.linspace(0, 1, 2, device=device)
            traj = odeint(
                torch_wrapper(new_net, latent_cond), x, t_span, rtol=FLAGS.tol, atol=FLAGS.tol, method=FLAGS.integration_method
            )
            
    # Lấy kết quả cuối cùng tại t=1
    traj = traj[-1, :]
    
    # Chuyển đổi từ [-1, 1] sang [0, 255] dạng uint8 cho cleanfid
    img = (traj * 127.5 + 127.5).clip(0, 255).to(torch.uint8)

    # Lưu một vài ảnh mẫu để kiểm tra
    if np.random.rand() < 0.01: # Chỉ lưu ảnh 1% số lần gọi hàm này
        traj_sample = traj[:16,].view([-1, 3, IMAGE_SIZE, IMAGE_SIZE]).clip(-1, 1)
        traj_sample = traj_sample / 2 + 0.5
        save_path = f"{FLAGS.input_dir}/{FLAGS.model}/Generated_images_step_{FLAGS.step}_Lcfm_OpenImages_FID.png"
        save_image(traj_sample, save_path, nrow=4)
        print(f"Đã lưu ảnh mẫu tại {save_path}")

    return img

# --- Tính toán và in ra FID ---
print("Bắt đầu tính toán FID...")
# cleanfid sẽ tự động tải dataset nếu chưa có
score = fid.compute_fid(
    gen=gen_1_img,
    dataset_name="coco_train2017", # Sử dụng COCO làm bộ dữ liệu tham chiếu
    dataset_res=IMAGE_SIZE,
    num_gen=FLAGS.num_gen,
    batch_size=FLAGS.batch_size_fid,
    dataset_split="train",
    mode="legacy_tensorflow",
)
print("\n-------------------------------------")
print("Đã tính toán FID xong!")
print(f"Model: {FLAGS.model}, Step: {FLAGS.step}, EMA: {FLAGS.ema}")
print(f"Integration: {FLAGS.integration_method}, Steps: {FLAGS.integration_steps}")
print(f"FID score (vs COCO train 2017, {FLAGS.num_gen} samples): {score}")
print("-------------------------------------")