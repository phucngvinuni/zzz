# File: train_latent_space.py
# Mô tả: Giai đoạn 1 - Huấn luyện Latent-CFM (Decoder) trong Latent Space
#        VỚI SỰ KẾT HỢP CỦA RECONSTRUCTION LOSS để đảm bảo hội tụ.

# --- Phần 1: Imports và Cài đặt ---
import sys, os, glob, copy, math
from PIL import Image
from collections import OrderedDict
sys.path.append('./code/cifar10/'); sys.path.append('./code/torchcfm/models/unet/')
import torch, torch.nn as nn
from absl import app, flags
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import trange
from torchdyn.core import NeuralODE
from torchdiffeq import odeint

# Imports từ dự án
from utils_cifar import ema, infiniteloop, setup
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from unet_latent import UNetModelWrapper
from diffusers.models import AutoencoderKL

# --- Phần 2: Các hàm và lớp phụ trợ ---

class VisODEWrapper(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, t, x, **kwargs): return self.model(t, x)

class TrainODEFunc(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, t, x):
        t_batch = t.repeat(x.shape[0]) if t.dim() == 0 and x.dim() > 1 else t
        return self.model(t_batch, x)

def visualize_step(net_model, vae, condition_image, step, savedir, device, scaling_factor, seed=42):
    print(f"\n[INFO] Visualizing at step {step}...")
    net_model.eval()
    vis_model = copy.deepcopy(net_model.module if hasattr(net_model, 'module') else net_model)
    torch.manual_seed(seed)
    with torch.no_grad():
        x1_pixel = condition_image.to(device)
        latent_x1_target = vae.encode(x1_pixel * 2.0 - 1.0).latent_dist.mean * scaling_factor
        latent_x0_noise = torch.randn_like(latent_x1_target)
        node = NeuralODE(VisODEWrapper(vis_model), solver="euler", sensitivity="adjoint")
        t_span = torch.linspace(0, 1, 101, device=device)
        traj = node.trajectory(latent_x0_noise, t_span)
        latent_hat = traj[-1]
        
        original_recon = vae.decode(latent_x1_target / scaling_factor).sample
        model_recon = vae.decode(latent_hat / scaling_factor).sample
        
        original_img = (x1_pixel.squeeze()).clamp(0, 1)
        original_recon_img = (original_recon.squeeze() / 2 + 0.5).clamp(0, 1)
        model_recon_img = (model_recon.squeeze() / 2 + 0.5).clamp(0, 1)
        
        save_image(torch.stack([original_img, original_recon_img, model_recon_img]),
                   os.path.join(savedir, f"vis_step_{step}.png"), nrow=3)
    print(f"[INFO] Visualization saved.")
    net_model.train()

# --- Flags ---
FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", None, "Đường dẫn đến thư mục ảnh huấn luyện.")
flags.DEFINE_string("output_dir", "./output_stage1_decoder/", "Thư mục lưu checkpoints của Latent-CFM.")
flags.DEFINE_bool("use_recon_loss", True, "Bật/tắt loss tái tạo trong latent space.")
flags.DEFINE_float("recon_loss_weight", 0.1, "Trọng số cho loss tái tạo latent.")
flags.DEFINE_integer("recon_ode_steps", 5, "Số bước ODE để tính loss tái tạo.")
flags.DEFINE_integer("image_size", 256, "Kích thước ảnh gốc để encode.")
flags.DEFINE_integer("num_channel", 128, "Số kênh cơ sở của U-Net.")
flags.DEFINE_integer("latent_channels", 4, "Số kênh của latent space.")
flags.DEFINE_integer("latent_size", 32, "Kích thước (H, W) của latent space.")
flags.DEFINE_float("lr", 5e-4, "Tốc độ học")
flags.DEFINE_integer("total_steps", 300001, "Tổng số bước huấn luyện")
flags.DEFINE_integer("batch_size", 16, "Kích thước batch trên MỖI GPU")
flags.DEFINE_integer("num_workers", 4, "Số luồng tải dữ liệu")
flags.DEFINE_bool("parallel", False, "Bật chế độ đa GPU (DDP)")
flags.DEFINE_string("master_addr", "localhost", help="Địa chỉ master DDP")
flags.DEFINE_string("master_port", "12355", help="Cổng master DDP")
flags.DEFINE_integer("save_step", 10000, "Tần suất lưu checkpoint")
flags.DEFINE_integer("vis_step", 2500, "Tần suất visualize")
flags.DEFINE_string("restart_dir", None, "Đường dẫn checkpoint để phục hồi huấn luyện")
flags.DEFINE_float("ema_decay", 0.9999, help="Hệ số phân rã EMA")
flags.DEFINE_float("grad_clip", 1.0, help="Ngưỡng gradient clipping")
flags.DEFINE_integer("warmup", 5000, help="Số bước warmup")

# --- Data Loading ---
class ImageDataset(Dataset):
    def __init__(self, r, t): self.p, self.t = [], t; [self.p.extend(glob.glob(os.path.join(r, e))) for e in ["*.jpg","*.jpeg","*.png","*.bmp"]]; print(f"Found {len(self.p)} images in '{r}'")
    def __len__(self): return len(self.p)
    def __getitem__(self, i): return self.t(Image.open(self.p[i]).convert("RGB"))
def get_dataloader(p, b, n, s, par):
    t = transforms.Compose([transforms.Resize((s,s)), transforms.ToTensor()])
    d = ImageDataset(p, t); sam = DistributedSampler(d, shuffle=True) if par else None
    return DataLoader(d, b, sampler=sam, shuffle=not par, num_workers=n, drop_last=True, pin_memory=True), sam

def warmup_lr(step): return min(step, FLAGS.warmup) / FLAGS.warmup

# --- Main Training Function ---
def train(rank, total_num_gpus, argv):
    is_parallel = FLAGS.parallel and total_num_gpus > 1
    if is_parallel: setup(rank, total_num_gpus, FLAGS.master_addr, FLAGS.master_port)
    is_main_process = (not is_parallel) or (rank == 0)
    device = rank if not is_parallel else torch.device(f"cuda:{rank}")

    dataloader, sampler = get_dataloader(FLAGS.data_dir, FLAGS.batch_size, FLAGS.num_workers, FLAGS.image_size, is_parallel)
    datalooper = infiniteloop(dataloader)
    fixed_vis_img = next(iter(dataloader))[:1] if is_main_process else None

    vae = AutoencoderKL.from_pretrained("./models/vae-ft-mse", torch_dtype=torch.float32).to(rank)
    vae.eval(); [p.requires_grad_(False) for p in vae.parameters()]
    scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
    
    net_model = UNetModelWrapper(
        dim=(FLAGS.latent_channels, FLAGS.latent_size, FLAGS.latent_size),
        num_res_blocks=2, num_channels=FLAGS.num_channel,
        channel_mult=(1, 2, 4, 8), attention_resolutions="16,8", dropout=0.1
    ).to(rank)
    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    
    global_step = 0
    if FLAGS.restart_dir and os.path.exists(FLAGS.restart_dir):
        if is_main_process: print(f"[INFO] Resuming training from: {FLAGS.restart_dir}")
        map_loc = {'cuda:0': f'cuda:{rank}'} if is_parallel else device
        checkpoint = torch.load(FLAGS.restart_dir, map_location=map_loc)
        net_model.load_state_dict(checkpoint['net_model'])
        ema_model.load_state_dict(checkpoint['ema_model'])
        optim.load_state_dict(checkpoint['optim'])
        sched.load_state_dict(checkpoint['sched'])
        global_step = checkpoint['step']
        if is_main_process: print(f"[INFO] Resumed successfully. Continuing from step {global_step + 1}.")

    if is_parallel:
        net_model = DistributedDataParallel(net_model, device_ids=[rank])
    
    if FLAGS.use_recon_loss:
        model_for_ode = net_model.module if is_parallel else net_model
        ode_func = TrainODEFunc(model_for_ode).to(rank)

    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
    savedir = os.path.join(FLAGS.output_dir, "checkpoints")
    visdir = os.path.join(FLAGS.output_dir, "visualizations")
    os.makedirs(savedir, exist_ok=True); os.makedirs(visdir, exist_ok=True)
    
    steps_per_epoch = len(dataloader)
    num_epochs = math.ceil(FLAGS.total_steps / steps_per_epoch)
    start_epoch = global_step // steps_per_epoch

    for epoch in range(start_epoch, num_epochs):
        if sampler: sampler.set_epoch(epoch)
        pbar = trange(steps_per_epoch, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not is_main_process)
        if global_step % steps_per_epoch != 0 and epoch == start_epoch: pbar.update(global_step % steps_per_epoch)

        for _ in pbar:
            if global_step >= FLAGS.total_steps: break
            
            optim.zero_grad(set_to_none=True)
            images_pixels = next(datalooper).to(rank, non_blocking=True)
            
            with torch.no_grad():
                latent_x1 = vae.encode(images_pixels * 2.0 - 1.0).latent_dist.mean * scaling_factor
            
            latent_x0 = torch.randn_like(latent_x1)
            t, latent_xt, ut_latent = FM.sample_location_and_conditional_flow(latent_x0, latent_x1)
            vt_latent = net_model(t, latent_xt)
            
            fm_loss = nn.MSELoss()(vt_latent, ut_latent)
            loss = fm_loss
            
            recon_loss_val = 0.0
            if FLAGS.use_recon_loss:
                t_span = torch.linspace(0, 1, FLAGS.recon_ode_steps, device=rank)
                traj_latent = odeint(ode_func, latent_x0, t_span, method='euler')
                latent_hat = traj_latent[-1]
                recon_loss = nn.L1Loss()(latent_hat, latent_x1)
                recon_loss_val = recon_loss.item()
                loss += FLAGS.recon_loss_weight * recon_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_((net_model.module if is_parallel else net_model).parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)
            global_step += 1

            if is_main_process: pbar.set_postfix(fm_loss=fm_loss.item(), recon_loss=recon_loss_val, lr=sched.get_last_lr()[0])

            if is_main_process and (global_step % FLAGS.vis_step == 0 or (global_step == 1 and not FLAGS.restart_dir)):
                visualize_step(ema_model, vae, fixed_vis_img, global_step, visdir, device, scaling_factor)
            
            if is_main_process and (global_step % FLAGS.save_step == 0):
                model_to_save = net_model.module if is_parallel else net_model
                ema_to_save = ema_model.module if hasattr(ema_model, 'module') else ema_model
                torch.save({
                    "net_model": model_to_save.state_dict(), "ema_model": ema_to_save.state_dict(),
                    "sched": sched.state_dict(), "optim": optim.state_dict(), "step": global_step,
                }, os.path.join(savedir, f"latent_cfm_step_{global_step}.pt"))

def main(argv):
    total_gpus = torch.cuda.device_count()
    if FLAGS.parallel and total_gpus > 1:
        train(int(os.getenv("RANK", 0)), total_gpus, argv)
    else:
        train(torch.device("cuda" if torch.cuda.is_available() else "cpu"), 1, argv)

if __name__ == "__main__":
    app.run(main)