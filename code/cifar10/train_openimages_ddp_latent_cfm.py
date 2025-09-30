# File: train_openimages_ddp_latent_cfm.py
# Mô tả: Kịch bản huấn luyện hoàn chỉnh, tích hợp Perceptual Loss.

# --- Phần 1: Imports và Cài đặt ---
import sys, os, glob, copy, math
from PIL import Image
from collections import OrderedDict
sys.path.append('./code/cifar10/'); sys.path.append('./code/torchcfm/models/unet/')
import torch, torch.nn as nn, torchvision.models as models
from absl import app, flags
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import trange
from torchdiffeq import odeint

from utils_cifar import ema, infiniteloop, setup
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher
from unet_resnetVAE import UNetModelWrapper
from diffusers.models import AutoencoderKL
from torchdyn.core import NeuralODE

# --- Phần 2: Các hàm và lớp phụ trợ ---

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        for param in vgg.parameters(): param.requires_grad = False
        self.feature_layers = {'3', '8', '17', '26', '35'}
        self.vgg = vgg
        self.loss_fn = nn.L1Loss()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, input_img, target_img):
        input_img, target_img = (input_img + 1) / 2, (target_img + 1) / 2
        input_img, target_img = (input_img - self.mean) / self.std, (target_img - self.mean) / self.std
        loss = 0.0
        x, y = input_img, target_img
        for name, layer in self.vgg._modules.items():
            x, y = layer(x), layer(y)
            if name in self.feature_layers:
                loss += self.loss_fn(x, y)
        return loss

class VisODEWrapper(torch.nn.Module):
    def __init__(self, model, precomputed_z): super().__init__(); self.model, self.precomputed_z = model, precomputed_z
    def forward(self, t, x, **kwargs): return self.model(t, x, y=None, precomputed_z=self.precomputed_z)[0]

class TrainODEFunc(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, t, x, y):
        t_batch = t.repeat(x.shape[0]) if t.dim() == 0 and x.dim() > 1 else t
        return self.model(t_batch, x, y=y)[0]

def visualize_step(net_model, vae, condition_image, step, savedir, device, mode, seed=42):
    print(f"\n[INFO] Visualizing at step {step} in '{mode}' mode...")
    net_model.eval()
    vis_model = copy.deepcopy(net_model.module if hasattr(net_model, 'module') else net_model)
    torch.manual_seed(seed)
    with torch.no_grad():
        x1 = condition_image.to(device)
        images_to_save = [(x1.squeeze() / 2 + 0.5).clamp(0, 1)]
        latent_features = vae.encode(x1).latent_dist.mean * vae.config.scaling_factor
        latent_features = latent_features.view(1, -1)
        _, mu, logvar = vis_model(torch.zeros(1, device=device), torch.zeros_like(x1), y=latent_features)
        
        node_recon = NeuralODE(VisODEWrapper(vis_model, precomputed_z=mu), solver="euler")
        x0 = torch.randn_like(x1)
        traj_recon = node_recon.trajectory(x0, torch.linspace(0, 1, 101, device=device))
        images_to_save.append((traj_recon[-1].squeeze() / 2 + 0.5).clamp(0, 1))

        if mode == 'generation' and logvar is not None:
            torch.manual_seed(seed + 1)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            node_gen = NeuralODE(VisODEWrapper(vis_model, precomputed_z=z), solver="euler")
            traj_gen = node_gen.trajectory(x0, torch.linspace(0, 1, 101, device=device))
            images_to_save.append((traj_gen[-1].squeeze() / 2 + 0.5).clamp(0, 1))

    save_image(torch.stack(images_to_save), os.path.join(savedir, f"vis_step_{step}.png"), nrow=len(images_to_save))
    print(f"[INFO] Visualization saved to '{savedir}/vis_step_{step}.png'")
    net_model.train()

class CustomImageDataset(Dataset):
    def __init__(self, r, t): self.p, self.t = [], t; [self.p.extend(glob.glob(os.path.join(r, e))) for e in ["*.jpg","*.jpeg","*.png","*.bmp"]]; print(f"Found {len(self.p)} images in '{r}'")
    def __len__(self): return len(self.p)
    def __getitem__(self, i): return {'pixel_values': self.t(Image.open(self.p[i]).convert("RGB"))}
def get_dataloader_from_path(p, b, n, s, par):
    t = transforms.Compose([transforms.Resize((s,s)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([.5,.5,.5],[.5,.5,.5])])
    d = CustomImageDataset(p, t)
    if len(d) == 0: raise ValueError(f"No images found in {p}")
    sam = DistributedSampler(d, shuffle=True) if par else None
    return DataLoader(d, b, sampler=sam, shuffle=not par, num_workers=n, drop_last=True, pin_memory=True), sam

def warmup_lr(step): return min(step, FLAGS.warmup) / FLAGS.warmup

# --- Flags ---
FLAGS = flags.FLAGS
flags.DEFINE_enum("mode", "reconstruction", ["generation", "reconstruction"], "Chế độ huấn luyện.")
flags.DEFINE_string("data_dir", None, "Đường dẫn đến thư mục chứa ảnh huấn luyện. BẮT BUỘC.")
flags.DEFINE_string("output_dir", "./results/", "Thư mục lưu checkpoints.")
flags.DEFINE_bool("use_recon_loss", True, "Bật/tắt loss L1.")
flags.DEFINE_float("l1_loss_weight", 0.5, "Trọng số cho L1 Reconstruction Loss.")
flags.DEFINE_bool("use_perceptual_loss", True, "Bật/tắt Perceptual Loss.")
flags.DEFINE_float("perceptual_loss_weight", 0.1, "Trọng số cho Perceptual Loss.")
flags.DEFINE_integer("recon_ode_steps", 5, "Số bước ODE để tính loss tái tạo.")
flags.DEFINE_integer("num_channel", 128, help="Số kênh cơ sở của U-Net")
flags.DEFINE_integer("image_size", 128, help="Kích thước ảnh huấn luyện")
flags.DEFINE_string("model", "otcfm", help="Sử dụng 'otcfm' cho Optimal Transport")
flags.DEFINE_float("lr", 1e-4, help="Tốc độ học")
flags.DEFINE_integer("total_steps", 200001, help="Tổng số bước huấn luyện")
flags.DEFINE_integer("batch_size", 4, help="Kích thước batch trên MỖI GPU")
flags.DEFINE_integer("num_workers", 0, help="Số luồng tải dữ liệu")
flags.DEFINE_bool("parallel", False, help="Bật chế độ đa GPU (DDP)")
flags.DEFINE_string("master_addr", "localhost", help="Địa chỉ master DDP")
flags.DEFINE_string("master_port", "12355", help="Cổng master DDP")
flags.DEFINE_integer("save_step", 10000, help="Tần suất lưu checkpoint")
flags.DEFINE_integer("vis_step", 5000, help="Tần suất visualize")
flags.DEFINE_string("restart_dir", None, "Đường dẫn checkpoint để phục hồi huấn luyện")
flags.DEFINE_float("ema_decay", 0.9999, help="Hệ số phân rã EMA")
flags.DEFINE_float("grad_clip", 1.0, help="Ngưỡng gradient clipping")
flags.DEFINE_integer("warmup", 5000, help="Số bước warmup")

# --- Main Training Function ---
def train(rank, total_num_gpus, argv):
    is_parallel = FLAGS.parallel and total_num_gpus > 1
    if is_parallel: setup(rank, total_num_gpus, FLAGS.master_addr, FLAGS.master_port)
    is_main_process = (not is_parallel) or (rank == 0)
    device = rank if not is_parallel else torch.device(f"cuda:{rank}")

    if not FLAGS.data_dir: raise ValueError("Cần cung cấp --data_dir")
    dataloader, sampler = get_dataloader_from_path(FLAGS.data_dir, FLAGS.batch_size, FLAGS.num_workers, FLAGS.image_size, is_parallel)
    datalooper = infiniteloop(dataloader)
    fixed_vis_img = next(iter(dataloader))['pixel_values'][:1] if is_main_process else None

    vae = AutoencoderKL.from_pretrained("./models/vae-ft-mse", torch_dtype=torch.float32).to(rank)
    vae.eval(); [p.requires_grad_(False) for p in vae.parameters()]

    deterministic_mode = (FLAGS.mode == 'reconstruction')
    net_model = UNetModelWrapper(
        dim=(3, FLAGS.image_size, FLAGS.image_size), num_res_blocks=2, num_channels=FLAGS.num_channel,
        channel_mult=(1, 2, 2) if FLAGS.image_size == 128 else (1, 1, 2, 2, 4, 4), num_heads=4, num_head_channels=64,
        attention_resolutions="16" if FLAGS.image_size == 128 else "32,16,8", num_latents=4*(FLAGS.image_size//8)**2,
        deterministic=deterministic_mode, dropout=0.2
    ).to(rank)
    ema_model = copy.deepcopy(net_model)

    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    
    global_step = 0
    if FLAGS.restart_dir and os.path.exists(FLAGS.restart_dir):
        if is_main_process: print(f"[INFO] Resuming training from: {FLAGS.restart_dir}")
        map_loc = {'cuda:0': f'cuda:{rank}'} if is_parallel else device
        checkpoint = torch.load(FLAGS.restart_dir, map_location=map_loc)
        
        # Load into the base model before DDP wrapping
        net_model.load_state_dict(checkpoint['net_model'])
        ema_model.load_state_dict(checkpoint['ema_model'])
        optim.load_state_dict(checkpoint['optim'])
        sched.load_state_dict(checkpoint['sched'])
        global_step = checkpoint['step']
        if is_main_process: print(f"[INFO] Resumed successfully. Continuing from step {global_step + 1}.")
    
    if is_parallel:
        net_model = DistributedDataParallel(net_model, device_ids=[rank], find_unused_parameters=not deterministic_mode)
    
    ode_func, perceptual_loss_fn = None, None
    if FLAGS.use_recon_loss or FLAGS.use_perceptual_loss:
        model_for_ode = net_model.module if is_parallel else net_model
        ode_func = TrainODEFunc(model_for_ode).to(rank)
    if FLAGS.use_perceptual_loss:
        perceptual_loss_fn = PerceptualLoss(device=rank).to(rank)

    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
    savedir = os.path.join(FLAGS.output_dir, f"{FLAGS.mode}_{FLAGS.model}")
    os.makedirs(savedir, exist_ok=True)
    
    steps_per_epoch, num_epochs = len(dataloader), math.ceil(FLAGS.total_steps / len(dataloader))
    start_epoch = global_step // steps_per_epoch

    for epoch in range(start_epoch, num_epochs):
        if sampler: sampler.set_epoch(epoch)
        pbar = trange(steps_per_epoch, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not is_main_process)
        if global_step % steps_per_epoch != 0 and epoch == start_epoch: pbar.update(global_step % steps_per_epoch)

        for _ in pbar:
            if global_step >= FLAGS.total_steps: break
            
            optim.zero_grad(set_to_none=True)
            x1 = next(datalooper)['pixel_values'].to(rank, non_blocking=True)
            with torch.no_grad():
                latent = vae.encode(x1).latent_dist.mean * vae.config.scaling_factor
                latent = latent.view(x1.size(0), -1)

            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt, mu, logvar = net_model(t, xt, y=latent)
            
            fm_loss = torch.mean((vt - ut) ** 2)
            loss = fm_loss
            if not deterministic_mode:
                loss += 0.001 * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)).mean()
            
            l1_loss_val, p_loss_val = 0.0, 0.0
            if (FLAGS.use_recon_loss and FLAGS.l1_loss_weight > 0) or (FLAGS.use_perceptual_loss and FLAGS.perceptual_loss_weight > 0):
                ode_func_wrapper = lambda t, x: ode_func(t, x, y=latent)
                t_span = torch.linspace(0, 1, FLAGS.recon_ode_steps, device=rank)
                with torch.set_grad_enabled(True):
                    traj = odeint(ode_func_wrapper, x0, t_span, method='euler')
                x_hat = traj[-1]
                
                if FLAGS.use_recon_loss and FLAGS.l1_loss_weight > 0:
                    l1_loss = nn.L1Loss()(x_hat, x1)
                    l1_loss_val = l1_loss.item()
                    loss += FLAGS.l1_loss_weight * l1_loss
                
                if FLAGS.use_perceptual_loss and FLAGS.perceptual_loss_weight > 0:
                    p_loss = perceptual_loss_fn(x_hat, x1)
                    p_loss_val = p_loss.item()
                    loss += FLAGS.perceptual_loss_weight * p_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_((net_model.module if is_parallel else net_model).parameters(), FLAGS.grad_clip)
            optim.step(); sched.step(); ema(net_model, ema_model, FLAGS.ema_decay)
            global_step += 1

            if is_main_process:
                pbar.set_postfix(fm=f"{fm_loss.item():.3f}", l1=f"{l1_loss_val:.3f}", percep=f"{p_loss_val:.3f}", lr=f"{sched.get_last_lr()[0]:.1e}")

            if is_main_process and (global_step % FLAGS.vis_step == 0 or (global_step == 1 and not FLAGS.restart_dir)):
                vis_dir = os.path.join(savedir, "visualizations")
                os.makedirs(vis_dir, exist_ok=True)
                visualize_step(ema_model, vae, fixed_vis_img, global_step, vis_dir, device, FLAGS.mode)

            if is_main_process and (global_step % FLAGS.save_step == 0):
                print(f"\n[INFO] Saving checkpoint at step {global_step}...")
                model_to_save = net_model.module if is_parallel else net_model
                ema_to_save = ema_model.module if hasattr(ema_model, 'module') else ema_model
                torch.save({
                    "net_model": model_to_save.state_dict(), "ema_model": ema_to_save.state_dict(),
                    "sched": sched.state_dict(), "optim": optim.state_dict(), "step": global_step,
                }, os.path.join(savedir, f"weights_step_{global_step}.pt"))

def main(argv):
    total_gpus = torch.cuda.device_count()
    if FLAGS.parallel and total_gpus > 1:
        train(int(os.getenv("RANK", 0)), total_gpus, argv)
    else:
        train(torch.device("cuda" if torch.cuda.is_available() else "cpu"), 1, argv)

if __name__ == "__main__":
    app.run(main)