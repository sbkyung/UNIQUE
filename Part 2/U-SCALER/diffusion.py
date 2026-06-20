import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as T
from tqdm import tqdm
from functools import partial

def identity(t, *args, **kwargs):
    return t

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=4000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())
        
two_tuple = lambda x: x if isinstance(x, (list, tuple)) else (x, x)


class Diffusion(nn.Module):
    def __init__(self, model, flow, timesteps=100, device="cuda",
                 training_target='x0', noise_schedule='cosine'):
        
        super().__init__()
        self.device = device
        self.model = model
        self.flow = flow
        
        self.training_target = training_target.lower()
        assert self.training_target in ['x0', 'noise', 'v']

        assert noise_schedule in ['linear', 'cosine']
        if noise_schedule == 'linear':
            betas = linear_noise_schedule(timesteps)
        else:
            betas = cosine_noise_schedule(timesteps)

        self.num_timesteps = int(betas.shape[0])

        alphas = 1. - betas
        alphas_hat = np.cumprod(alphas, axis=0)
        alphas_hat_prev = np.append(1., alphas_hat[:-1])

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_hat', to_torch(alphas_hat))
        self.register_buffer('alphas_hat_prev', to_torch(alphas_hat_prev))
        self.register_buffer('sqrt_alphas_hat', to_torch(np.sqrt(alphas_hat)))
        self.register_buffer('sqrt_one_minus_alphas_hat', to_torch(np.sqrt(1. - alphas_hat)))
        self.register_buffer('log_one_minus_alphas_hat', to_torch(np.log(1. - alphas_hat)))
        self.register_buffer('sqrt_recip_alphas_hat', to_torch(np.sqrt(1. / alphas_hat)))
        self.register_buffer('sqrt_recipm1_alphas_hat', to_torch(np.sqrt(1. / alphas_hat - 1)))
        posterior_variance = betas * (1. - alphas_hat_prev) / (1. - alphas_hat)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_hat_prev) / (1. - alphas_hat)))
        self.register_buffer('posterior_mean_coef2',
                             to_torch((1. - alphas_hat_prev) * np.sqrt(alphas) / (1. - alphas_hat)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_hat[t] * x_t - self.sqrt_recipm1_alphas_hat[t] * noise

    def predict_noise_from_start(self, x_t, t, x0):
        return (self.sqrt_recip_alphas_hat[t] * x_t - x0) / self.sqrt_recipm1_alphas_hat[t]
    
    def predict_v(self, x_start, t, noise):
        return self.sqrt_alphas_hat[t] * noise - self.sqrt_one_minus_alphas_hat[t] * x_start
    
    def predict_start_from_v(self, x_t, t, v):
        return self.sqrt_alphas_hat[t] * x_t - self.sqrt_one_minus_alphas_hat[t] * v
    
    
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_variance = self.posterior_variance[t]
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, background, t, context=None, clip_denoised=False):
        batch_size = x.shape[0]
        t_tensor = torch.full((batch_size,), t, dtype=torch.int64, device=self.device)

        if self.training_target == 'x0':
            x_recon = self.model(x, t_tensor, background=background, context=context)
        elif self.training_target == 'noise':
            noise = self.model(x, t_tensor, background=background, context=context)
            x_recon = self.predict_start_from_noise(x, t=t, noise=noise)            
        elif self.training_target == 'v':
            v = self.model(x, t_tensor, background=background, context=context)
            x_recon = self.predict_start_from_v(x, t=t, v=v)
        
        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample(self, x, background, t, context=None, clip_denoised=False):
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, background = background, t=t, context=context, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)  # no noise when t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start
    
    @torch.no_grad()
    def ddim_sample(self, background, context=None, x_init=None, cloud_mask=None, clip_x_start=False, custom_timesteps=None, xinit_timesteps=None, eta=0, tau=0):
        batch, c, h, w = background.shape
        device = self.device
        total_timesteps = self.num_timesteps
        sampling_timesteps = custom_timesteps or self.num_timesteps
        
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        
        warped, context = self.flow(background)
        
        x = torch.randn(size=(batch, 1, h, w), device=self.device)
        
        if cloud_mask is not None:
            x_start_gt = x_init-warped
            known_mask = 1. - cloud_mask
            
            x_noisy_gt = self.q_sample(x_start_gt, time_pairs[0][0], noise=x)
            x = x * cloud_mask + x_noisy_gt * known_mask
        
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
           t_tensor = torch.full((batch,), time, dtype=torch.int64, device=device)
           
           if self.training_target == 'x0':
               x_start = self.model(x, t_tensor, background= torch.cat([background, warped],dim=1), context=context)
               x_start = maybe_clip(x_start)
               pred_noise = self.predict_noise_from_start(x, time, x_start)
           elif self.training_target == 'noise':
               pred_noise = self.model(x, t_tensor, background= torch.cat([background, warped],dim=1), context=context)
               x_start = self.predict_start_from_noise(x, time, pred_noise)
               x_start = maybe_clip(x_start)
           elif self.training_target == 'v':
               v = self.model(x, t_tensor, background= torch.cat([background, warped],dim=1), context=context)
               x_start = self.predict_start_from_v(x, time, v)
               x_start = maybe_clip(x_start)
               pred_noise = self.predict_noise_from_start(x, time, x_start)
           
           if time_next < 0:
               x = x_start
               
               if cloud_mask is not None:
                   if (xinit_timesteps is None) or (xinit_timesteps < time):
                       x = x * cloud_mask + x_start_gt * known_mask
               
               continue
           
            
           alpha = self.alphas_hat[time]
           alpha_next = self.alphas_hat[time_next]
           sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
           c = (1 - alpha_next - sigma ** 2).sqrt()
           noise = torch.randn_like(x)
           
           x = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
           
           if cloud_mask is not None:
               if (xinit_timesteps is None) or (xinit_timesteps < time):
                   x_noisy_gt = self.q_sample(x_start_gt, time)
                   x = x * cloud_mask + x_noisy_gt * known_mask
        
        sres = warped + x
        
        return sres, warped
        

    @torch.no_grad()
    def sample(self, background, x_init=None, cloud_mask=None, custom_timesteps=None, xinit_timesteps=None, tau=2, clip_denoised=False):
        
        b, c, h, w = background.shape
        #c=1
        timesteps = custom_timesteps or self.num_timesteps
        timestep_list = list(range(timesteps))
        
        warped, context = self.flow(background)
        
        x = torch.randn(size=(b, 1, h, w), device=self.device)
        
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_denoised else identity
        
        if cloud_mask is not None:
            x_start_gt = x_init-warped
            known_mask = 1. - cloud_mask
            
            x_noisy_gt = self.q_sample(x_start_gt, timestep_list[-1], noise=x)
            x = x * cloud_mask + x_noisy_gt * known_mask
        
        for t in tqdm(reversed(timestep_list), desc="Processing with x_init"):
            x, x_start = self.p_sample(x, torch.cat([background, warped],dim=1), t, context=context)
            
            if cloud_mask is not None:
                if (xinit_timesteps is None) or (xinit_timesteps < t):
                    x_noisy_gt = self.q_sample(x_start_gt, t)
                    x = x * cloud_mask + x_noisy_gt * known_mask
            
            x_start = maybe_clip(x_start)
            
        
        sres = warped + x
        
        return sres, warped

    def q_sample(self, x_start, t, noise=None):
        """
        Perform forward diffusion (noising) in a single step.
        This method returns x_t, which is x_0 noised for t timesteps.

        Args:
            x_start (torch.Tensor): Represents the original image (x_0).
            t (int): The timestep that measures the amount of noise to add.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        return self.sqrt_alphas_hat[t] * x_start + self.sqrt_one_minus_alphas_hat[t] * noise

    def train_loss(self, x, background,loss_weight=None, loss='mse', context_loss=None, aoi=None, *args, **kwargs):
        batch_size = x.shape[0]

        # Sample t uniformly
        t = np.random.randint(0, self.num_timesteps)
        
        # Generate white noise
        noise = torch.randn_like(x)
        
        warped, context = self.flow(background)
        
        if context_loss == 'mae':
            context_loss_tensor = F.l1_loss(warped, x, reduction='none')
        elif context_loss == 'mse':
            context_loss_tensor = F.mse_loss(warped, x, reduction='none')
        else:
            raise ValueError(f"Unsupported loss: {loss}")
        
        x_start = x-warped
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # Attempt to reconstruct white noise that was used in forward process
        t_tensor = torch.full((batch_size, ), t, dtype=torch.int64, device=self.device)
        losses = []
        
        background = torch.cat([background, warped], dim=1)
        
        if self.training_target == 'x0':
            pred = self.model(x_noisy, t_tensor, background=background, context=context)
            target = x
        elif self.training_target == 'noise':
            pred = self.model(x_noisy, t_tensor, background=background, context=context)
            target = noise
        elif self.training_target == 'v':
            target = self.predict_v(x_start=x_start, t=t, noise=noise)
            pred = self.model(x_noisy, t_tensor, background=background, context=context)
        
        if loss == 'mae':
            loss_tensor = F.l1_loss(pred, target, reduction='none')
        elif loss == 'mse':
            loss_tensor = F.mse_loss(pred, target, reduction='none')
        else:
            raise ValueError(f"Unsupported loss: {loss}")
        
        # 2. AOI 마스크 적용 (있을 경우)
        if aoi is not None:
            while aoi.dim() < pred.dim():
                aoi = aoi.unsqueeze(1)  # (N,1,H,W) 형태로 확장
            loss_tensor = loss_tensor * aoi
            if context_loss_tensor is not None:
                context_loss_tensor = context_loss_tensor * aoi
            valid_count = aoi.sum()
        
        # 3. Loss weight 적용
        if loss_weight is not None:
            loss_tensor = loss_tensor * loss_weight
            if context_loss_tensor is not None:
                context_loss_tensor = context_loss_tensor * loss_weight
        
        # 4. 평균 계산 (AOI or 전체)
        if aoi is not None:
            loss_tensor = loss_tensor.sum() / valid_count
            if context_loss_tensor is not None:
                context_loss_tensor = context_loss_tensor.sum() / valid_count
            else:
                context_loss_tensor = None
        else:
            loss_tensor = loss_tensor.mean()
            if context_loss_tensor is not None:
                context_loss_tensor = context_loss_tensor.mean()
        
        
        if context_loss is not None:
            return loss_tensor, context_loss_tensor
        else:
            return loss_tensor



import os

import numpy as np
import torch
from PIL import Image


def cosine_noise_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


def linear_noise_schedule(timesteps):
    """
    linear noise schedule.
    as proposed in https://arxiv.org/abs/2006.11239
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)


def uniform(timesteps):
    """
    Uniform noise schedule. Used in some experiments. Currently unused.
    """
    return np.ones(timesteps) * 1/timesteps


def custom_noise_schedule(timesteps, p):
    """
    Noise schedule as proposed in https://arxiv.org/abs/2206.00364
    """
    scale = 1000 / timesteps
    beta_min = scale * 0.0001
    beta_max = 1
    betas = [beta_max]
    for i in range(1, timesteps):
        beta_i = (beta_max ** (1/p) + (i/(timesteps-1))*(beta_min**(1/p)-beta_max**(1/p)))**p
        betas.append(beta_i)
    return np.clip(betas[::-1], a_min=0, a_max=0.999)


def to_torch(tensor):
    return torch.tensor(tensor, dtype=torch.float32)
