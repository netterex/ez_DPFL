import torch
import numpy as np


def cal_sensitivity(clip, data_size):
    return 2 * clip / data_size


def add_noise_on_grads(model, data_size, clip, device, epsilon, noisy_grads):
    sensitivity = cal_sensitivity(clip, data_size)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.grad is not None:
                noise = torch.tensor(np.random.laplace(loc=0, scale=sensitivity / epsilon, size=param.grad.shape),
                                         device=device)
                noise = noise.to(param.grad.dtype)
                # 累加带噪声的梯度
                noisy_grads[name] += noise


def add_noise_on_param(model, data_size, total_size, clip, device, epsilon):
    sensitivity = cal_sensitivity(clip, data_size)
    contribution = data_size / total_size
    if contribution > 0.1:
        epsilon = epsilon * (10 + contribution)    # 0.501
    elif contribution < 0.1 and contribution > 0.05:
        epsilon = epsilon * (10 + contribution)     # 0.050342
    else:
        epsilon = epsilon * (10 + contribution)     # 0.050041
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.tensor(np.random.laplace(loc=0, scale=sensitivity / epsilon, size=param.grad.shape),
                                         device=device)
                noise = noise.to(param.dtype)
                # 将梯度添加到参数上
                param.add_(noise)


def per_sample_clip(model, clipping, norm):
    grad_samples = [x.grad_sample for x in model.parameters()]
    per_param_norms = [
        g.reshape(len(g), -1).norm(norm, dim=-1) for g in grad_samples
    ]
    per_sample_norms = torch.stack(per_param_norms, dim=1).norm(norm, dim=1)
    per_sample_clip_factor = (
        torch.div(clipping, (per_sample_norms + 1e-6))
    ).clamp(max=1.0)
    for grad in grad_samples:
        factor = per_sample_clip_factor.reshape(per_sample_clip_factor.shape + (1,) * (grad.dim() - 1))
        grad.detach().mul_(factor.to(grad.device))
    # average per sample gradient after clipping and set back gradient
    for param in model.parameters():
        param.grad = param.grad_sample.detach().mean(dim=0)


def clip_gradients(model, clip):
    per_sample_clip(model, clip, norm=1)
