import numpy as np
import torch
import math

try:
    from torch_utils.ops import conv2d_gradfix
    from torch_utils.ops import upfirdn2d
except:
    from .torch_utils.ops import conv2d_gradfix
    from .torch_utils.ops import upfirdn2d


def run_G(G, z, condition, style_mixing_prob, update_emas=False):
        c = G.calculate_embedding(t=None, condition=condition)
        ws = G.mapping(z, c, update_emas=update_emas)
        if style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        img = G.synthesis(ws, update_emas=update_emas)
        return img, ws


def run_AE_G(G, z, condition, style_mixing_prob, style_image=None, update_emas=False):
        c = G.calculate_embedding(t=None, condition=condition)
        encoded_style_img = None
        init_z = None
        if G.image_encoder is not None and style_image is not None:
            init_z, encoded_style_img = G.image_encoder(torch.concat([style_image, z], dim=1), emb=c)
        else:
            init_z, encoded_style_img = G.image_encoder(z, emb=c)

        ws = G.mapping(init_z, c, update_emas=update_emas)

        if style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                new_z = torch.randn_like(z)
                if G.image_encoder is not None and style_image is not None:
                    new_init_z, _ = G.image_encoder(torch.concat([style_image, new_z], dim=1), emb=c)
                else:
                    new_init_z, _ = G.image_encoder(new_z, emb=c)

                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = G.mapping(new_init_z, c, update_emas=False)[:, cutoff:]

        pred_img = G.synthesis(ws, style_features=encoded_style_img, update_emas=update_emas)

        if G.sum_with_style:
             pred_img = pred_img + style_image
             
        return pred_img, ws


def run_UNET_G(G, z, condition, style_mixing_prob, style_image=None):
        c = G.calculate_embedding(t=None, condition=condition)

        ws = G.mapping(z, c)
        if style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = G.mapping(torch.randn_like(z), c)[:, cutoff:]

        pred_img = G.synthesis(ws, style_image=style_image, emb=c)
             
        return pred_img, ws


def run_D(D, img, condition, blur_sigma=0, augment_pipe=None, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if augment_pipe is not None:
            img = augment_pipe(img)
        logits = D(img, condition=condition, update_emas=update_emas)
        return logits


def run_time_D(D, img, t, condition, blur_sigma=0, augment_pipe=None, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if augment_pipe is not None:
            img = augment_pipe(img)
        logits = D(img, t=t, condition=condition, update_emas=update_emas)
        return logits


def d_r1_loss(real_pred, real_img):
    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
        grad_real, = torch.autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True, only_inputs=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01, pl_no_weight_grad=False):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )

    with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(pl_no_weight_grad):
        grad, = torch.autograd.grad(
            outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True, only_inputs=True
        )

    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def styled_g_path_regularize(fake_img, latents, style_image, mean_path_length, decay=0.01, pl_no_weight_grad=False):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )

    with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(pl_no_weight_grad):
        latent_grad, style_grad, = torch.autograd.grad(
            outputs=(fake_img * noise).sum(), inputs=(latents, style_image), create_graph=True, only_inputs=True
        )

    path_lengths = torch.sqrt(latent_grad.pow(2).sum(2).mean(1)) + torch.sqrt(style_grad.pow(2).sum(2).mean(1).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def g_path_regularize_without_decorators(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )

    grad, = torch.autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True, only_inputs=True
    )

    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def styled_g_path_regularize_without_decorators(fake_img, latents, style_image, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )

    latent_grad, style_grad, = torch.autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=(latents, style_image), create_graph=True, only_inputs=True
    )

    latent_grad_path = torch.sqrt(latent_grad.pow(2).sum(2).mean(1).mean(1))
    style_grad_path = torch.sqrt(style_grad.pow(2).sum(2).mean(1).mean(1))

    path_lengths = latent_grad_path + style_grad_path

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths
