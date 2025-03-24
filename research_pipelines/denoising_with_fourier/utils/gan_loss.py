from typing import Tuple

import torch

from styleganae.stylegan_v2_models import ConditionalDiscriminator
from styleganae.gan_utils import d_r1_loss, run_time_D


def d_logistic_loss(real_pred, fake_pred):
    real_loss = torch.nn.functional.softplus(-real_pred)
    fake_loss = torch.nn.functional.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def g_nonsaturating_loss(fake_pred, use_reduction: bool = True):
    loss = torch.nn.functional.softplus(-fake_pred)

    if use_reduction:
        return loss.mean()
    return loss


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


class StyleGANv2Loss(torch.nn.Module):
    def __init__(self, image_size: int, lr: float = 0.0002, betas=(0, 0.99), regularize: bool = True,
                 channels: int = 3,
                 time_embedder=None, time_embedder_kwargs={},
                 cond_embedder: torch.nn.Module = None, cond_embedder_kwargs={},
                 discriminator_kwargs={}):
        super(StyleGANv2Loss, self).__init__()
        self.d_reg_every = 16
        d_reg_ratio = self.d_reg_every / (self.d_reg_every + 1)
        self.r1 = 10
        self.step = 1
        self.regularize = regularize

        self.discriminator = ConditionalDiscriminator(
            img_resolution=image_size, img_channels=channels, encode_embedding= False,
            time_embedder=time_embedder, time_embedder_kwargs=time_embedder_kwargs,
            cond_embedder=cond_embedder, cond_embedder_kwargs=cond_embedder_kwargs,
            **discriminator_kwargs
        )

        self.optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr * d_reg_ratio,
            betas=(betas[0] ** d_reg_ratio, betas[1] ** d_reg_ratio)
        )

        self.r1_loss_val = 0
    
    def change_requeres_grad(self, stat: bool):
        requires_grad(self.discriminator, stat)

    def all_zero_grad(self):
        self.discriminator.zero_grad()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, t=None, condition=None) -> Tuple[torch.Tensor, float, float]:
        self.change_requeres_grad(True)
        
        fake_pred = run_time_D(self.discriminator, pred.detach(), t=t, condition=condition)
        real_pred = run_time_D(self.discriminator, target, t=t, condition=condition)
        d_loss = d_logistic_loss(real_pred, fake_pred)
        
        self.all_zero_grad()

        d_loss.backward()
        self.optimizer.step()
        
        d_regularize = self.regularize and (self.step % self.d_reg_every == 0)
        if d_regularize:
            prev_grad_status = target.requires_grad
            target.requires_grad = True
                
            real_pred = run_time_D(self.discriminator, target, t=t, condition=condition)
            r1_loss = d_r1_loss(real_pred, target)
        
            self.all_zero_grad()

            (self.r1 / 2 * r1_loss * self.d_reg_every + 0 * real_pred[0]).backward()
        
            self.optimizer.step()
            target.requires_grad = prev_grad_status

            self.r1_loss_val = r1_loss.item()
        
        self.change_requeres_grad(False)
        
        fake_pred = run_time_D(self.discriminator, pred, t=t, condition=condition)
        g_loss = g_nonsaturating_loss(fake_pred)

        self.step += 1
        if self.step > 10000000:
            self.step = 1

        return g_loss
