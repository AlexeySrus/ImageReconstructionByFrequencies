import math
from typing import Any, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.optim.lr_scheduler as lrs


def make_optimizer(optimizer, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': 0.9}
    elif optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1E-8
        }
    elif optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': 1E-8}

    kwargs['lr'] = 1E-3
    kwargs['weight_decay'] = 0
    
    return optimizer_function(trainable, **kwargs)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=(kernel_size // 2),
                     bias=bias)


def default_conv1(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=(kernel_size // 2),
                     bias=bias,
                     groups=1)


#def shuffle_channel()


# 使用哈尔 haar 小波变换来实现二维离散小波
def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


# 使用哈尔 haar 小波变换来实现二维离散小波
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r**2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


# 二维离散小波
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


# 逆向二维离散小波
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


# 使用 mean shift 算法来修正偏移均值
# https://blog.csdn.net/google19890102/article/details/51030884
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False
        if sign == -1:
            self.create_graph = False
            self.volatile = True


class MeanShift2(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift2, self).__init__(4, 4, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(4).view(4, 4, 1, 1)
        self.weight.data.div_(std.view(4, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False
        if sign == -1:
            self.volatile = True


# 通用基本卷积单元，可以是空洞卷积
class BasicBlock(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 bias=False,
                 bn=False,
                 act=nn.ReLU(True)):

        m = [
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      padding=(kernel_size // 2),
                      stride=stride,
                      bias=bias)
        ]

        if bn:
            m.append(nn.BatchNorm2d(out_channels))

        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


# 带ReLu激活函数的普通单层卷积
class BBlock(nn.Module):
    def __init__(self,
                 conv,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 bn=False,
                 act=nn.ReLU(True),
                 res_scale=1):

        super(BBlock, self).__init__()

        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))

        if bn:
            m.append(nn.BatchNorm2d(out_channels))

        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x).mul(self.res_scale)
        return x


# 双层残差卷积单元，最末层不使用激活函数
class ResBlock(nn.Module):
    def __init__(self,
                 conv,
                 n_feat,
                 kernel_size,
                 bias=True,
                 bn=False,
                 act=nn.ReLU(True),
                 res_scale=1):

        super(ResBlock, self).__init__()

        m = []

        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)

        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


# 卷积组合单元（含4层卷积）
class Block(nn.Module):
    def __init__(self,
                 conv,
                 n_feat,
                 kernel_size,
                 bias=True,
                 bn=False,
                 act=nn.ReLU(True),
                 res_scale=1):

        super(Block, self).__init__()

        m = []

        for i in range(4):
            # 维持卷积层的维度不变
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            # 第一层卷积层添加激活层
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        # 控制卷积单元的作用系数
        res = self.body(x).mul(self.res_scale)
        # res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        # 使用位运算来判断，scale 是否是2^n。
        # 生成2倍的超分辨重建的结果
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feat))
                if act:
                    m.append(act())
        # 生成三倍的超分辨的结果
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if act:
                m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class Discriminator(nn.Module):
    def __init__(self, n_colors: int = 3, gan_type='GAN', patch_size=384):
        super(Discriminator, self).__init__()

        in_channels = 3
        out_channels = 64
        depth = 7
        #bn = not gan_type == 'WGAN_GP'
        bn = True
        act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        m_features = [
            BasicBlock(n_colors, out_channels, 3, bn=bn, act=act)
        ]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(BasicBlock(
                in_channels, out_channels, 3, stride=stride, bn=bn, act=act
            ))

        self.features = nn.Sequential(*m_features)

        patch_size = patch_size // (2**((depth + 1) // 2))
        m_classifier = [
            nn.Linear(out_channels * patch_size**2, 1024),
            act,
            nn.Linear(1024, 1)
        ]
        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features.view(features.size(0), -1))

        return output


class Adversarial(nn.Module):
    def __init__(self, gan_k: int = 1, gan_type: str = 'GAN', image_size: int = 512):
        super(Adversarial, self).__init__()
        self.gan_type = gan_type
        self.gan_k = gan_k
        self.discriminator = Discriminator(gan_type=gan_type, patch_size=image_size)
        if gan_type != 'WGAN_GP':
            self.optimizer = make_optimizer('ADAM', self.discriminator)
        else:
            self.optimizer = optim.Adam(
                self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-8, lr=1e-5
            )

    def forward(self, fake, real):
        fake_detach = fake.detach()

        self.loss = 0
        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            d_fake = self.discriminator(fake_detach)
            d_real = self.discriminator(real)
            if self.gan_type == 'GAN':
                label_fake = torch.zeros_like(d_fake)
                label_real = torch.ones_like(d_real)
                loss_d \
                    = F.binary_cross_entropy_with_logits(d_fake, label_fake) \
                    + F.binary_cross_entropy_with_logits(d_real, label_real)
            elif self.gan_type.find('WGAN') >= 0:
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand_like(fake).view(-1, 1, 1, 1)
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty

            # Discriminator update
            self.loss += loss_d.item()
            loss_d.backward()
            self.optimizer.step()

            if self.gan_type == 'WGAN':
                for p in self.discriminator.parameters():
                    p.data.clamp_(-1, 1)

        self.loss /= self.gan_k

        d_fake_for_g = self.discriminator(fake)
        if self.gan_type == 'GAN':
            loss_g = F.binary_cross_entropy_with_logits(
                d_fake_for_g, label_real
            )
        elif self.gan_type.find('WGAN') >= 0:
            loss_g = -d_fake_for_g.mean()

        # Generator loss
        return loss_g
    
    def state_dict(self, *args, **kwargs):
        state_discriminator = self.discriminator.state_dict(*args, **kwargs)
        state_optimizer = self.optimizer.state_dict()

        return {
            'optimizer': state_discriminator,
            'descriminator': state_optimizer
        }
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        self.discriminator.load_state_dict(state_dict['descriminator'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
               
# Some references
# https://github.com/kuc2477/pytorch-wgan-gp/blob/master/model.py
# OR
# https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py


if __name__ == '__main__':
    adv_loss = Adversarial()
    t1 = torch.rand(1, 3, 512, 512, requires_grad=True)
    t2 = torch.rand(1, 3, 512, 512, requires_grad=True)
    loss = adv_loss(t1, t2)
    print(loss.item())
