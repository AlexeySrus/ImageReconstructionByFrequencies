import torch
from torch import nn, einsum
from einops import rearrange


class ComplexBatchNorm(nn.Module):
    def __init__(self, channels: int, *,
                    eps: float = 1e-5, momentum: float = 0.1,
                    affine: bool = True, track_running_stats: bool = True):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.scale = nn.Parameter(torch.ones(channels, dtype=torch.cfloat))
            self.shift = nn.Parameter(torch.zeros(channels, dtype=torch.cfloat))

        if self.track_running_stats:
            self.register_buffer('exp_mean', torch.zeros(channels, dtype=torch.cfloat))
            self.register_buffer('exp_var', torch.ones(channels, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor):
        x_shape = x.shape
        batch_size = x_shape[0]
        assert self.channels == x.shape[1]
        x = x.view(batch_size, self.channels, -1)
        if self.training or not self.track_running_stats:
            mean = x.mean(dim=[0, 2])
            mean_x2 = (x ** 2).mean(dim=[0, 2])
            var = mean_x2 - mean ** 2
            if self.training and self.track_running_stats:
                self.exp_mean = (1 - self.momentum) * self.exp_mean + self.momentum * mean
                self.exp_var = (1 - self.momentum) * self.exp_var + self.momentum * var
        else:
            mean = self.exp_mean
            var = self.exp_var

        x_norm = (x - mean.view(1, -1, 1)) / torch.sqrt(var + self.eps).view(1, -1, 1)
        
        if self.affine:
            x_norm = self.scale.view(1, -1, 1) * x_norm + self.shift.view(1, -1, 1)
        
        return x_norm.view(x_shape)


# helpers functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_rel_pos(n):
    pos = torch.meshgrid(torch.arange(n), torch.arange(n))
    pos = rearrange(torch.stack(pos), 'n i j -> (i j) n')  # [n*n, 2] pos[n] = (i, j)
    rel_pos = pos[None, :] - pos[:, None]                  # [n*n, n*n, 2] rel_pos[n, m] = (rel_i, rel_j)
    rel_pos += n - 1                                       # shift value range from [-n+1, n-1] to [0, 2n-2]
    return rel_pos

# lambda layer

class ComplexLambdaLayer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_k,
        n = None,
        r = None,
        heads = 4,
        dim_out = None,
        dim_u = 1):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.u = dim_u # intra-depth dimension
        self.heads = heads

        assert (dim_out % heads) == 0, 'values dimension must be divisible by number of heads for multi-head query'
        dim_v = dim_out // heads

        self.to_q = nn.Conv2d(dim, dim_k * heads, 1, bias = False, dtype=torch.cfloat)
        self.to_k = nn.Conv2d(dim, dim_k * dim_u, 1, bias = False, dtype=torch.cfloat)
        self.to_v = nn.Conv2d(dim, dim_v * dim_u, 1, bias = False, dtype=torch.cfloat)

        self.norm_q = ComplexBatchNorm(dim_k * heads)
        self.norm_v = ComplexBatchNorm(dim_v * dim_u)

        self.local_contexts = exists(r)
        if exists(r):
            assert (r % 2) == 1, 'Receptive kernel size should be odd'
            self.pos_conv = nn.Conv3d(dim_u, dim_k, (1, r, r), padding = (0, r // 2, r // 2), dtype=torch.cfloat)
        else:
            assert exists(n), 'You must specify the window size (n=h=w)'
            rel_lengths = 2 * n - 1
            self.rel_pos_emb = nn.Parameter(torch.randn(rel_lengths, rel_lengths, dim_k, dim_u).to(torch.cfloat))
            self.rel_pos = calc_rel_pos(n)

    def forward(self, x):
        b, c, hh, ww, u, h = *x.shape, self.u, self.heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.norm_q(q)
        v = self.norm_v(v)

        q = rearrange(q, 'b (h k) hh ww -> b h k (hh ww)', h = h)
        k = rearrange(k, 'b (u k) hh ww -> b u k (hh ww)', u = u)
        v = rearrange(v, 'b (u v) hh ww -> b u v (hh ww)', u = u)

        k = torch.abs(k).softmax(dim=-1).to(torch.cfloat)

        λc = einsum('b u k m, b u v m -> b k v', k, v)
        Yc = einsum('b h k n, b k v -> b h v n', q, λc)

        if self.local_contexts:
            v = rearrange(v, 'b u v (hh ww) -> b u v hh ww', hh = hh, ww = ww)
            λp = self.pos_conv(v)
            Yp = einsum('b h k n, b k v n -> b h v n', q, λp.flatten(3))
        else:
            n, m = self.rel_pos.unbind(dim = -1)
            rel_pos_emb = self.rel_pos_emb[n, m]
            λp = einsum('n m k u, b u v m -> b n k v', rel_pos_emb, v)
            Yp = einsum('b h k n, b n k v -> b h v n', q, λp)

        Y = Yc + Yp
        out = rearrange(Y, 'b h v (hh ww) -> b (h v) hh ww', hh = hh, ww = ww)
        return out


if __name__ == '__main__':
    print('AAAAAAAAAAAAAAA')
    # lambda_layer = ComplexLambdaLayer(
    #     dim = 32,       # channels going in
    #     dim_out = 32,   # channels out
    #     n = 64,         # size of the receptive window - max(height, width)
    #     dim_k = 16,     # key dimension
    #     heads = 4,      # number of heads, for multi-query
    #     dim_u = 1       # 'intra-depth' dimension
    # )
    lambda_layer = ComplexLambdaLayer(
        dim = 32,
        dim_out = 32,
        r = 23,         # the receptive field for relative positional encoding (23 x 23)
        dim_k = 16,
        heads = 4,
        dim_u = 4
    )

    t = torch.rand(1, 32, 128, 128).to(torch.cfloat)
    print('AAAAAAAAAAAAAAA')
    out = lambda_layer(t)
    print('AAAAAAAAAAAAAAA', out.shape)
