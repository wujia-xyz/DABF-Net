"""
KBNet Utility Modules
Copied and adapted from BasicSR implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormFunction(torch.autograd.Function):
    """Custom LayerNorm with manual forward/backward for 2D images"""

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        weight, bias, y = weight.contiguous(), bias.contiguous(), y.contiguous()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), \
               grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    """2D Layer Normalization for image tensors"""

    def __init__(self, channels, eps=1e-6, requires_grad=True):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels), requires_grad=requires_grad))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels), requires_grad=requires_grad))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    """Gated Linear Unit: split channels and multiply"""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class KBAFunction(torch.autograd.Function):
    """
    Knowledge-Based Attention (KBA) Function
    Dynamic convolution with learnable kernel bank
    """

    @staticmethod
    def forward(ctx, x, att, selfk, selfg, selfb, selfw):
        """
        Args:
            x: input feature [B, C, H, W]
            att: attention map [B, nset, H, W]
            selfk: kernel size (int)
            selfg: group number (int)
            selfb: bias parameters [1, nset, C]
            selfw: weight parameters [1, nset, C*C//g*k^2]
        """
        B, nset, H, W = att.shape
        KK = selfk ** 2
        selfc = x.shape[1]

        att = att.reshape(B, nset, H * W).transpose(-2, -1)  # [B, HW, nset]

        ctx.selfk, ctx.selfg, ctx.selfc, ctx.KK, ctx.nset = selfk, selfg, selfc, KK, nset
        ctx.x, ctx.att, ctx.selfb, ctx.selfw = x, att, selfb, selfw

        # Generate dynamic weights and bias
        bias = att @ selfb  # [B, HW, C]
        attk = att @ selfw  # [B, HW, C*C//g*k^2]

        # Unfold input to extract patches
        uf = torch.nn.functional.unfold(x, kernel_size=selfk, padding=selfk // 2)  # [B, C*k^2, HW]

        # Reshape for group-wise convolution
        uf = uf.reshape(B, selfg, selfc // selfg * KK, H * W).permute(0, 3, 1, 2)  # [B, HW, g, C//g*k^2]
        attk = attk.reshape(B, H * W, selfg, selfc // selfg, selfc // selfg * KK)  # [B, HW, g, C//g, C//g*k^2]

        # Apply dynamic convolution
        x = attk @ uf.unsqueeze(-1)  # [B, HW, g, C//g, 1]
        del attk, uf

        x = x.squeeze(-1).reshape(B, H * W, selfc) + bias
        x = x.transpose(-1, -2).reshape(B, selfc, H, W)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, att, selfb, selfw = ctx.x, ctx.att, ctx.selfb, ctx.selfw
        selfk, selfg, selfc, KK, nset = ctx.selfk, ctx.selfg, ctx.selfc, ctx.KK, ctx.nset

        B, selfc, H, W = grad_output.size()

        # Gradient for bias
        dbias = grad_output.reshape(B, selfc, H * W).transpose(-1, -2)  # [B, HW, C]
        dselfb = att.transpose(-2, -1) @ dbias  # [B, nset, C]
        datt = dbias @ selfb.transpose(-2, -1)  # [B, HW, nset]

        # Reconstruct forward pass intermediate values
        attk = att @ selfw
        uf = F.unfold(x, kernel_size=selfk, padding=selfk // 2)
        uf = uf.reshape(B, selfg, selfc // selfg * KK, H * W).permute(0, 3, 1, 2)
        attk = attk.reshape(B, H * W, selfg, selfc // selfg, selfc // selfg * KK)

        # Gradient for attention kernel weights
        dx = dbias.view(B, H * W, selfg, selfc // selfg, 1)
        dattk = dx @ uf.view(B, H * W, selfg, 1, selfc // selfg * KK)
        duf = attk.transpose(-2, -1) @ dx
        del attk, uf

        # Gradient for weight parameters
        dattk = dattk.view(B, H * W, -1)
        datt += dattk @ selfw.transpose(-2, -1)
        dselfw = att.transpose(-2, -1) @ dattk  # [B, nset, C*C//g*k^2]

        # Gradient for input
        duf = duf.permute(0, 2, 3, 4, 1).view(B, -1, H * W)
        dx = F.fold(duf, output_size=(H, W), kernel_size=selfk, padding=selfk // 2)

        datt = datt.transpose(-1, -2).view(B, nset, H, W)

        return dx, datt, None, None, dselfb, dselfw


if __name__ == '__main__':
    # Test KBA function
    B, C, H, W = 2, 64, 32, 32
    nset = 32
    k = 3
    gc = 4

    x = torch.randn(B, C, H, W, requires_grad=True)
    att = torch.randn(B, nset, H, W, requires_grad=True)

    g = C // gc
    w = torch.randn(1, nset, C * C // g * k ** 2, requires_grad=True)
    b = torch.randn(1, nset, C, requires_grad=True)

    out = KBAFunction.apply(x, att, k, g, b, w)
    print(f"KBA output shape: {out.shape}")

    loss = out.sum()
    loss.backward()
    print("Backward pass successful!")
