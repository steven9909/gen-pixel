import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from einops.layers.torch import Rearrange
from typing import Optional
from einops import rearrange, repeat, einsum
from functools import partial


# Following the block architecture from Imagen
class ChanRMSNorm(nn.Module):
    def __init__(self, dim: int):
        """Channel-wise RMS Normalization

        Args:
            dim (int): Channel dimension of the input tensor
        """
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Tensor: Output tensor of shape (batch_size, channels, height, width)
        """
        return F.normalize(x, dim=1) * self.scale * self.gamma


class CrossAttention(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        """Cross Attention Layer

        Args:
            dim_in (int): Channel dimension of the input.
            dim_out (int): Channel dimension of the output.
        """
        super().__init__()


class Block(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, norm=True):
        """Block for UNet

        Args:
            dim_in (int): Channel dimension of the input.
            dim_out (int): Channel dimension of the output.
            norm (bool, optional): Whether to use normalization layer. Defaults to True.
        """
        super().__init__()
        if norm:
            self.norm = ChanRMSNorm(dim_in)
        else:
            self.norm = nn.Identity()
        self.act = nn.SiLU()
        self.project = nn.Conv2d(dim_in, dim_out, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass

        Args:
            x (Tensor): Input tensor of shape (batch_size, dim_in, height, width)

        Returns:
            Tensor: Output tensor of shape (batch_size, dim_out, height, width)
        """
        x = self.norm(x)
        x = self.act(x)
        return self.project(x)


class SinusoidPE(nn.Module):
    def __init__(self, out_dim: int):
        """Sinusodial Positional Embedding

        Args:
            out_dim (int): Dimension of the output embedding
        """
        super().__init__()
        self.out_dim = out_dim

        assert self.out_dim % 2 == 0, "Output dimension of SinusoidalPE must be even."

    def forward(self, time: Tensor) -> Tensor:
        """Forward pass

        Args:
            time (Tensor): Input tensor of shape (batch_size)

        Returns:
            Tensor: Sinusoidal positional embedding of shape (batch_size, out_dim)
        """
        device = time.device
        half_dim = self.out_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )  # (half_dim,)
        embeddings = time[:, None] * embeddings[None, :]  # (batch_size, half_dim)
        embeddings = torch.cat(
            (embeddings.sin(), embeddings.cos()), dim=-1
        )  # (batch_size, out_dim)
        return embeddings


class UpsampleBlock(nn.Module):
    def __init__(
        self,
        dim_in: int,
        scale_factor: int = 2,
        dim_out: Optional[int] = None,
        mode: str = "nearest",
    ):
        """Creates an upsampling block using the specified mode.

        Args:
            dim_in (int): Channel dimension of the input.
            scale_factor (int): Scaling factor for upsampling. Defaults to 2.
            dim_out (Optional[int], optional): Channel dimension of the output. If set to None, dim_out becomes dim_in. Defaults to None.
            mode (str, optional): Upsampling mode. Defaults to nearest interpolation.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode=mode),
            nn.Conv2d(
                dim_in,
                dim_out if dim_out else dim_in,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass
        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Tensor: Output tensor of shape (batch_size, channels, height * scale_factor, width * scale_factor)
        """
        return self.block(x)


class DownsampleBlock(nn.Module):
    def __init__(
        self, dim_in: int, scale_factor: int = 2, dim_out: Optional[int] = None
    ):
        """Creates an downsampling block

        Args:
            dim_in (int): Channel dimension of the input.
            scale_factor (int, optional): Scaling factor for downsampling. Defaults to 2.
            dim_out (Optional[int], optional): Channel dimension of the output. If set to None, dim_out becomes dim_in. Defaults to None.
        """
        super().__init__()
        self.block = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (c p1 p2) h w", p1=scale_factor, p2=scale_factor
            ),
            nn.Conv2d(
                dim_in * scale_factor * scale_factor,
                dim_out if dim_out else dim_in,
                kernel_size=1,
                stride=1,
            ),
        )
        self.scale_factor = scale_factor

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass
        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Tensor: Output tensor of shape (batch_size, channels, height // scale_factor, width // scale_factor)
        """
        _, _, h, w = x.shape
        assert (
            h % self.scale_factor == 0 and w % self.scale_factor == 0
        ), f"Input shape {x.shape} is not divisible by scale factor {self.scale_factor}."
        return self.block(x)


class CrossAttention(nn.Module):
    def __init__(
        self, dim: int, context_dim: int, heads: int, head_dim: int, scale: int = 8
    ):
        """Cross Attention Layer

        Args:
            dim (int): Channel dimension of the input
            context_dim (int): Channel dimension of the context
            heads (int): Number of attention heads
            head_dim (int): Dimension of each attention head
            scale (int): Scaling factor for attention scores
        """
        super().__init__()
        self.scale = scale
        self.heads = heads
        inner_dim = head_dim * heads

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.null_kv = nn.Parameter(torch.randn((2, head_dim)))

        self.norm_input = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim)

        self.q_scale = nn.Parameter(torch.ones(head_dim))
        self.k_scale = nn.Parameter(torch.ones(head_dim))

        self.scale = scale

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False), nn.LayerNorm(dim)
        )

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        """Forward function of Cross Attention

        Args:
            x (Tensor): Input tensor of shape (batch_size, (height * width), channel)
            context (Tensor): Context tensor of shape (batch_size, text_dim, context_dim)

        Returns:
            Tensor: Output tensor of shape (batch_size, (height * width), channel)
        """
        b = x.shape[0]
        x = self.norm_input(x)
        context = self.norm_context(context)

        q, k, v = self.to_q(x), *self.to_kv(context).chunk(2, dim=-1)  #

        q = rearrange(q, "b c (h d) -> b h c d", h=self.heads)
        k = rearrange(k, "b c (h d) -> b h c d", h=self.heads)
        v = rearrange(v, "b c (h d) -> b h c d", h=self.heads)

        nulls = [
            repeat(t, "c -> b h 1 c", h=self.heads, b=b)
            for t in torch.unbind(self.null_kv)
        ]
        nk = nulls[0]
        nv = nulls[1]

        k = torch.cat((k, nk), dim=-2)
        v = torch.cat((v, nv), dim=-2)

        q = F.normalize(q, dim=-1) * self.q_scale
        k = F.normalize(k, dim=-1) * self.k_scale

        attn_weights = einsum(q, k, "b h i d, b h k d -> b h i k") * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)

        attn = einsum(attn_weights, v, "b h i k, b h k d -> b h i d")
        attn = rearrange(attn, "b h i d -> b i (h d)")

        return self.to_out(attn)


if __name__ == "__main__":
    block = CrossAttention(dim=8, context_dim=8, heads=8, head_dim=64, scale=8)

    x = torch.randn(1, 16, 8)
    context = torch.randn(1, 16, 8)

    print(block(x, context).shape)
