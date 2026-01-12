# diffusion + encoder + decoder
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn


def Normalize(
    in_channels: int,
    num_groups: int = 32,
):
    return nn.GroupNorm(
        num_groups=num_groups,
        num_channels=in_channels,
        eps=1e-6,
        affine=True,
    )


def SiLU(x: torch.Tensor):
    return x * torch.sigmoid(x)


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        with_conv: bool,
    ):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=0,
            )


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        temb_channels: int = 512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)

        self.norm2 = Normalize(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.dropout = nn.Dropout(dropout)

        # apply shortcut
        if self.in_channels != out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ):
        h = x
        h = self.norm1(x)
        h = SiLU(h)
        h = self.conv1(h)

        # b, out, h, w
        if temb is not None:
            h = h + self.temb_proj(SiLU(temb))[:, :, None, None]

        h = self.norm2(h)
        h = SiLU(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
    ):
        super().__init__()

        self.norm = SiLU(in_channels)
        self.q = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.k = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.v = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.proj_out = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        h = x
        h = self.norm(h)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        v = v.reshape(b, c, h*w)

        w = torch.bmm(q, k)
        w = w * (int(c) ** (-0.5))  # apply scaling factor
        w = nn.Softmax(w, dim=2)  # shape: b, hw, hw

        w = w.permute(0, 2, 1)  # b, hw: for k, hw: for q
        h = torch.bmm(v, w)
        h = h.reshape(b, c, h, w)

        h = self.proj_out(h)

        return x + h


def make_attn(
    in_channels: int,
    attn_type: str = "vanilla",
):
    assert attn_type in ["vanilla", "none"], f'attn_type {attn_type} unknown'
    print(
        f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,  # num_channels after first conv1
        out_ch: int,
        ch_mult: Tuple[int],  # multiplier for ch
        num_res_blocks: int,
        attn_resolutions: Tuple[int],
        dropout: float,
        resamp_with_conv: bool,
        in_channels: int,
        resolution: int,
        z_channels: int,
        double_z: bool,
        attn_type: str = "vanilla",
        **ignore_kwargs,
    ):
        super().__init__()

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = nn.Conv2d(
            in_channels,
            out_channels=self.ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        curr_res = resolution
        in_ch_mult = (1, ) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        # 1 -> 1, 1 -> 2, 2 -> 4, 4 -> 8, ...
        # self.down: block, attn, downsample
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(
                        make_attn(block_in, block_out, attn_type=attn_type))

            down = nn.Module()
            down.block = block
            down.attn = attn

            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            block_in,
            block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            block_in,
            block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        # return the mean and sigma
        self.conv_out = nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn > 0):
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = SiLU(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
