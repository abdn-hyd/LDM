from typing import Optional, List
import torch
import torch.nn as nn

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        ddconfig,
        embed_dim: int,
        ckpt_path: Optional[str] = None,
        ignore_keys: List[str] = [],
        image_key: str = "image",
    ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quant_conv = nn.Conv2d(
            2 * ddconfig["z_channels"],
            2 * embed_dim,
            1,
        )
        self.post_quant_conv = nn.Conv2d(
            ddconfig["z_channels"],
            embed_dim,
            1,
        )

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)

    # load the checkpoint
    def init_from_ckpt(
        self,
        path: str,
        ignore_keys: List[Optional[str]] = [],
    ):
        sd = torch.load(path, map_location="cpu",
                        weights_only=False)["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(
        self,
        x: torch.Tensor,
    ):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(
        self,
        z: torch.Tensor,
    ):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(
        self,
        input: torch.Tensor,
        sample_posterior: bool = True,
    ):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec
