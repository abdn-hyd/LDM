from typing import Optional, Tuple
import torch
import numpy as np


class DiagonalGaussianDistribution:
    def __init__(
        self,
        parameters: torch.Tensor,  # b, z_c, h, w
        deterministic: bool = False,
    ):
        super().__init__()
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 ** self.logvar)
        self.var = torch.exp(self.logvar)

        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean).to(device=self.parameters.device)

    def sample(self):
        # reparameterization trick
        x = self.mean + self.std * \
            torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    # regularization (penalty) term
    def kl(
        self,
        # other distribution
        other: Optional[DiagonalGaussianDistribution] = None,
    ):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                """
                1. KL(P|Q) = E[log(P) - log(Q)], where P -> N(mu, sig^2), Q -> (0, 1)
                2. log(P) = -0.5 * (log(2 * PI) + log(sig^2) + (x - mu)^2 / sig^2),
                3. log(Q) = -0.5 * (log(2 * PI) + x^2)
                use 2 and 3 in 1, we can derive:
                4. KL(P|Q) = E[-0.5 * (log(sig^2) - (x - mu)^2 / sig^2 + x^2] = -0.5 * (log(sig^2) + 1 / sig^2 * E[(x - mu)^2] - E[x^2])
                5. we have Var(x) = E[(x - mu)^2] = E[x^2] - E[x]^2
                6. KL(P|Q) = 0.5 * (sig^2 + mu^2 - log(sig^2) - 1.0)
                """
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var
                    - 1.0 - self.logvar, dim=[1, 2, 3]
                )
            else:
                """
                notice: random variable x only follows the distribution of P.
                1. KL(P|Q) = E[log(P) - log(Q)], where P -> N(mu, sig^2), Q -> (0, 1)
                2. log(P) = -0.5 * (log(2 * PI) + log(sig1^2) + (x - mu1)^2 / sig1^2),
                3. log(Q) = -0.5 * (log(2 * PI) + log(sig2^2) + (x - mu2)^2 / sig2^2)
                use 2 and 3 in 1, we can derive:
                4. KL(P|Q) = 0.5 * E[log(sig2^2) - log(sig1^2) + (x - mu2)^2 / sig2^2 - (x - mu1)^2 / sig1^2]
                5. we have Var(x) = E[(x - mu)^2] = E[x^2] - E[x]^2, x - mu2 = (x - mu1) + (mu1 - mu2), E[(x - mu1)] = 0
                6. KL(P|Q) = 0.5 * (log(sig2^2) - log(sig1^2) - 1.0 + (sig1^2 + (mu1 - mu2)^2) / sig2^2)
                """
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / orther.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3]
                )

    # nll: Negative Log-Likelihood, reconstruction loss
    def nll(
        self,
        sample: torch.Tensor,
        dims: Tuple[int] = [1, 2, 3],
    ):
        """
        log(P) = 0.5 * (log(2 * PI) + log(sig^2) + (x - mu)^2 / sig^2),
        """
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar +
            torch.pow(sample - self.mean, 2) / self.var,
            dim=dims
        )

    def mode(self):
        return self.mean
