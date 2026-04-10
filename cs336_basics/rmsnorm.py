from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization（无 bias），与常见 Transformer 实现一致。"""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_f = x.float() * torch.rsqrt(v + self.eps)
        return (self.weight * x_f).to(dtype=x.dtype)


def rmsnorm_forward(
    in_features: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Functional RMSNorm：在最后一维上按平方均值归一化，再乘以 `weight`。"""
    v = in_features.float().pow(2).mean(dim=-1, keepdim=True)
    x_f = in_features.float() * torch.rsqrt(v + float(eps))
    return (weight * x_f).to(dtype=in_features.dtype)
