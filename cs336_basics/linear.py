from __future__ import annotations

import math

import torch
import torch.nn as nn


class Linear(nn.Module):
    """Linear layer without bias. Weight shape is (out_features, in_features), i.e. 𝑊 in y = x @ 𝑊ᵀ."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs: dict = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        sigma = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma,a= -3*sigma,b=3*sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y[..., o] = sum_i x[..., i] * W[o, i]  （等价于 x @ Wᵀ）
        return torch.einsum("...i,o i->...o", x, self.weight)
