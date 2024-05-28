# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 adapter_scalar="1.0",):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        # nn.init.zeros_(self.down_proj.weight)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=False, residual=None):
        # add_residual=True leads to bad performance
        residual = x if residual is None else residual

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        up = self.up_proj(down)

        up = up * self.scale

        if add_residual:
            output = up + residual
        else:
            output = up

        return output