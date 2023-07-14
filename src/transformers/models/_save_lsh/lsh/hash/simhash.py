import abc

import torch
import torch.nn as nn

from .base import HashFunction


class SimHash(nn.Module, HashFunction):
    def __init__(self, input_dim, num_tables, num_hashes):
        super().__init__()
        self.num_hashes = num_hashes
        self.num_tables = num_tables
        # signed projection matrix
        self.projection = torch.nn.Parameter(
            data=torch.sign(torch.normal(0, 0.02, (input_dim, num_hashes * num_tables))), requires_grad=False
        )

    def forward(self, x):
        # project weights
        x = x @ self.projection

        # reshape to (..., num_tables, num_hashes)
        shape = x.shape[:-1] + (self.num_tables, self.num_hashes)
        x = x.view(*shape)

        # binarize hash functions
        x = torch.masked_fill(x, x >= 0, 0)
        x = torch.masked_fill(x, x < 0, 1)

        # decode binary representation e.g. a tensor of [0, 1, 1] -> 3
        x = self.binary(x, self.num_hashes)
        return x

    @staticmethod
    def binary(x, bits):
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
        return torch.sum(mask * x, -1).long()
