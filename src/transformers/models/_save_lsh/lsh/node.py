from collections import OrderedDict
from typing import Type

import numpy as np
import torch
import torch.nn as nn

from lsh.hash.base import HashFunction
from lsh.sampling.base import Sampling


class LSHLinear(nn.Module):
    """
    Locality Sensitive Hashing based linear layer
    It is based on Maximum Inner Product Search (MIPS) which is trying to sample vectors in order to maximize their
    inner product.
    From an NN perspective it can be seen as calculating only neurons with high activations.
    https://arxiv.org/abs/1405.5869
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_tables: int,
        num_hashes: int,
        hash_module: Type[HashFunction],
        sampling_module: Type[Sampling],
        sampling_config: dict,
        bias=True,
    ):
        super().__init__()
        self.num_tables, self.num_hashes = num_tables, num_hashes

        self.hash = hash_module(input_dim, num_tables, num_hashes)
        self.sampling = sampling_module(**sampling_config)
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.tables = [[[] for j in range(2**num_hashes)] for i in range(num_tables)]
        self.bias = bias
        # init tables
        self.rehash()

    def forward(self, x: torch.Tensor):
        batched = True
        if len(x.shape) < 3:
            batched = False
            x = torch.unsqueeze(x, dim=0)

        # hash the neurons
        hash_ = self.hash(x)  # bucket ids for each table (..., num_vectors, num_tables)
        shape = x.shape[:-1] + (self.linear.weight.shape[0],)
        output: torch.Tensor = torch.zeros(shape, device=x.device)  # a "sparse" matrix which will store the result

        # sparse matmul
        for b in range(shape[0]):
            for i in range(len(output)):
                # unique set of active neurons, sampled according to the sampling strategy
                input_vector = hash_[b, i]
                ind = self.sampling.sample(input_vector, self.tables, self.training)

                lhs = x[b, i, :]  # input vector
                rhs = self.linear.weight.data[ind]  # weights for the active neurons

                res = torch.sum(lhs * rhs, dim=1)  # matmul with the reduced set
                if self.bias:  # add bias
                    bias = self.linear.bias.data[ind]
                    res += bias

                # saving results
                output[b, i, ind] = res

        if not batched:
            output = torch.squeeze(output, dim=0)

        return output

    def rehash(self):
        self.tables = [[[] for j in range(2**self.num_hashes)] for i in range(self.num_tables)]
        hashed_weights = self.hash(self.linear.weight)
        for neuron, T in enumerate(hashed_weights):
            for table, hash_ in enumerate(T):
                self.tables[table][hash_.item()].append(neuron)

    def get_extra_state(self) -> OrderedDict:
        state = OrderedDict()
        state["tables"] = self.tables
        return state

    def set_extra_state(self, state: OrderedDict):
        self.tables = state["tables"]


# python model_training.py --directory "temp/" --accelerator "gpu" --batch_size 2
