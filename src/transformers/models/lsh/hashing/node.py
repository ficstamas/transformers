from collections import OrderedDict
from typing import Type

import torch
import torch.nn as nn
import numpy as np

from .hash.base import HashFunction
from .sampling.base import Sampling
from .sampling.vanilla import VanillaSamplingStrided


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

        self.num_tables = num_tables
        self.num_hashes = num_hashes
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
        size = np.max(np.array([[len(bucket) for bucket in table] for table in self.tables]))
        tensor = -torch.ones((self.num_tables, 2**self.num_hashes, size), dtype=torch.long)
        for i in range(len(self.tables)):
            table = self.tables[i]
            for j in range(len(table)):
                bucket = table[j]
                for k in range(len(bucket)):
                    value = bucket[k]
                    tensor[i, j, k] = value
        return tensor

    def set_extra_state(self, state):
        tensor = state
        self.tables = [[[] for j in range(2**self.num_hashes)] for i in range(self.num_tables)]
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                for k in range(tensor.shape[2]):
                    value = tensor[i, j, k].item()
                    if value == -1:
                        break
                    self.tables[i][j].append(value)


class LSHLinearStrided(nn.Module):
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

        self.num_tables = num_tables
        self.num_hashes = num_hashes
        self.hash = hash_module(input_dim, num_tables, num_hashes)
        self.sampling = VanillaSamplingStrided(**sampling_config)
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.tables = torch.nn.Parameter(
            data=torch.zeros((num_tables, 2**num_hashes+1), dtype=torch.long), requires_grad=False
        )
        self.buckets = torch.nn.Parameter(
            data=torch.zeros((num_tables, output_dim), dtype=torch.long), requires_grad=False
        )
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
                ind = self.sampling.sample(input_vector, self.tables, self.buckets, self.training)

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
        tab = {i: {j: [] for j in range(2**self.num_hashes)} for i in range(self.num_tables)}

        hashed_weights = self.hash(self.linear.weight)
        for neuron, T in enumerate(hashed_weights):
            for table, hash_ in enumerate(T):
                tab[table][hash_.item()].append(neuron)

        for table_id, buckets in tab.items():
            for bucket_id, bucket in buckets.items():
                start_ind = self.tables[table_id, bucket_id]
                bucket_tensor = torch.tensor(bucket, requires_grad=False, dtype=torch.long)
                self.tables[table_id, bucket_id+1] = start_ind+len(bucket)
                self.buckets[table_id, start_ind:self.tables[table_id, bucket_id+1]] = bucket_tensor[:]
