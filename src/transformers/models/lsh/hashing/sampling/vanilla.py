import numpy as np
import torch

from .base import Sampling


class VanillaSampling(Sampling):
    """
    Vanilla Sampling algorithm:
    1. Iterate over the tables (T) in random order
    2. Select the bucket (T_i^{h_i}) according to the hash function (h_i) from that table (T_i)
    3. The selected bucket will populate the number of active neurons (A)
    4. Stop and return if |A| > num_target_neurons or if we visited every table
    """

    def __init__(self, sampling_num_target_neurons, **kwargs):
        self.num_target_neurons = sampling_num_target_neurons

    def sample(self, hash_value, tables, train=True):
        # look up tables in random order
        num_tables = len(tables)
        table_ids = np.arange(num_tables)
        if train:
            np.random.shuffle(table_ids)

        neurons = set()  # set of active neurons
        for i in table_ids:
            bucket = tables[i][hash_value[i].item()]  # retrieve bucket
            # if the number of active neurons would be higher than the target
            if self.num_target_neurons != -1 and len(neurons) + len(bucket) > self.num_target_neurons:
                # add the minimal subset to the set
                rest = self.num_target_neurons - len(neurons)
                neurons = neurons.union(bucket[:rest])
                break
            else:
                neurons = neurons.union(bucket)

        return torch.tensor(list(neurons), dtype=torch.long, device=hash_value.device)


class VanillaSamplingStrided:
    """
    Vanilla Sampling algorithm:
    1. Iterate over the tables (T) in random order
    2. Select the bucket (T_i^{h_i}) according to the hash function (h_i) from that table (T_i)
    3. The selected bucket will populate the number of active neurons (A)
    4. Stop and return if |A| > num_target_neurons or if we visited every table
    """

    def __init__(self, sampling_num_target_neurons, **kwargs):
        self.num_target_neurons = sampling_num_target_neurons

    def sample(self, hash_value, tables, buckets, train=True):
        # look up tables in random order
        num_tables = len(tables)
        table_ids = np.arange(num_tables)
        if train:
            np.random.shuffle(table_ids)

        neurons = []  # set of active neurons
        # from_ = tables[table_ids][:, hash_value[table_ids]]
        # to_ = tables[table_ids][:, hash_value[table_ids]+1]
        for i in table_ids:
            hash_ = hash_value[i]
            # retrieve bucket
            bucket = tables[i]
            from_, to_ = bucket[hash_], bucket[hash_+1]
            active_neurons = buckets[i, from_:to_]
            neurons.append(active_neurons)

        return torch.cat(neurons).unique(sorted=False)[:self.num_target_neurons]
