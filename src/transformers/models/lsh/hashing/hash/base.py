import abc


class HashFunction(abc.ABC):
    @abc.abstractmethod
    def __init__(self, input_dim, num_tables, num_hashes):
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def forward(self, x):
        pass
