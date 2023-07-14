import abc


class Sampling(abc.ABC):
    @abc.abstractmethod
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def sample(self, hash_value, tables, train):
        pass
