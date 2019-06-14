import numpy as np
import operator

from collections import deque
from abc import abstractmethod, ABCMeta


class BaseMemory(object):

    __metaclass__ = ABCMeta

    def __init__(self, size=50000,
                 keys=["states", "actions", "rewards", "next_states", "dones"]):
        self.size = size
        self.keys = keys
        self.memory = {key: deque(maxlen=self.size) for key in keys}

    def __len__(self):
        return len(self.memory[self.keys[0]])

    def _get_batch(self, indices):
        """Get data by indices."""
        return tuple([np.vstack(operator.itemgetter(*indices)(self.memory[key])) for key in self.keys])

    def _get_single(self, index):
        """Get single item by index."""
        return tuple([self.memory[key][index] for key in self.keys])

    def get(self, index):
        """Get item(s) by index."""
        if isinstance(index, int):
            return self._get_single(index)
        else:
            return self._get_batch(index)

    def store(self, *args):
        """Store an experience tuple in memory. Take care that arguments are provided in the
        same order as the :code:`keys` argument when initializing.
        """ 
        for i, key in enumerate(self.keys):
            self.memory[key].append(args[i])

    def store_batch(self, *args):
        for i, key in enumerate(self.keys):
            self.memory[key].extend(list(args[i]))

    @abstractmethod
    def sample(self, batch_size):
        """Sample a batch of experiences. This depends on specific memory type
        (e.g., prioritized vs random)."""

    @abstractmethod
    def get_config(self):
        """Get configuration parameters as a dictionary."""

class ReplayBuffer(BaseMemory):

    name = "ReplayBuffer"

    def __init__(self, size=50000, keys=["states", "actions", "rewards", "next_states", "terminal"]):
        super().__init__(size=size, keys=keys)

    def sample(self, batch_size):
        index = np.random.randint(0, self.__len__(), size=batch_size)
        return self.get(index)

    def get_config(self):
        return {
            "name": self.name,
            "size": self.size
        }
