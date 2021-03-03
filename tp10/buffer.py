import numpy as np
import torch

from torch.utils.data import IterableDataset

class Memory():
    def __init__(self, mem_size, items=5, replace=True):
        self.memory = []
        for i in range(items):
            self.memory.append([])
        self.mem_size = mem_size
        self.size = 0
        self.num_items = items
        self.replace = replace
    
    def add(self, transition, replacement_idx=None):
        if self.size >= self.mem_size:
            if self.replace:
                if replacement_idx is None:
                    replacement_idx = np.random.choice(np.arange(self.mem_size))
                for i,item in enumerate(transition):
                    self.memory[i][replacement_idx] = item
            else:
                raise MemoryError(f'Buffer cannot store more than {self.mem_size} events.')
        else:
            for i,item in enumerate(transition):
                self.memory[i].append(item)
            self.size += 1

    def sample(self, n, device="cpu"):
        if n > self.size:
            raise ValueError(f'Buffer only has {self.size} stored transitions')
        sample_idx = torch.randperm(self.size)[:n]
        return [
            torch.stack([self.memory[i][idx] for idx in sample_idx], 0).to(device) for i in range(self.num_items)
        ]

    def get_indices(self, indices, device="cpu"):
        return [
            torch.stack([self.memory[i][idx] for idx in indices], 0).to(device) for i in range(self.num_items)
        ]

    def as_dataset(self):
        return BufferDataset(self.sample(self.size))

    def empty(self):
        self.memory = []
        for i in range(self.num_items):
            self.memory.append([])
        self.size = 0

    def get(self, item, idx):
        return self.memory[item][idx]

class BufferDataset(IterableDataset):
    """Dataset to iterate over transition tuples"""
    def __init__(self, data):
        super().__init__()
        self.data = data
    
    def __iter__(self):
        return iter(zip(*self.data))

    def __len__(self):
        return self.data[0].size(0)

class MultiAgentMemory():
    def __init__(self, mem_size, n, items=5, replace=True):
        self.memories = [
            Memory(mem_size, items, replace) for i in range(n)
        ]
        self.mem_size = mem_size
        self.size = 0
        self.num_items = items
        self.n = n
        self.replace = replace

    def add(self, transitions):
        assert len(transitions) == self.n, f"{self.n} transitions must be passed, received {len(transitions)}"
        replacement_idx = np.random.choice(np.arange(self.mem_size)) if self.size > self.mem_size else None
        for j, transition in enumerate(transitions):
            self.memories[j].add(transition, replacement_idx)
        self.size = self.memories[0].size

    def sample(self, n, device="cpu"):
        """Returns a tuple of lists of items"""
        if n > self.size:
            raise ValueError(f'Buffer only has {self.size} stored transitions')
        sample_idx = torch.randperm(self.size)[:n]
        return zip(
            *[self.memories[i].get_indices(sample_idx, device=device) for i in range(self.n)]
        )