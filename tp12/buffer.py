import numpy as np
import torch

from torch.utils.data import IterableDataset

class Memory():
    def __init__(self, mem_size, items=5, replace=True, weighted=False, replace_type="random"):
        self.memory = []
        for i in range(items):
            self.memory.append([])
        self.mem_size = mem_size
        self.size = 0
        self.num_items = items
        self.replace = replace
        self.weighted = weighted
        if weighted:
            self.weights = []
            self.p = None
        if replace_type not in ["random", "fifo"]:
            raise ValueError("`replace_type` must be `fifo` or `random`")
        self.replace_type = replace_type
    
    def add(self, transition, replacement_idx=None, weight=None):
        if (self.weighted) and (weight is None):
            print("No weight provided for transition in prioritized buffer. Weight defaults to 0")
            weight = 0
        if self.size >= self.mem_size:
            if self.replace:
                if (replacement_idx is not None) or self.replace_type == "random":
                    # Given replacement IDs or random replacement
                    if replacement_idx is None:
                        replacement_idx = np.random.choice(np.arange(self.mem_size))                        
                    for i,item in enumerate(transition):
                        self.memory[i][replacement_idx] = item
                    if self.weighted:
                        self.weights[replacement_idx] = weight
                else: # FIFO replacement
                    for i,item in enumerate(transition):
                        self.memory[i].pop(0)
                        self.memory[i].append(item)
                    if self.weighted:
                        self.weights.pop(0)
                        self.weights.append(weight)
                        self.normalize_weights()
            else:
                raise MemoryError(f'Buffer cannot store more than {self.mem_size} events.')
        else:
            for i,item in enumerate(transition):
                self.memory[i].append(item)
            if self.weighted:
                self.weights.append(weight)
                self.normalize_weights()
            self.size += 1

    def sample(self, n, return_indices=False):
        if n > self.size:
            raise ValueError(f'Buffer only has {self.size} stored transitions')
        if self.weighted:
            sample_idx = np.random.choice(self.size, size=n, replace=False, p=self.p)
        else:
            sample_idx = torch.randperm(self.size)[:n]
        
        samples = [
            torch.stack([self.memory[i][idx] for idx in sample_idx], 0) for i in range(self.num_items)
        ]
        if return_indices:
            return samples, sample_idx
        return samples

    def get_indices(self, indices):
        return [
            torch.stack([self.memory[i][idx] for idx in indices], 0) for i in range(self.num_items)
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
    
    def update_item(self, item, idx, new_v):
        self.memory[item][idx] = new_v

    def get_weight(self, idx):
        return self.weights[idx]

    def update_weight(self, idx, new_p):
        self.weights[idx] = new_p
        self.normalize_weights()

    def normalize_weights(self):
        self.p = np.exp(self.weights) / np.sum(np.exp(self.weights))


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

    def sample(self, n):
        if n > self.size:
            raise ValueError(f'Buffer only has {self.size} stored transitions')
        sample_idx = torch.randperm(self.size)[:n]
        return [
            torch.stack([
                torch.stack([memory.memory[i][idx] for memory in self.memories], 0) for idx in sample_idx
            ])
            for i in range(self.num_items)
        ]
