import torch.utils.data
from torch.utils.data import ConcatDataset
from torch.utils.data import Sampler, BatchSampler, RandomSampler
from typing import Optional, Iterator
import numpy as np


class ratio_sampler(Sampler[int]):
    def __init__(self, data_source: ConcatDataset, replacement: bool = False, num_samples: Optional[int] = None, ratio=1, generator=None):
        '''
        params: ratio, means the ratio of the labeled and unlabeled data in a batch.
        >>> ratio_sampler(data_source)
        '''
        super().__init__(data_source)
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.cumulative_sizes = data_source.cumulative_sizes
        self.ratio = ratio
        self.generator = generator
        self.l = []

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        n = len(self.data_source)
        n1, n2 = [len(i) for i in self.data_source.datasets]
        assert n == n1 + n2, "data_source_datasets'length is more than two."

        l1 = np.random.choice(np.arange(n1), size=n1, replace=False)
        l2 = np.random.choice(np.arange(n1, n1 + n2), size=n1, replace=True)

        if self.ratio == 1:
            l = np.array([list(i) for i in zip(l1, l2)]).reshape(-1).tolist()
        elif self.ratio == 2:
            l = np.array([list(i) for i in zip(l1, l2, l2)]).reshape(-1).tolist()
        else:
            print('self.ratio should be one of [1, 2], now reset to ratio=1')
            l = np.array([list(i) for i in zip(l1, l2)]).reshape(-1).tolist()
        return len(l)

    def __iter__(self) -> Iterator[int]:
        if self.generator is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
            np.random.seed(int(str(seed)[-3:]))
        else:
            raise ValueError
        n = len(self.data_source)
        n1, n2 = [len(i) for i in self.data_source.datasets]
        assert n == n1 + n2, "data_source_datasets'length is more than two."

        if self.replacement:
            raise ValueError("don't wanna self.replacement=True")
        else:
            l1 = np.random.choice(np.arange(n1), size=n1, replace=False)
            np.random.seed(1)
            l2 = np.random.choice(np.arange(n1, n1+n2), size=n1, replace=True)
            np.random.seed(2)
            l22 = np.random.choice(np.arange(n1, n1+n2), size=n1, replace=True)

            if self.ratio == 1:
                l = np.array([list(i) for i in zip(l1, l2)]).reshape(-1).tolist()
            elif self.ratio == 2:
                l = np.array([list(i) for i in zip(l1, l2, l22)]).reshape(-1).tolist()
            else:
                print('self.ratio should be one of [1, 2], now reset to ratio=1')
                l = np.array([list(i) for i in zip(l1, l2)]).reshape(-1).tolist()

            yield from l

    def __len__(self) -> int:
        return self.num_samples

