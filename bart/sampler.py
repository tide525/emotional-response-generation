import random

import torch
from torch.utils.data import Sampler
from torch._six import int_classes as _int_classes


class MultitaskSampler(Sampler):
    r"""Samples elements for multi-task learning.

    Args:
        data_source (Dataset): dataset to sample from
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """

    def __init__(self, data_source, batch_size, drop_last):
        if (
            not isinstance(batch_size, _int_classes)
            or isinstance(batch_size, bool) or batch_size <= 0
        ):
            raise ValueError(
                "batch_size should be a positive integer value, "
                "but got batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                "drop_last should be a boolean value, "
                "but got drop_last={}".format(drop_last)
            )
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch_dict = {}
        for idx in torch.randperm(len(self.data_source)).tolist():
            task = self.data_source[idx]["task"]
            if task not in batch_dict:
                batch_dict[task] = []
            batch_dict[task].append(idx)
            for batch in batch_dict.values():
                if len(batch) == self.batch_size:
                    yield batch
                    batch_dict[task] = []
        for batch in batch_dict.values():
            if len(batch) > 0 and not self.drop_last:
                yield batch

    def __len__(self):
        if self.drop_last:
            num_samples_dict = {}
            for data in self.data_source:
                task = data["task"]
                if task not in num_samples_dict:
                    num_samples_dict[task] = 0
                num_samples_dict[task] += 1
            return sum(
                num_samples // self.batch_size for num_samples
                in num_samples_dict.values()
            )
        else:
            return (
                (len(self.data_source) + self.batch_size - 1)
                // self.batch_size
            )
            