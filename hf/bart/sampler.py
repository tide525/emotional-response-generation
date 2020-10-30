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


# https://pytorch-lightning.readthedocs.io/en/stable/trainer.html#reload-dataloaders-every-epoch
class CurriculumSampler(Sampler):
    def __init__(self, data_source, weights):
        self.data_source = data_source
        self.weights = torch.as_tensor(weights, dtype=torch.double)

    def __iter__(self):
        num_samples = len(self.data_source)

        num_groups = len(self.weights)
        nums_tensor = torch.as_tensor(
            [(num_samples + i) // num_groups for i in range(num_groups)],
            dtype=torch.int64
        )
        cums_tensor = torch.cumsum(
            torch.cat((torch.zeros(1, dtype=torch.int64), nums_tensor), 0),
            dim=0
        )

        rand_tensor = torch.multinomial(self.weights, num_samples, True)
        for group in rand_tensor.tolist():
            yield torch.randint(
                cums_tensor[group],
                high=cums_tensor[group+1],
                size=(1,),
                dtype=torch.int64
            ).item()

    def __len__(self):
        return len(self.data_source)


class CurriculumBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, weights):
        self.data_source = data_source
        self.batch_size = batch_size
        self.weights = torch.as_tensor(weights, dtype=torch.double)

    def __iter__(self):
        num_groups = len(self.weights)
        num_samples = len(self.data_source)

        num_tensor = torch.as_tensor(
            [
                (num_samples + group_idx) // num_groups
                for group_idx in range(num_groups)
            ],
            dtype=torch.int64
        )
        cum_tensor = torch.cumsum(
            torch.cat(
                (torch.zeros(1, dtype=torch.int64), num_tensor),
                dim=0
            ),
            dim=0
        )

        num_batches = (num_samples - 1) // self.batch_size + 1
        group_tensor = torch.multinomial(self.weights, num_batches, True)

        for batch_idx in range(num_batches):
            num_sample = min(
                num_samples - batch_idx * self.batch_size,
                self.batch_size
            )

            group_idx = group_tensor[batch_idx]
            sample_tensor = torch.randint(
                cum_tensor[group_idx],
                cum_tensor[group_idx+1],
                size=(num_sample,),
                dtype=torch.int64
            )
            yield sample_tensor.tolist()

    def __len__(self):
        return (len(self.data_source) - 1) // self.batch_size + 1


class TaskCurriculumSampler(Sampler):
    def __init__(self, data_source, batch_size, tasks, weights):
        self.data_source = data_source
        self.batch_size = batch_size

        self.tasks = tasks
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        assert len(self.tasks) == len(self.weights)

    def __iter__(self):
        num_samples = len(self.data_source)
        num_tasks = len(self.tasks)

        num_tensor = torch.zeros(num_tasks, dtype=torch.int64)
        for data in self.data_source:
            task_idx = self.tasks.index(data['task'])
            num_tensor[task_idx] += 1
        cum_tensor = torch.cumsum(
            torch.cat((torch.zeros(1, dtype=torch.int64), num_tensor), dim=0),
            dim=0
        )

        num_batches = (num_samples - 1) // self.batch_size + 1
        task_rand_tensor = torch.multinomial(self.weights, num_batches, True)

        for batch_idx in range(num_batches):
            num_sample = min(
                num_samples - batch_idx * self.batch_size,
                self.batch_size
            )

            task_idx = task_rand_tensor[batch_idx]
            sample_rand_tensor = torch.randint(
                cum_tensor[task_idx],
                cum_tensor[task_idx + 1],
                size=(num_sample,),
                dtype=torch.int64
            )
            yield sample_rand_tensor.tolist()

    def __len__(self):
        return (len(self.data_source) - 1) // self.batch_size + 1


class TaskCurriculumBatchSampler(Sampler):
    def __init__(
        self,
        data_source,
        batch_size,
        tasks,
        task_weights,
        group_weights
    ):
        self.data_source = data_source
        self.batch_size = batch_size

        self.tasks = tasks
        self.task_weights = torch.as_tensor(task_weights, dtype=torch.double)
        assert len(self.tasks) == len(self.task_weights)

        self.group_weights = torch.as_tensor(
            group_weights,
            dtype=torch.double
        )

    def __iter__(self):
        num_samples = len(self.data_source)
        num_tasks = len(self.tasks)

        task_num_tensor = torch.zeros(num_tasks, dtype=torch.int64)
        for data in self.data_source:
            task_idx = self.tasks.index(data['task'])
            task_num_tensor[task_idx] += 1
        task_cum_tensor = torch.cumsum(
            torch.cat(
                (torch.zeros(1, dtype=torch.int64), task_num_tensor),
                dim=0
            ),
            dim=0
        )

        num_groups = len(self.group_weights)

        group_num_tensor = torch.as_tensor(
            [
                [
                    (num_samples + group_idx) // num_groups
                    for group_idx in range(num_groups)
                ]
                for num_samples in task_num_tensor
            ],
            dtype=torch.int64
        )
        group_cum_tensor = (
            torch.cumsum(
                torch.cat(
                    (
                        torch.zeros((num_tasks, 1), dtype=torch.int64),
                        group_num_tensor
                    ),
                    dim=1
                ),
                dim=1
            )
            + task_cum_tensor[:-1].unsqueeze(1)
        )

        num_batches = (num_samples - 1) // self.batch_size + 1

        task_rand_tensor = torch.multinomial(self.task_weights, num_batches, True)
        group_rand_tensor = torch.multinomial(self.group_weights, num_batches, True)

        for batch_idx in range(num_batches):
            num_sample = min(
                num_samples - batch_idx * self.batch_size,
                self.batch_size
            )

            task_idx = task_rand_tensor[batch_idx]
            group_idx = group_rand_tensor[batch_idx]
            sample_rand_tensor = torch.randint(
                group_cum_tensor[task_idx][group_idx],
                group_cum_tensor[task_idx][group_idx + 1],
                size=(num_sample,),
                dtype=torch.int64
            )
            yield sample_rand_tensor.tolist()

    def __len__(self):
        return (len(self.data_source) - 1) // self.batch_size + 1
