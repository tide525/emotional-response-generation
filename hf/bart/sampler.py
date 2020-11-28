import math
import random
from collections import Counter

import torch
from torch.utils.data import Sampler
from torch._six import int_classes as _int_classes


class MultitaskBatchSampler(Sampler):
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


class CurriculumSampler(Sampler):
    def __init__(self, data_source, step, num_steps):
        self.data_source = data_source

        self.step = step
        self.num_steps = num_steps

    def __iter__(self):
        num_samples = len(self.data_source)
        weights = torch.pow(
            min(0.1, self.step / self.num_steps),
            torch.arange(num_samples, dtype=torch.double) / num_samples
        )

        rand_tensor = torch.multinomial(weights, num_samples, replacement=True)
        return iter(rand_tensor.tolist())

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


class CompetenceSampler(Sampler):
    def __init__(self, data_source, difficulties, step, num_steps, init_competence):
        self.data_source = data_source

        self.difficulties = torch.as_tensor(difficulties, dtype=torch.double)
        assert len(self.data_source) == len(self.difficulties)

        self.step = step
        self.num_steps = num_steps

        self.init_competence = init_competence

    def __iter__(self):
        t, T = self.step, self.num_steps
        c_0 = self.init_competence

        i2d = self.difficulties

        counter = Counter(i2d.tolist())
        j2d = torch.as_tensor(list(counter), dtype=torch.double)
        j2f = torch.as_tensor(list(counter.values()), dtype=torch.int64)

        j2cdf = j2f.cumsum(dim=0, dtype=torch.double) / j2f.sum()

        c = min(1, math.sqrt(t * (1 - c_0 ** 2) / T + c_0 ** 2))

        j_max = ((j2cdf <= c).sum() - 1).item()
        assert j_max >= 0
        d_max = j2d[j_max]

        i_max = ((i2d <= d_max).sum() - 1).item()

        yield from torch.randint(high=i_max + 1, size=(len(self.data_source),), dtype=torch.int64).tolist()

    def __len__(self):
        return len(self.data_source)
