import os

import torch
from torch.utils.data import Dataset


class MultitaskDataset(Dataset):
    def __init__(
        self,
        tasks,
        task_dirs,
        tokenizer,
        data_dir,
        type_path,
        max_len=512
    ):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self.labels = []
        self.tasks = []

        self._build(tasks, task_dirs, data_dir, type_path)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]['input_ids'].squeeze()
        source_mask = self.inputs[index]['attention_mask'].squeeze()  # might need to squeeze
        target_ids = self.targets[index]['input_ids'].squeeze()
        target_mask = self.targets[index]['attention_mask'].squeeze()  # might need to squeeze

        label = self.labels[index]
        task = self.tasks[index]

        return {
            'source_ids': source_ids,
            'source_mask': source_mask,
            'target_ids': target_ids,
            'target_mask': target_mask,
            'label': label,
            'task': task
        }

    def _build(self, tasks, task_dirs, data_dir, type_path):
        for task, task_dir in zip(tasks, task_dirs):
            input_path = os.path.join(
                data_dir,
                task_dir,
                type_path + '.source'
            )
            with open(input_path) as f:
                for line in f:
                    input_ = line.rstrip()
                    # tokenize inputs
                    tokenized_inputs = self.tokenizer.batch_encode_plus(
                        [input_],
                        max_length=self.max_len,
                        pad_to_max_length=True,
                        return_tensors='pt',
                        truncation=True
                    )
                    self.inputs.append(tokenized_inputs)

            target_path = os.path.join(
                data_dir,
                task_dir,
                type_path + '.target'
            )
            with open(target_path) as f:
                for line in f:
                    target = ' '
                    label = -1

                    if task == 'response':
                        target = line.rstrip()
                    else:
                        label = int(line.rstrip())

                    # tokenize targets
                    tokenized_targets = self.tokenizer.batch_encode_plus(
                        [target],
                        max_length=self.max_len,
                        pad_to_max_length=True,
                        return_tensors='pt',
                        truncation=True
                    )
                    tensor_labels = torch.tensor([label])
                    
                    self.targets.append(tokenized_targets)
                    self.labels.append(tensor_labels)

                    self.tasks.append(task)
                    