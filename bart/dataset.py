import os

import torch
from torch.utils.data import Dataset


class TaskDataset(Dataset):
    def __init__(self, task, tokenizer, data_dir, type_path, max_len=512):
        assert task in ['emotion', 'response', 'sentiment']

        self.task = task
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.type_path = type_path
        self.max_len = max_len

        self.inputs = []
        self.targets = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]['input_ids'].squeeze()
        source_mask = self.inputs[index]['attention_mask'].squeeze()  # might need to squeeze

        target_ids = self.targets[index]['input_ids'].squeeze()
        target_mask = self.targets[index]['attention_mask'].squeeze()  # might need to squeeze

        return {
            'task': self.task,
            'source_ids': source_ids,
            'source_mask': source_mask,
            'target_ids': target_ids,
            'target_mask': target_mask,
        }
    
    def _build(self):
        input_path = os.path.join(self.data_dir, self.type_path + '.source')
        with open(input_path) as f:
            for line in f:
                # tokenize inputs
                tokenized_inputs = self.tokenizer.batch_encode_plus(
                    [line.rstrip()],
                    max_length=self.max_len,
                    pad_to_max_length=True,
                    return_tensors='pt',
                    truncation=True
                )
                self.inputs.append(tokenized_inputs)

        target_path = os.path.join(self.data_dir, self.type_path + '.target')
        with open(target_path) as f:
            for line in f:
                if self.task == 'response':
                    # tokenize targets
                    tokenized_targets = self.tokenizer.batch_encode_plus(
                        [line.rstrip()],
                        max_length=self.max_len,
                        pad_to_max_length=True,
                        return_tensors='pt',
                        truncation=True
                    )
                else:
                    tokenized_targets = {
                        'input_ids': torch.tensor([[int(line)]]),
                        'attention_mask': torch.tensor([[1]])
                    }
                self.targets.append(tokenized_targets)
