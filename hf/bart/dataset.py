import os

import torch
from torch.utils.data import Dataset


data_dict = dict(
    emotion='uniemo',
    response='crawled',
    sentiment='sst2',
)


class MultitaskDataset(Dataset):
    def __init__(self, tasks, tokenizer, data_dir, type_path, max_len=512):
        self.path_dict = {
            task: os.path.join(data_dir, data_dict[task], type_path + '.tsv')
            for task in tasks
        }

        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self.tasks = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        task = self.tasks[index]

        source_ids = self.inputs[index]['input_ids'].squeeze(0)
        target_ids = self.targets[index]['input_ids'].squeeze(0)

        # might need to squeeze
        src_mask = self.inputs[index]['attention_mask'].squeeze(0)
        # might need to squeeze
        target_mask = self.targets[index]['attention_mask'].squeeze(0)

        return {
            'task': task,
            'source_ids': source_ids,
            'source_mask': src_mask,
            'target_ids': target_ids,
            'target_mask': target_mask,
        }

    def _build(self):
        for task, path in self.path_dict.items():
            with open(path) as f:
                for line in f:
                    input_, target = line.strip().split('\t')

                    tokenized_inputs = self.tokenizer.batch_encode_plus(
                        [input_],
                        max_length=self.max_len,
                        pad_to_max_length=True,
                        return_tensors='pt',
                        truncation=True,
                    )

                    if task == 'response':
                        tokenized_targets = self.tokenizer.batch_encode_plus(
                            [target],
                            max_length=self.max_len,
                            pad_to_max_length=True,
                            return_tensors='pt',
                            truncation=True,
                        )
                    elif task in ['emotion', 'sentiment']:
                        tokenized_targets = {
                            'input_ids': torch.tensor([[int(target)]]),
                            'attention_mask': torch.tensor([[1]]),
                        }

                    self.inputs.append(tokenized_inputs)
                    self.targets.append(tokenized_targets)

                    self.tasks.append(task)
