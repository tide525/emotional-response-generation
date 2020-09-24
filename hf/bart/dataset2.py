import os

import torch
from torch.utils.data import Dataset

data_dict = {
    'emotion_response': os.path.join('dd', 'dial_emo', 'wo_ne')
}


class MultitaskDataset2(Dataset):
    def __init__(self, tasks, tokenizer, data_dir, type_path, max_len=512):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self.tasks = []

        self._build(tasks, data_dir, type_path)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        task = ''  # self.tasks[index]

        source_ids = self.inputs[index]['input_ids'].squeeze(0)
        target_ids = [
            self.targets[index][0]['input_ids'].squeeze(0),
            self.targets[index][1]['input_ids'].squeeze(0)
        ]

        # might need to squeeze
        src_mask = self.inputs[index]['attention_mask'].squeeze(0)
        # might need to squeeze
        target_mask = [
            self.targets[index][0]['attention_mask'].squeeze(0),
            self.targets[index][1]['attention_mask'].squeeze(0)
        ]

        return {
            'task': task,
            'source_ids': source_ids,
            'source_mask': src_mask,
            'target_ids': target_ids,
            'target_mask': target_mask
        }

    def _build(self, tasks, data_dir, type_path):
        input_path = os.path.join(
            data_dir,
            data_dict['emotion_response'],
            type_path + '.source'
        )
        with open(input_path) as f:
            for line in f:
                input_ = line.strip()
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
            data_dict['emotion_response'],
            type_path + '.target'
        )
        with open(target_path) as f:
            for line in f:
                target, label = line.strip().rsplit('\t', maxsplit=1)
                # tokenize targets
                tokenized_targets = self.tokenizer.batch_encode_plus(
                    [target],
                    max_length=self.max_len,
                    pad_to_max_length=True,
                    return_tensors='pt',
                    truncation=True
                )
                label = int(label) - 1
                tokenized_labels = {
                    'input_ids': torch.tensor([[label]]),
                    'attention_mask': torch.tensor([[1]])
                }
                self.targets.append([tokenized_labels, tokenized_targets])
