import os

import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=512):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build(data_dir, type_path)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]['input_ids'].squeeze()
        source_mask = self.inputs[index]['attention_mask'].squeeze()  # might need to squeeze

        target_ids = self.targets[index]['input_ids'].squeeze(0)
        target_mask = self.targets[index]['attention_mask'].squeeze(0)  # might need to squeeze

        return {
            'source_ids': source_ids,
            'source_mask': source_mask,
            'target_ids': target_ids,
            'target_mask': target_mask
        }

    def _build(self, data_dir, type_path):
        input_path = os.path.join(data_dir, 'tec', type_path + '.source')
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

        target_path = os.path.join(data_dir, 'tec', type_path + '.target')
        with open(target_path) as f:
            for line in f:
                # tokenize targets
                tokenized_targets = {
                    'input_ids': torch.LongTensor([[int(line.rstrip())]]),
                    'attention_mask': torch.LongTensor([[1]])
                }
                self.targets.append(tokenized_targets)
