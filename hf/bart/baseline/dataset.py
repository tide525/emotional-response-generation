import os

import torch
from torch.utils.data import Dataset


class ResponseDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=512):
        self.path = os.path.join(data_dir, type_path + '.tsv')

        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]['input_ids'].squeeze()
        target_ids = self.targets[index]['input_ids'].squeeze()

        # might need to squeeze
        src_mask = self.inputs[index]['attention_mask'].squeeze()
        # might need to squeeze
        target_mask = self.targets[index]['attention_mask'].squeeze()

        return {
            'source_ids': source_ids,
            'source_mask': src_mask,
            'target_ids': target_ids,
            'target_mask': target_mask
        }

    def _build(self):
        with open(self.path) as f:
            for line in f:
                input_, target = line.strip().split('\t')

                # input_ = input_ + ' </s>'
                # target = target + ' </s>'

                # tokenize inputs
                tokenized_inputs = self.tokenizer.batch_encode_plus(
                    [input_],
                    max_length=self.max_len,
                    pad_to_max_length=True,
                    return_tensors='pt',
                    truncation=True
                )
                # tokenize targets
                tokenized_targets = self.tokenizer.batch_encode_plus(
                    [target],
                    max_length=self.max_len,
                    pad_to_max_length=True,
                    return_tensors='pt',
                    truncation=True
                )

                self.inputs.append(tokenized_inputs)
                self.targets.append(tokenized_targets)
