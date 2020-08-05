import glob
import os

from torch.utils.data import Dataset

path_dict = dict(
    ec='ued',
    rg='cmdc',
    sa='sst2',
    
    revec='ued',
    rg2='crawled',
)

prefix_dict = dict(
    ec='classify emotion: ',
    rg='generate response: ',
    sa='analyze sentiment: ',

    revec='generate emotion: ',
    rg2='generate response: ',
)


class MultitaskDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, task_names, max_len=512):
        self.file_paths = [os.path.join(data_dir, path_dict[task_name], type_path + '.tsv') for task_name in task_names]
        self.prefixes = [prefix_dict[task_name] for task_name in task_names]

        self.reverses = [task_name == 'rec' for task_name in task_names]

        self.task_num = len(task_names)

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

        src_mask = self.inputs[index]['attention_mask'].squeeze()  # might need to squeeze
        target_mask = self.targets[index]['attention_mask'].squeeze()  # might need to squeeze

        return {'source_ids': source_ids, 'source_mask': src_mask, 'target_ids': target_ids, 'target_mask': target_mask}
    
    def _build(self):
        for i in range(self.task_num):
            self._build_examples_from_file(self.file_paths[i], self.prefixes[i], self.reverses[i])

    def _build_examples_from_file(self, file_path, prefix, reverse=False):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                input_, target = line.strip().split('\t')
                if reverse:
                    input_, target = target, input_

                # tokenize inputs
                tokenized_inputs = self.tokenizer.batch_encode_plus([prefix + input_ + ' </s>'], max_length=self.max_len, pad_to_max_length=True, return_tensors='pt', truncation=True)
                # tokenize targets
                tokenized_targets = self.tokenizer.batch_encode_plus([target + ' </s>'], max_length=self.max_len, pad_to_max_length=True, return_tensors='pt', truncation=True)

                self.inputs.append(tokenized_inputs)
                self.targets.append(tokenized_targets)
