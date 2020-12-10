import glob
import os

from torch.utils.data import Dataset

path_dict = {
    'emotion': 'tec',
    'response': 'dd_dial_ne',
    'sentiment': 'sst2_qtr'
}

prefix_dict = {
    'emotion': 'recognize emotion: ',
    'response': 'generate response: ',
    'sentiment': 'analyze sentiment: '
}

emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
sentiments = ['positive', 'negative']


class MultitaskDataset(Dataset):
    def __init__(self, tasks, tokenizer, data_dir, type_path, max_len=512):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build(tasks, data_dir, type_path)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]['input_ids'].squeeze()
        target_ids = self.targets[index]['input_ids'].squeeze()

        src_mask = self.inputs[index]['attention_mask'].squeeze()  # might need to squeeze
        target_mask = self.targets[index]['attention_mask'].squeeze()  # might need to squeeze

        return {
            'source_ids': source_ids,
            'source_mask': src_mask,
            'target_ids': target_ids,
            'target_mask': target_mask
        }
    
    def _build(self, tasks, data_dir, type_path):
        for task in tasks:
            input_path = os.path.join(
                data_dir,
                path_dict[task],
                type_path + '.source'
            )
            with open(input_path) as f:
                for line in f:
                    input_ = prefix_dict[task] + line.strip() + ' </s>'
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
                path_dict[task],
                type_path + '.target'
            )
            with open(target_path) as f:
                for line in f:
                    if task == 'emotion':
                        emotion = line.strip()
                        if emotion == 'joy':
                            emotion = 'happiness'
                        target = emotion + ' </s>'
                    elif task == 'sentiment':
                        target = sentiments[int(line)] + ' </s>'
                    else:
                        target = line.strip() + ' </s>'
                    # tokenize targets
                    tokenized_targets = self.tokenizer.batch_encode_plus(
                        [target],
                        max_length=self.max_len,
                        pad_to_max_length=True,
                        return_tensors='pt',
                        truncation=True
                    )
                    self.targets.append(tokenized_targets)
