import math
import os
import sys
from tqdm.auto import tqdm

from torch import cuda
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification
)

in_path = os.path.join(sys.argv[1], '{0}', 'dialogues_{0}.txt')
out_path = os.path.join(sys.argv[2], '{}.{}')

device = 'cuda' if cuda.is_available() else 'cpu'


class RobertaDataset(Dataset):
    def __init__(self, tokenizer, inputs, max_len=512):
        self.inputs = []
        for input_ in inputs:
            # tokenize inputs
            tokenized_inputs = tokenizer.batch_encode_plus(
                [input_],
                max_length=max_len,
                pad_to_max_length=True,
                return_tensors='pt',
                truncation=True
            )
            self.inputs.append(tokenized_inputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]['input_ids'].squeeze()
        source_mask = self.inputs[index]['attention_mask'].squeeze()  # might need to squeeze

        return {'source_ids': source_ids, 'source_mask': source_mask}


tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
model = RobertaForSequenceClassification.from_pretrained('roberta_tec').to(device)

model.eval()

for split in tqdm(['train', 'validation', 'test']):
    inputs = []
    targets = []

    with open(in_path.format(split)) as f:
        for line in f:
            texts = line.split('__eou__')[:-1]
            for i in range(len(texts) - 1):
                inputs.append(texts[i].strip())
                targets.append(texts[i+1].strip())
    
    pairs = list(map(list, zip(inputs, targets)))

    dataset = RobertaDataset(tokenizer, inputs, 64)
    loader = DataLoader(dataset, batch_size=32)

    ents = []

    for batch in tqdm(loader):
        outs = model(
            input_ids=batch["source_ids"].to(device),
            attention_mask=batch["source_mask"].to(device)
        )
        dec = outs[0].softmax(-1)

        ent = (-dec * dec.log()).sum(-1)
        ents.extend(ent.tolist())

    pairs.sort(key=lambda pair: ents[inputs.index(pair[0])])

    if split == 'validation':
        split = 'val'

    for i, lang in enumerate(['source', 'target']):
        with open(out_path.format(split, lang), 'w') as f:
            for j in range(len(pairs)):
                f.write(pairs[j][i] + '\n')

    with open(out_path.format(split, 'in_ent'), 'w') as f:
        for ent in sorted(ents):
            f.write(str(ent) + '\n')



"""import glob
import os
import sys
from tqdm.auto import tqdm

from torch import cuda
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification
)

from dataset import EmotionDataset
from model import RobertaFinetuner


class RobertaFinetuner(pl.LightningModule):
    def __init__(self, hparams, get_dataset):
        super().__init__()
        self.hparams = hparams
        self.get_dataset = get_dataset

        config = RobertaConfig.from_pretrained(hparams['model_name_or_path'])
        config.num_labels = 6

        self.model = RobertaForSequenceClassification.from_pretrained(
            hparams['model_name_or_path'],
            config=config
        )
        self.tokenizer = RobertaTokenizer.from_pretrained(
            hparams['tokenizer_name_or_path']
        )


device = 'cuda' if cuda.is_available() else 'cpu'

output_dir = sys.argv[1]

checkpoint_path = glob.glob(output_dir + '/checkpointepoch=*.ckpt')[0]
model = RobertaFinetuner.load_from_checkpoint(
    checkpoint_path,
    get_dataset=None
).model.to(device)

# model = RobertaForSequenceClassification.from_pretrained(output_dir).to(device)
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

dataset = EmotionDataset(tokenizer, '../data', 'test', 64)
loader = DataLoader(dataset, batch_size=32)

model.eval()
outputs = []

for batch in tqdm(loader):
    outs = model(
        input_ids=batch["source_ids"].to(device),
        attention_mask=batch["source_mask"].to(device),
        # labels=batch["target_label"].to(device)
    )

    dec = outs[0].argmax(-1).tolist()
    outputs.extend(dec)

with open(os.path.join(output_dir, 'pred.txt'), 'w') as f:
    for output in outputs:
        f.write(str(output) + '\n')
"""
