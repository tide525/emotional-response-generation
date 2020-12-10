import glob
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
