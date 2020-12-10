import glob
import os
import sys
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import BartTokenizer

from multitask_bart import BartForMultitaskLearning
from dataset import TaskDataset


class BartFinetuner(pl.LightningModule):
    def __init__(self, hparams, get_dataset):
        super().__init__()
        self.hparams = hparams
        self.get_dataset = get_dataset

        self.model = BartForMultitaskLearning.from_pretrained(
            hparams['model_name_or_path'],
        )
        self.tokenizer = BartTokenizer.from_pretrained(
            hparams['tokenizer_name_or_path']
        )


output_dir = sys.argv[1]

checkpoint_path = glob.glob(output_dir + '/checkpointepoch=*.ckpt')[0]
model = BartFinetuner.load_from_checkpoint(
    checkpoint_path,
    get_dataset=None
).model.cuda()

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

dataset = TaskDataset('emotion', tokenizer, '../data/tec', 'test', 64)
loader = DataLoader(dataset, batch_size=32)

model.eval()
outputs = []

for batch in tqdm(loader):
    outs = model(
        input_ids=batch["source_ids"].cuda(),
        attention_mask=batch["source_mask"].cuda(),
        # lm_labels=batch["target_ids"].cuda(),
        task=batch["task"][0]
    )

    dec = outs[0].argmax(-1).tolist()
    outputs.extend(dec)

with open(os.path.join(output_dir, 'cls.txt'), 'w') as f:
    for output in outputs:
        f.write(str(output) + '\n')
