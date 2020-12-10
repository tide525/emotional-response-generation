import glob
import os
import sys

from torch import cuda
import pytorch_lightning as pl
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification
)

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


output_dir = sys.argv[1]
save_dir = sys.argv[2]

checkpoint_path = glob.glob(output_dir + '/checkpointepoch=*.ckpt')[0]
model = RobertaFinetuner.load_from_checkpoint(
    checkpoint_path,
    get_dataset=None
).model

model.save_pretrained(save_dir)
