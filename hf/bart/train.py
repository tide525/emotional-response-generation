import argparse
import os
import shutil
import sys

import pytorch_lightning as pl

from dataset import MultitaskDataset
from model import MultitaskBartFinetuner, LoggingCallback, args_dict

parser = argparse.ArgumentParser()

for name, default in args_dict.items():
    if name in ['data_dir', 'output_dir']:
        parser.add_argument(name, type=str)
    else:
        parser.add_argument('--' + name, type=type(default), default=default)

parser.add_argument('--num_train_steps', type=int, default=None)
parser.add_argument('--tasks', type=str, default='')
parser.add_argument('--weights', type=str, default='')

parser.add_argument('--val_scoring', action='store_true')

args = parser.parse_args()

if os.path.isdir(args.output_dir):
    shutil.rmtree(args.output_dir)
os.mkdir(args.output_dir)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filepath=args.output_dir,
    prefix='checkpoint',
    monitor='val_loss',
    mode='min',
    save_last=True,
    save_top_k=1
)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    max_steps=args.num_train_steps,
    early_stop_callback=False,
    precision=16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    callbacks=[LoggingCallback()]
)


def get_dataset(tokenizer, type_path, args):
    return MultitaskDataset(
        tasks=args.tasks.split(','),
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        type_path=type_path,
        max_len=args.max_seq_length
    )


# initialize model
model = MultitaskBartFinetuner(args, get_dataset)

# initialize trainer
trainer = pl.Trainer(**train_params)

# start fine-tuning
trainer.fit(model)

model.model.save_pretrained(args.output_dir)
