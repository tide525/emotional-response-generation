import argparse
import os
import shutil
import sys

import pytorch_lightning as pl

from dataset import MultitaskDataset
from model import MultitaskBartFinetuner, LoggingCallback, args_dict
from adv import AdversarialMultitaskBartFinetuner

parser = argparse.ArgumentParser()

for name, default in args_dict.items():
    parser.add_argument('--' + name, type=type(default), default=default)
parser.add_argument('--tasks', type=str, default='')
parser.add_argument('--task_dirs', type=str, default='')

parser.add_argument('--task_weights', type=str, default='')
parser.add_argument('--adversarial', action='store_true')

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
        task_dirs=args.task_dirs.split(','),
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        type_path=type_path,
        max_len=args.max_seq_length
    )


# initialize model
if args.adversarial:
    model = AdversarialMultitaskBartFinetuner(args, get_dataset)
else:
    model = MultitaskBartFinetuner(args, get_dataset)

# initialize trainer
trainer = pl.Trainer(**train_params)

# start fine-tuning
trainer.fit(model)

model.model.save_pretrained(args.output_dir)
