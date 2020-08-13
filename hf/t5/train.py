import argparse
import os
import sys

import pytorch_lightning as pl

from model import T5FineTuner, LoggingCallback, args_dict
from dataset import MultitaskDataset

task_names = sorted(sys.argv[1].split(','))

args_dict.update(dict(
    data_dir='data',
    output_dir=os.path.join('output', ''.join(task_names)),
    max_seq_length=256,
    num_train_epochs=2,
))
args = argparse.Namespace(**args_dict)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filepath=args.output_dir,
    prefix='checkpoint',
    monitor='val_loss',
    mode='min',
    save_top_k=5,
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
    callbacks=[LoggingCallback()],
)


def get_dataset(tokenizer, type_path, args):
    return MultitaskDataset(
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        type_path=type_path,
        task_names=task_names,
        max_len=args.max_seq_length,
    )


model = T5FineTuner(args, get_dataset)

trainer = pl.Trainer(**train_params)
trainer.fit(model)

model.model.save_pretrained(os.path.join('model', ''.join(task_names)))
