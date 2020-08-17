import argparse
import os
import shutil

import pytorch_lightning as pl
from transformers import BartTokenizer

from model import BartFineTuner, LoggingCallback, args_dict

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

output_dir = 'bart_response'
if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)

args_dict.update(dict(
    data_dir='response_data',
    output_dir=output_dir,
    max_seq_length=256,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_steps=500,
    num_train_epochs=4,
    max_grad_norm=0.1,
))
args = argparse.Namespace(**args_dict)
# print(args_dict)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filepath=args.output_dir,
    prefix='checkpoint',
    monitor='val_loss',
    mode='min',
    save_top_k=5
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

model = BartFineTuner(args)
trainer = pl.Trainer(**train_params)
trainer.fit(model)

model_dir = 'bart_large_response'
if os.path.isdir(model_dir):
    shutil.rmtree(model_dir)
os.mkdir(model_dir)

model.model.save_pretrained(model_dir)
