import os
import logging
import random

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import (
    AdamW,
    BartForConditionalGeneration,
    BartTokenizer,
    get_linear_schedule_with_warmup,
)

from dataset import ResponseDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# https://github.com/huggingface/transformers/blob/24107c2c83e79d195826f18f66892feab6b000e9/src/transformers/optimization.py#L168
def get_polynomial_decay_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    lr_end=1e-7,
    power=2.0,
    last_epoch=-1
):
    lr_init = optimizer.defaults['lr']
    assert (
        lr_init > lr_end,
        f'lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})'
    )

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = (
                1 - (current_step - num_warmup_steps) / decay_steps
            )
            decay = lr_range * pct_remaining ** power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# https://github.com/huggingface/transformers/blob/e92efcf7286c955e6901f894be39cf6154af48b7/examples/seq2seq/utils.py#L22
def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)

    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss
    

class BartFineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.tokenizer = BartTokenizer.from_pretrained(
            hparams.tokenizer_name_or_path,
        )
        self.model = BartForConditionalGeneration.from_pretrained(
            hparams.model_name_or_path,
        )

    def is_logger(self):
        return self.trainer.global_rank <= 0

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def _step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        target_ids = batch['target_ids']

        decoder_input_ids = target_ids[:, :-1].contiguous()  # Why this line?
        lm_labels = target_ids[:, 1:].clone()  # why clone?

        outputs = self(
            batch['source_ids'],
            attention_mask=batch['source_mask'],
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        )

        lprobs = torch.nn.functional.log_softmax(outputs[0], dim=-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            lm_labels,
            0.1,
            ignore_index=pad_token_id,
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        return {'train_loss': train_loss_mean}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': val_loss_mean}

    def get_lr_scheduler(self):
        train_batch_size = self.hparams.train_batch_size
        dataloader = self.get_dataloader('train', train_batch_size)
        self.total_steps = (
            (
                len(dataloader.dataset)
                // (
                    train_batch_size * max(1, self.hparams.n_gpu)
                )
            )
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )

        scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
        }
        return scheduler

    def configure_optimizers(self):
        model = self.model
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay': self.hparams.weight_decay,
            },
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon
        )
        self.opt = optimizer

        # scheduler = self.get_lr_scheduler()

        return [optimizer]  #, [scheduler]

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        second_order_closure=None
    ):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
               
        optimizer.zero_grad()
        self.lr_scheduler.step()

    # def get_progress_bar_dict(self):
    def get_tqdm_dict(self):
        progress_bar_dict = {
            'loss': '{:.3f}'.format(self.trainer.avg_loss),
            'lr': self.lr_scheduler.get_last_lr()[-1],
        }
        return progress_bar_dict

    def get_dataset(self, type_path):
        return ResponseDataset(
            tokenizer=self.tokenizer,
            data_dir=self.hparams.data_dir,
            type_path=type_path,
            max_len=self.hparams.max_seq_length,
        )

    def get_dataloader(self, type_path, batch_size, shuffle=False):
        dataset = self.get_dataset(type_path)
        sampler = None

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            sampler=sampler,
        )
        return dataloader

    def train_dataloader(self):
        dataloader = self.get_dataloader(
            'train',
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
        )

        t_total = (
            (
                len(dataloader.dataset)
                // (
                    self.hparams.train_batch_size * max(1, self.hparams.n_gpu)
                )
            )
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )

        scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=t_total
        )
        self.lr_scheduler = scheduler

        return dataloader

    def val_dataloader(self):
        return self.get_dataloader(
            'val',
            batch_size=self.hparams.eval_batch_size,
        )


logger = logging.getLogger(__name__)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info('***** Validation results *****')
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ['log', 'progress_bar']:
                logger.info('{} = {}\n'.format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info('***** Test results *****')

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

        # Log and save results to file
        output_test_results_file = os.path.join(
            pl_module.hparams.output_dir, 'test_results.txt')
        with open(output_test_results_file, 'w') as writer:
            for key in sorted(metrics):
                if key not in ['log', 'progress_bar']:
                    logger.info('{} = {}\n'.format(key, str(metrics[key])))
                    writer.write('{} = {}\n'.format(key, str(metrics[key])))


args_dict = dict(
    data_dir='',  # path for data files
    output_dir='',  # path to save the checkpoints
    model_name_or_path='facebook/bart-large',
    tokenizer_name_or_path='facebook/bart-large',
    max_seq_length=512,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=8,
    eval_batch_size=8,
    num_train_epochs=2,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False,  # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1',  # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    max_grad_norm=1.0,
    seed=42,
)
