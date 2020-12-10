import logging
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    BartConfig,
    BartTokenizer,
    get_linear_schedule_with_warmup
)
from transformers.modeling_bart import (
    shift_tokens_right,
    BartClassificationHead
)
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu

from dataset import TaskDataset
from multitask_bart import BartForMultitaskLearning
from sampler import (
    MultitaskBatchSampler,
    CompetenceBatchSampler,
    TaskCompetenceBatchSampler
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    '''From fairseq'''
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


class MultitaskBartFinetuner(pl.LightningModule):
    def __init__(self, hparams, get_dataset):
        super().__init__()
        self.hparams = hparams
        self.get_dataset = get_dataset

        self.model = BartForMultitaskLearning.from_pretrained(
            hparams.model_name_or_path
        )
        self.tokenizer = BartTokenizer.from_pretrained(
            hparams.tokenizer_name_or_path
        )

        self.tasks = self.hparams.tasks.split(',')

        # loss weighting
        self.loss_weights = (
            [float(weight) for weight in hparams.loss_weights.split(',')]
            if hparams.loss_weights
            else [1.0 for task in self.tasks]
        )
        assert len(self.tasks) == len(loss_weights)

        # competence-based curriculum learning
        if self.hparams.competence:
            self.difficulties = []
            task_dir = self.hparams.task_dirs.split(',')[
                self.hparams.tasks.split(',').index('response')
            ]
            diff_path = os.path.join(
                hparams.data_dir,
                os.path.join(self.hparams.data_dir, task_dir),
                'train.in_ent'
            )
            with open(diff_path) as f:
                for line in f:
                    self.difficulties.append(float(line))

        # adversarial learning
        if hparams.adversarial:
            self.discriminator = BartClassificationHead(
                self.model.config.d_model,
                self.model.config.d_model,
                2,  # generation and classification
                self.model.config.classif_dropout
            )

        # BLEU
        if hparams.val_bleu:
            tasks = hparams.tasks.split(',')
            task_dirs = hparams.task_dirs.split(',')

            assert 'response' in tasks
            task_dir = task_dirs[tasks.index('response')]

            bleu_dataset = TaskDataset(
                task='response',
                tokenizer = self.tokenizer,
                data_dir=os.path.join(hparams.data_dir, task_dir),
                type_path='val',
                max_len=hparams.max_seq_length
            )
            bleu_loader = DataLoader(
                bleu_dataset,
                batch_size=self.hparams.train_batch_size
            )
            self.bleu_loader = bleu_loader

            list_of_references = []
            for batch in bleu_loader:
                decs = [
                    self.tokenizer.decode(ids, skip_special_tokens=True)
                    for ids in batch['target_ids']
                ]
                list_of_references.extend([[word_tokenize(dec)] for dec in decs])
            self.list_of_references = list_of_references

    def is_logger(self):
        return self.trainer.global_rank <= 0

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        lm_labels=None,
        use_cache=None,
        task=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            lm_labels=lm_labels,
            use_cache=use_cache,
            task=task
        )

    # def _step(self, batch):
    #     lm_labels = batch['target_ids']
    #     lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
    #
    #     outputs = self(
    #         input_ids=batch['source_ids'],
    #         attention_mask=batch['source_mask'],
    #         lm_labels=lm_labels,
    #         decoder_attention_mask=batch['target_mask']
    #     )
    #     loss = outputs[0]
    #     return loss

    def _step(self, batch):
        if self.hparams.adversarial:
            label = int(batch['task'][0] == 'response')
            labels = torch.full((len(batch['task']), 1), label).cuda()

        if batch['task'][0] == 'response':
            pad_token_id = self.tokenizer.pad_token_id
            target_ids = batch['target_ids']

            decoder_input_ids = shift_tokens_right(target_ids, pad_token_id)

            outputs = self(
                input_ids=batch['source_ids'],
                attention_mask=batch['source_mask'],
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
                task=batch['task'][0]
            )

            lprobs = torch.nn.functional.log_softmax(outputs[0], dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs,
                target_ids,
                self.hparams.label_smoothing,
                ignore_index=pad_token_id
            )

            loss = self.loss_weights[self.tasks.index('response')] * loss

            if self.hparams.adversarial:
                x = outputs[1]  # last hidden state
                eos_mask = batch['source_ids'].eq(self.model.config.eos_token_id)
                if len(torch.unique(eos_mask.sum(1))) > 1:
                    raise ValueError('All examples must have the same number of <eos> tokens.')
                sentence_representation = x[eos_mask, :].view(x.size(0), -1, x.size(-1))[:, -1, :]
                logits = self.discriminator(sentence_representation)
                loss = loss + F.cross_entropy(logits.view(-1, 2), labels.view(-1))

        elif batch['task'][0] in ['emotion', 'sentiment', 'cfemotion']:
            outputs = self(
                input_ids=batch['source_ids'],
                attention_mask=batch['source_mask'],
                lm_labels=batch['target_ids'],
                task=batch['task'][0]
            )
            loss = outputs[0]

            loss = self.loss_weights[self.tasks.index(batch['task'][0])] * loss

            if self.hparams.adversarial:
                x = outputs[2]  # last hidden state
                eos_mask = batch['source_ids'].eq(self.model.config.eos_token_id)
                if len(torch.unique(eos_mask.sum(1))) > 1:
                    raise ValueError('All examples must have the same number of <eos> tokens.')
                sentence_representation = x[eos_mask, :].view(x.size(0), -1, x.size(-1))[:, -1, :]
                logits = self.discriminator(sentence_representation)
                loss = loss + F.cross_entropy(logits.view(-1, 2), labels.view(-1))

        else:
            raise ValueError('The dataset contains an invalid task.')

        return loss

    def _stepD(self, batch):
        label = int(batch['task'][0] == 'response')
        labels = torch.full((len(batch['task']), 1), label).cuda()

        if batch['task'][0] == 'response':
            pad_token_id = self.tokenizer.pad_token_id
            target_ids = batch['target_ids']
            decoder_input_ids = shift_tokens_right(target_ids, pad_token_id)
            outputs = self(
                input_ids=batch['source_ids'],
                attention_mask=batch['source_mask'],
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
                task=batch['task'][0]
            )
        elif batch['task'][0] in ['emotion', 'sentiment']:
            outputs = self(
                input_ids=batch['source_ids'],
                attention_mask=batch['source_mask'],
                task=batch['task'][0]
            )
        else:
            raise ValueError('The dataset contains an invalid task.')

        x = outputs[1]  # last hidden state
        eos_mask = batch['source_ids'].eq(self.model.config.eos_token_id)
        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError(
                'All examples must have the same number of <eos> tokens.'
            )
        sentence_representation = x[eos_mask, :].view(
            x.size(0),
            -1,
            x.size(-1)
        )[:, -1, :]
        logits = self.discriminator(sentence_representation)
        loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1))

        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:  # model
            loss = self._step(batch)
            tensorboard_logs = {'train_loss': loss}
        if optimizer_idx == 1:  # discriminator
            lossD = self._stepD(batch)
            tensorboard_logs = {'train_lossD': lossD}
        return {'loss': loss, 'log': tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x['loss'] for x in outputs[0]]).mean()
        tensorboard_logs = {'avg_train_loss': avg_train_loss}
        return {
            'avg_train_loss': avg_train_loss,
            'log': tensorboard_logs,
            'progress_bar': tensorboard_logs
        }

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}

        if self.hparams.val_bleu:
            hypotheses = []

            # generate
            self.model.eval()
            for batch in self.bleu_loader:
                outs = self.model.generate(
                    input_ids=batch['source_ids'].cuda(),
                    attention_mask=batch['source_mask'].cuda(), 
                    max_length=self.hparams.max_seq_length,
                    num_beams=5,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    task=batch['task'][0]
                )
                decs = [
                    self.tokenizer.decode(ids, skip_special_tokens=True)
                    for ids in outs
                ]
                hypotheses.extend([word_tokenize(dec) for dec in decs])
            self.model.train()

            bleu = corpus_bleu(self.list_of_references, hypotheses)
            tensorboard_logs['val_bleu'] = bleu
            
        return {
            'avg_val_loss': avg_loss,
            'log': tensorboard_logs,
            'progress_bar': tensorboard_logs
        }

    def configure_optimizers(self):
        model = self.model
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay': self.hparams.weight_decay
            },
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.0
            }
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon
        )
        self.opt = optimizer
        optimizers = [optimizer]

        if self.hparams.adversarial:
            # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
            optimizerD = optim.Adam(
                self.discriminator.parameters(),
                lr=0.0002,
                betas=(0.5, 0.999)
            )
            optimizers.append(optimizerD)

        return optimizers

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        second_order_closure=None,
        using_native_amp=None
    ):
        if optimizer_idx == 0:  # model
            if self.trainer.use_tpu:
                xm.optimizer_step(optimizer)
            else:
                optimizer.step()
            optimizer.zero_grad()
            self.lr_scheduler.step()

        if optimizer_idx == 1:  # discriminator
            optimizer.step()
            optimizer.zero_grad()

    def get_tqdm_dict(self):
        tqdm_dict = {
            'loss': '{:.3f}'.format(self.trainer.avg_loss),
            'lr': self.lr_scheduler.get_last_lr()[-1]
        }
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = self.get_dataset(
            tokenizer=self.tokenizer,
            type_path='train',
            args=self.hparams
        )

        # competence
        if self.hparams.competence:
            assert len(self.tasks) == 1
            sampler = CompetenceBatchSampler(
                data_source=train_dataset,
                batch_size=self.hparams.train_batch_size,
                drop_last=False,
                epoch=self.current_epoch,
                num_epochs=self.hparams.num_train_epochs,
                difficulties=self.difficulties,
                init_competence=0.1
            )

        # task competence
        elif self.hparams.task_competence:
            assert len(self.tasks) == 3
            sorted_tasks = ['response', 'sentiment', 'emotion']
            difficulties = [0, 2, 6]  # number of labels
            sampler = TaskCompetenceBatchSampler(
                data_source=train_dataset,
                batch_size=self.hparams.train_batch_size,
                drop_last=False,
                epoch=self.current_epoch,
                num_epochs=self.hparams.num_train_epochs,
                tasks=sorted_tasks,
                difficulties=difficulties,
                init_competence=0.5
            )

        # default
        else:
            sampler = MultitaskBatchSampler(
                data_source=train_dataset,
                batch_size=self.hparams.train_batch_size,
                drop_last=False
            )

        dataloader = DataLoader(
            dataset=train_dataset,
            batch_sampler=sampler,
            num_workers=4
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
        val_dataset = self.get_dataset(
            tokenizer=self.tokenizer,
            type_path='val',
            args=self.hparams
        )
        sampler = MultitaskBatchSampler(
            data_source=val_dataset,
            batch_size=self.hparams.train_batch_size,
            drop_last=False
        )
        return DataLoader(
            dataset=val_dataset,
            batch_sampler=sampler,
            num_workers=4
        )


logger = logging.getLogger(__name__)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(
                pl_module.hparams.output_dir,
                "test_results.txt"
            )
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info(
                            "{} = {}\n".format(key, str(metrics[key]))
                        )
                        writer.write(
                            "{} = {}\n".format(key, str(metrics[key]))
                        )


args_dict = dict(
    data_dir='',  # path for data files
    output_dir='',  # path to save the checkpoints
    model_name_or_path='facebook/bart-large',
    tokenizer_name_or_path='facebook/bart-large',
    max_seq_length=512,
    learning_rate=3e-5,
    weight_decay=0.01,
    adam_epsilon=1e-8,
    warmup_steps=500,
    train_batch_size=4,
    eval_batch_size=4,
    num_train_epochs=2,
    gradient_accumulation_steps=4,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False,  # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1',  # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=0.1,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
    label_smoothing=0.1
)
