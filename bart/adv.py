import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AdamW, BartConfig
from transformers.modeling_bart import (
    shift_tokens_right,
    BartClassificationHead
)

from multitask_bart import BartForMultitaskLearning
from model import label_smoothed_nll_loss, MultitaskBartFinetuner


class BartForAdversarialMultitaskLearning(BartForMultitaskLearning):
    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        task=None,
        **unused
    ):
        if "lm_labels" in unused:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed "
                "in a future version, use `labels` instead.",
                DeprecationWarning
            )
            labels = unused.pop("lm_labels")

        if labels is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        if task == "response":
            lm_logits = F.linear(
                outputs[0],
                self.model.shared.weight,
                bias=self.final_logits_bias
            )
            outputs = (lm_logits,) + outputs#[1:]  # Add cache, hidden states and attention if they are here

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                # TODO(SS): do we need to ignore pad tokens in labels?
                masked_lm_loss = loss_fct(
                    lm_logits.view(-1, self.config.vocab_size),
                    labels.view(-1)
                )
                outputs = (masked_lm_loss,) + outputs

        elif task in ["cfemotion", "emotion", "sentiment"]:
            x = outputs[0]  # last hidden state

            eos_mask = input_ids.eq(self.config.eos_token_id)
            if len(torch.unique(eos_mask.sum(1))) > 1:
                raise ValueError(
                   "All examples must have the same number of <eos> tokens."
                )

            if task == "cfemotion":
                classification_head = self.cfemotion_head
                num_labels = self.num_cfemotions
            elif task == "emotion":
                classification_head = self.emotion_head
                num_labels = self.num_emotions
            else:
                classification_head = self.sentiment_head
                num_labels = self.num_sentiments

            sentence_representation = x[eos_mask, :].view(
                x.size(0),
                -1,
                x.size(-1)
            )[:, -1, :]
            logits = classification_head(sentence_representation)

            # Prepend logits
            outputs = (logits,) + outputs#[1:]  # Add hidden states and attention if they are here
            if labels is not None:  # prepend loss to output,
                loss = F.cross_entropy(
                    logits.view(-1, num_labels),
                    labels.view(-1)
                )
                outputs = (loss,) + outputs
        
        else:
            raise ValueError("The dataset contains an invalid task.")

        return outputs


class AdversarialMultitaskBartFinetuner(MultitaskBartFinetuner):
    def __init__(self, hparams, get_dataset):
        super().__init__(hparams, get_dataset)

        self.model = BartForAdversarialMultitaskLearning.from_pretrained(
            hparams.model_name_or_path,
            config=BartConfig()
        )

        self.discriminator = BartClassificationHead(
            self.model.config.d_model,
            self.model.config.d_model,
            2,  # gen and cls
            self.model.config.classif_dropout
        )

    def _step(self, batch):
        labels = torch.full(
            (len(batch['task']), 1),
            int(batch['task'][0] != 'response'),  # fake
            dtype=torch.long
        ).cuda()

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

            x = outputs[1]  # last hidden state

        elif batch['task'][0] in ['cfemotion', 'emotion', 'sentiment']:
            outputs = self(
                input_ids=batch['source_ids'],
                attention_mask=batch['source_mask'],
                lm_labels=batch['label'],
                task=batch['task'][0]
            )
            loss = outputs[0]

            x = outputs[2]  # last hidden state

        else:
            raise ValueError('The dataset contains an invalid task.')

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
        adv_loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1))

        return loss + adv_loss

    def _step_D(self, batch):
        labels = torch.full(
            (len(batch['task']), 1),
            int(batch['task'][0] == 'response'),  # real
            dtype=torch.long
        ).cuda()

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

        elif batch['task'][0] in ['cfemotion', 'emotion', 'sentiment']:
            outputs = self(
                input_ids=batch['source_ids'],
                attention_mask=batch['source_mask'],
                # lm_labels=batch['label'],
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
            loss = self._step_D(batch)
            tensorboard_logs = {'train_loss_D': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x['loss'] for x in outputs[0]]).mean()
        tensorboard_logs = {'avg_train_loss': avg_train_loss}
        return {
            'avg_train_loss': avg_train_loss,
            'log': tensorboard_logs,
            'progress_bar': tensorboard_logs
        }

    def configure_optimizers(self):
        # model
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

        # discriminator
        optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=0.0002,
            betas=(0.5, 0.999)
        )
        return [optimizer, optimizer_D]

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
