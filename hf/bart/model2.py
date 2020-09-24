import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_bart import shift_tokens_right

from multitask_bart import BartForMultitaskLearning
from model import label_smoothed_nll_loss, MultitaskBartFinetuner


class BartForMultitaskLearning2(BartForMultitaskLearning):
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

        x = outputs[0]  # last hidden state
        eos_mask = input_ids.eq(self.config.eos_token_id)
        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError(
                "All examples must have the same number of <eos> tokens."
            )

        sentence_representation = x[eos_mask, :].view(
            x.size(0),
            -1,
            x.size(-1)
        )[:, -1, :]
        logits = self.emotion_head(sentence_representation)

        lm_logits = F.linear(
            outputs[0],
            self.model.shared.weight,
            bias=self.final_logits_bias
        )

        # Prepend logits
        outputs = (logits,) + outputs[1:]  # Add hidden states and attention if they are here

        outputs = (lm_logits,) + outputs#[1:]  # Add cache, hidden states and attention if they are here

        if labels is not None:  # prepend loss to output,
            loss = F.cross_entropy(
                logits.view(-1, self.num_emotions),
                labels[0].view(-1)
            )
            outputs = (loss,) + outputs

            loss_fct = nn.CrossEntropyLoss()
            # TODO(SS): do we need to ignore pad tokens in labels?
            masked_lm_loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size),
                labels[1].view(-1)
            )
            outputs[0] = masked_lm_loss + outputs[0]
        
        return outputs


class MultitaskBartFinetuner2(MultitaskBartFinetuner):
    def __init__(self, hparams, get_dataset):
        super().__init__(hparams, get_dataset)
        self.model = BartForMultitaskLearning2.from_pretrained(
            hparams.model_name_or_path
        )

    """def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch["target_mask"]
        )
        loss = outputs[0]
        return loss
    """

    def _step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        target_ids = batch["target_ids"][1]

        decoder_input_ids = shift_tokens_right(target_ids, pad_token_id)

        labels = batch["target_ids"][0]

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        )
        
        lm_logits, logits = outputs[:2]  # Add hidden states and attention if they are here

        loss = F.cross_entropy(
            logits.view(-1, self.model.num_emotions),
            labels.view(-1)
        )
        outputs = (loss,) + outputs

        lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target_ids,
            self.hparams.label_smoothing,
            ignore_index=pad_token_id
        )
        outputs = (loss,) + outputs

        return outputs[1] + outputs[0]
