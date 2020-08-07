import os
import sys

from transformers import BartTokenizer
from torch.utils.data import Dataset, DataLoader
from nltk.translate import bleu_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn import metrics

from multitask_bart import BartForMultitaskLearning
from dataset import MultitaskDataset
from sampler import MultitaskSampler


def eval_classification(model, tokenizer, loader, labels):
    y_true = []
    y_pred = []

    model.eval()
    for batch in loader:
        lm_labels = batch['target_ids'].cuda()
        lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100

        outs = model(
            input_ids=batch['source_ids'].cuda(),
            attention_mask=batch['source_mask'].cuda(),
            decoder_attention_mask=batch['target_mask'].cuda(),
            lm_labels=lm_labels,
            task=batch['task'][0],
        )
        outs = outs[1].argmax(1)

        dec = [label.item() for label in outs]
        target = [label.item() for label in batch['target_ids']]

        y_pred.extend(dec)
        y_true.extend(target)

    print(metrics.classification_report(y_true, y_pred, labels=labels))


def eval_generation(model, tokenizer, loader):
    list_of_references = []
    hypotheses = []

    for batch in loader:
        outs = model.generate(
            input_ids=batch['source_ids'].cuda(),
            attention_mask=batch['source_mask'].cuda(),
            max_length=256,
            task=batch['task'][0],
        )

        dec = [tokenizer.convert_ids_to_tokens(ids) for ids in outs]
        target = [
            tokenizer.convert_ids_to_tokens(ids)
            for ids in batch['target_ids']
        ]

        hypotheses.extend(dec)
        list_of_references.extend(list(map(lambda tokens: [tokens], target)))

    print(
        'BLEU score:',
        bleu_score.corpus_bleu(list_of_references, hypotheses)
    )

    print()
    for i in range(4):
        weights = tuple(float(j == i) for j in range(4))
        print(
            str(i + 1) + '-gram:',
            bleu_score.corpus_bleu(
                list_of_references, hypotheses, weights=weights
            )
        )


tasks = sys.argv[1:len(sys.argv)]

model = BartForMultitaskLearning.from_pretrained(
    os.path.join('model', ''.join(task[0] for task in tasks))
).cuda()
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

for task in tasks:
    dataset = MultitaskDataset([task], tokenizer, '../data', 'test', 512)
    loader = DataLoader(dataset, batch_size=8)

    if task == 'emotion':
        labels = ['joy', 'anger', 'sadness', 'disgust', 'fear', 'surprise']
        labels = list(range(6))
        print('Emotion\n')
        eval_classification(model, tokenizer, loader, labels)

    elif task == 'response':
        print('Response\n')
        eval_generation(model, tokenizer, loader)

    elif task == 'sentiment':
        labels = ['positive', 'negative']

        print('Sentiment analysis\n')
        eval_classification(model, tokenizer, loader, labels)

    print()
