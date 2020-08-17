import os
import sys
import textwrap
from tqdm.auto import tqdm

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

tasks = sys.argv[1:]

model = BartForMultitaskLearning.from_pretrained(
    os.path.join('model', ''.join(task[0] for task in tasks))
).cuda()
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

# visualize few predictions on test dataset
dataset = MultitaskDataset(tasks, tokenizer, '../data', 'test', 256)
sampler = MultitaskSampler(dataset, 32, False)

loader = DataLoader(dataset, batch_sampler=sampler)

it = iter(loader)

labels_dict = {
    'emotion': ['joy', 'anger', 'sadness', 'disgust', 'fear', 'surprise'],
    'sentiment': ['positive', 'negative']
}

batch = next(it)

for i in range(8):
    texts = [tokenizer.decode(ids, True) for ids in batch['source_ids']]

    if batch['task'][0] == 'response':
        outs = model.generate(
            input_ids=batch['source_ids'].cuda(),
            attention_mask=batch['source_mask'].cuda(),
            max_length=256,
            min_length=16,
            num_beams=5,
            task=batch['task'][0]
        )

        dec = [tokenizer.decode(ids, True) for ids in outs]
        targets = [
            tokenizer.decode(ids, True) for ids in batch['target_ids']
        ]
    
    elif batch['task'][0] in ['emotion', 'sentiment']:
        lm_labels = batch['target_ids'].cuda()
        lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100

        outs = model(
            input_ids=batch['source_ids'].cuda(),
            attention_mask=batch['source_mask'].cuda(),
            decoder_attention_mask=batch['target_mask'].cuda(),
            lm_labels=lm_labels,
            task=batch['task'][0]
        )[1].argmax(1)

        labels = labels_dict[batch['task'][0]]

        dec = [labels[label.item()] for label in outs]
        targets = [labels[label.item()] for label in batch['target_ids']]

    else:
        raise ValueError("A task must be emotion, response or sentiment.")
    
    for i in range(4):
        lines = textwrap.wrap('Input: {}'.format(texts[i]))
        print('\n'.join(lines))

        lines = textwrap.wrap('Actual target: {}'.format(targets[i]))
        print('\n' + '\n'.join(lines))
        lines = textwrap.wrap('Predicted target: {}'.format(dec[i]))
        print('\n'.join(lines))

        print('=' * 70 + '\n')
    
    batch = next(it)


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
            task=batch['task'][0]
        )[1].argmax(1)

        dec = [label.item() for label in outs]
        targets = [label.item() for label in batch['target_ids']]

        y_pred.extend(dec)
        y_true.extend(targets)

    print(metrics.classification_report(y_true, y_pred, labels=labels))


def eval_generation(model, tokenizer, loader):
    list_of_references = []
    hypotheses = []

    for batch in loader:
        outs = model.generate(
            input_ids=batch['source_ids'].cuda(),
            attention_mask=batch['source_mask'].cuda(),
            max_length=256,
            min_length=16,
            num_beams=5,
            task=batch['task'][0]
        )

        dec = [tokenizer.convert_ids_to_tokens(ids, True) for ids in outs]
        targets = [
            tokenizer.convert_ids_to_tokens(ids, True)
            for ids in batch['target_ids']
        ]

        hypotheses.extend(dec)
        list_of_references.extend([[tokens] for tokens in targets])

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


# predict on all the test dataset
for task in tasks:
    dataset = MultitaskDataset([task], tokenizer, '../data', 'test', 256)
    loader = DataLoader(dataset, batch_size=32)

    if task == 'emotion':
        # labels = ['joy', 'anger', 'sadness', 'disgust', 'fear', 'surprise']
        labels = list(range(6))
        
        print('Emotion\n')
        eval_classification(model, tokenizer, loader, labels)

    elif task == 'response':
        print('Response\n')
        eval_generation(model, tokenizer, loader)

    elif task == 'sentiment':
        # labels = ['positive', 'negative']
        labels = list(range(2))

        print('Sentiment analysis\n')
        eval_classification(model, tokenizer, loader, labels)

    else:
        raise ValueError("A task must be emotion, response or sentiment.")

    print()
