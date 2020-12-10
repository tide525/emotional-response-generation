import os
import sys
import textwrap
from tqdm.auto import tqdm

from transformers import T5ForConditionalGeneration, T5Tokenizer
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

from dataset import MultitaskDataset

tasks = sys.argv[1:]

model = T5ForConditionalGeneration.from_pretrained(
    os.path.join('model', ''.join(task[0] for task in tasks))
).cuda()
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# visualize few predictions on test dataset
dataset = MultitaskDataset(tasks, tokenizer, 'data', 'test', 256)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

it = iter(loader)
batch = next(it)

outs = model.generate(
    input_ids=batch['source_ids'].cuda(),
    attention_mask=batch['source_mask'].cuda(),
    max_length=256,
)

dec = [tokenizer.decode(ids) for ids in outs]

texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
targets = [
    tokenizer.decode(ids) for ids in batch['target_ids']
]
    
for i in range(32):
    lines = textwrap.wrap('Input: ' + texts[i])
    print('\n'.join(lines))

    lines = textwrap.wrap('Actual target: ' + targets[i])
    print('\n' + '\n'.join(lines))
    lines = textwrap.wrap('Predicted target: ' + dec[i])
    print('\n'.join(lines))

    print('=' * 70 + '\n')


def eval_classification(model, tokenizer, loader, labels):
    y_true = []
    y_pred = []

    for batch in tqdm(loader):
        outs = model.generate(
            input_ids=batch['source_ids'].cuda(),
            attention_mask=batch['source_mask'].cuda(),
            max_length=2
        )

        dec = [tokenizer.decode(ids) for ids in outs]
        target = [tokenizer.decode(ids) for ids in batch['target_ids']]

        y_pred.extend(dec)
        y_true.extend(target)

    print(metrics.classification_report(y_true, y_pred, labels=labels))


def eval_generation(model, tokenizer, loader):
    list_of_references = []
    hypotheses = []

    for batch in tqdm(loader):
        outs = model.generate(
            input_ids=batch['source_ids'].cuda(),
            attention_mask=batch['source_mask'].cuda(),
            max_length=256,
        )

        dec = [tokenizer.convert_ids_to_tokens(ids, True) for ids in outs]
        target = [
            tokenizer.convert_ids_to_tokens(ids, True)
            for ids in batch['target_ids']
        ]

        hypotheses.extend(dec)
        list_of_references.extend([[tokens] for tokens in target])

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
    dataset = MultitaskDataset([task], tokenizer, 'data', 'test', 256)
    loader = DataLoader(dataset, batch_size=32)

    if task == 'emotion':
        labels = ['joy', 'anger', 'sadness', 'disgust', 'fear', 'surprise']
        
        print('Emotion\n')
        eval_classification(model, tokenizer, loader, labels)

    elif task == 'response':
        print('Response\n')
        eval_generation(model, tokenizer, loader)

    elif task == 'sentiment':
        labels = ['positive', 'negative']

        print('Sentiment analysis\n')
        eval_classification(model, tokenizer, loader, labels)

    else:
        pass

    print()
