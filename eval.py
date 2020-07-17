import sys

from nltk.tokenize import word_tokenize
from nltk.translate import bleu_score
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer

from dataset import MultitaskDataset


def eval_classification(model, tokenizer, loader, labels):
    y_true = []
    y_pred = []

    for batch in loader:
        y_true += [labels.index(tokenizer.decode(ids)) for ids in batch['target_ids']]

        outs = model.generate(input_ids=batch['source_ids'].cuda(), attention_mask=batch['source_mask'].cuda(), max_length=2)
        y_pred += [labels.index(tokenizer.decode(ids)) for ids in outs]

    print('Accuracy: {}\n'.format(accuracy_score(y_true, y_pred)))

    print('Confusion matrix:\n{}\n'.format(confusion_matrix(y_true, y_pred), labels=labels))

    if len(labels) == 2:
        print('Precision: {}\nRecall: {}\n'.format(precision_score(y_true, y_pred), recall_score(y_true, y_pred)))
        print('F1 score: {}\n'.format(f1_score(y_true, y_pred)))

    else:
        for average in ['macro', 'micro']:
            print('# {}\n'.format(average))

            print('Precision: {}\nRecall: {}\n'.format(precision_score(y_true, y_pred, average=average), recall_score(y_true, y_pred, average=average)))
            print('F1 score: {}\n'.format(f1_score(y_true, y_pred, average=average)))


def eval_generation(model, tokenizer, loader):
    list_of_references = []
    hypotheses = []

    for batch in loader:
        targets = [tokenizer.decode(ids) for ids in batch['target_ids']]
        list_of_references += [[word_tokenize(tokenizer.decode(ids))] for ids in batch['target_ids']]

        outs = model.generate(input_ids=batch['source_ids'].cuda(), attention_mask=batch['source_mask'].cuda())
        hypotheses += [word_tokenize(tokenizer.decode(ids)) for ids in outs]

    print('BLEU score: {}\n'.format(bleu_score.corpus_bleu(list_of_references, hypotheses)))


task_names = sys.argv[1].split(',')

model = T5ForConditionalGeneration.from_pretrained(''.join(task_names)).cuda()
tokenizer = T5Tokenizer.from_pretrained('t5-base')

for task_name in task_names:
    dataset = MultitaskDataset(tokenizer, 'data', 'test', [task_name], 512)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    if task_name == 'ec':
        labels = ['joy', 'anger', 'sadness', 'disgust', 'fear', 'surprise']

        print('# Emotion classification\n')
        eval_classification(model, tokenizer, loader, labels)

    elif task_name == 'rg':
        print('# Response generation\n')
        eval_generation(model, tokenizer, loader)
    
    elif task_name == 'sa':
        labels = ['positive', 'negative']

        print('# Sentiment analysis\n')
        eval_classification(model, tokenizer, loader, labels)

    else:
        pass
