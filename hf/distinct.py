# https://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.word_tokenize
# https://www.nltk.org/api/nltk.html#nltk.util.ngrams

# https://arxiv.org/abs/1510.03055

import sys

from nltk.tokenize import word_tokenize
from nltk.util import ngrams


def corpus_distinct(corpus, n):
    num_tokens = 0
    num_distinct_ngrams = 0

    for sentence in corpus:
        tokens = word_tokenize(sentence)
        num_tokens += len(tokens)

        distinct_ngrams = set(ngrams(tokens, n))
        num_distinct_ngrams += len(distinct_ngrams)
    
    return num_distinct_ngrams / num_tokens


def sentence_distinct(sentence, n):
    return corpus_distinct([sentence], n)


pred_file = sys.argv[1]

with open(pred_file, encoding='utf-8') as f:
    preds = [line.strip() for line in f]

print('# micro')
for i in range(2):
    print('distinct-{}: {}'.format(i + 1, corpus_distinct(preds, i + 1)))
print()

print('# macro')
for i in range(2):
    print(
        'distinct-{}: {}'.format(
            i + 1,
            sum(sentence_distinct(pred, i + 1) for pred in preds) / len(preds)
        )
    )
