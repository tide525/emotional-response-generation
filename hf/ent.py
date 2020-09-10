# https://arxiv.org/abs/1809.05972

import math
import sys
from collections import Counter

from nltk.tokenize import word_tokenize
from nltk.util import ngrams


def corpus_ent(corpus, n=4):
    tokens = []
    for sentence in corpus:
        tokens.extend(word_tokenize(sentence))
    
    F = Counter(ngrams(tokens, n))

    sum_F = 0
    sum_FlogF = 0
    for w in F:
        sum_F += F[w]
        sum_FlogF += F[w] * math.log(F[w])
    
    return math.log(sum_F) - sum_FlogF / sum_F


def sentence_ent(sentence, n=4):
    return corpus_ent([sentence], n)


pred_file = sys.argv[1]

with open(pred_file, encoding='utf-8') as f:
    preds = [line.strip() for line in f]

for i in range(4):
    print('Ent-' + str(i + 1) + ':', corpus_ent(preds, i + 1))
