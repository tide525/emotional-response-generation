# https://arxiv.org/abs/1809.05972

import math
import sys
from collections import Counter

from nltk.tokenize import word_tokenize
from nltk.util import ngrams


def sentence_ent(sentence, n=4):
    tokens = word_tokenize(sentence)
    F = Counter(ngrams(tokens, n))
    
    sum_F = 0
    sum_FlogF = 0
    for w in F:
        sum_F += F[w]
        sum_FlogF += F[w] * math.log(F[w])
    
    return math.log(sum_F) - sum_FlogF / sum_F


pred_file = sys.argv[1]

with open(pred_file, encoding='utf-8') as f:
    preds = [line.strip() for line in f]

for i in range(4):
    print(
        'Ent-{}: {}'.format(
            i + 1,
            sum(sentence_ent(pred) for pred in preds) / len(preds)
        )
    )
