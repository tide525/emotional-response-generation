# https://arxiv.org/abs/1510.03055

import sys

from nltk.tokenize import word_tokenize
from nltk.util import ngrams


'''def corpus_dist(corpus, n):
    num_tokens = 0
    num_distinct_ngrams = 0

    for sentence in corpus:
        tokens = word_tokenize(sentence)
        num_tokens += len(tokens)

        distinct_ngrams = set(ngrams(tokens, n))
        num_distinct_ngrams += len(distinct_ngrams)
    
    return num_distinct_ngrams / num_tokens
'''


def corpus_dist(corpus, n):
    ngrams_set = set()
    num_tokens = 0

    for sentence in corpus:
        tokens = word_tokenize(sentence)
        num_tokens += len(tokens)

        ngrams_set |= set(ngrams(tokens, n))
            
    return len(ngrams_set) / num_tokens


def sentence_dist(sentence, n):
    return corpus_dist([sentence], n)


pred_file = sys.argv[1]

with open(pred_file) as f:
    preds = [line.strip() for line in f]

for i in range(2):
    print('Dist-' + str(i + 1) + ':', corpus_dist(preds, i + 1))
