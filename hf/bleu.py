# https://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.word_tokenize
# https://www.nltk.org/api/nltk.translate.html#nltk.translate.bleu_score.corpus_bleu

import os
import sys

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu

target_file = sys.argv[1]
pred_file = sys.argv[2]

with open(target_file, encoding='utf-8') as f:
    list_of_references = [[line.strip()] for line in f]

with open(pred_file, encoding='utf-8') as f:
    hypotheses = [line.strip() for line in f]

print(
    'BLEU score:',
    corpus_bleu(list_of_references, hypotheses)
)
print()

for i in range(4):
    weights = tuple(float(j == i) for j in range(4))
    print(
        str(i + 1) + '-gram:',
        corpus_bleu(list_of_references, hypotheses, weights=weights)
    )
