import os
import sys

from nltk.tokenize import word_tokenize
from nltk.translate.nist_score import corpus_nist

target_file = sys.argv[1]
pred_file = sys.argv[2]

with open(target_file, encoding='utf-8') as f:
    list_of_references = [[line.strip()] for line in f]

with open(pred_file, encoding='utf-8') as f:
    hypotheses = [line.strip() for line in f]

print('NIST score:', corpus_nist(list_of_references, hypotheses))
print()

for i in range(4):
    print(
        'NIST-{}: {}'.format(
            i + 1,
            corpus_nist(list_of_references, hypotheses, i + 1)
        )
    )
