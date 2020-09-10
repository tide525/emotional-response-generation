import os
import sys

from nltk.tokenize import word_tokenize
from nltk.translate.nist_score import corpus_nist

target_file = sys.argv[1]
pred_file = sys.argv[2]

list_of_references = []
with open(target_file, encoding='utf-8') as f:
    for line in f:
        list_of_references.append([word_tokenize(line.strip())])

hypotheses = []
with open(pred_file, encoding='utf-8') as f:
    for line in f:
        hypotheses.append(word_tokenize(line.strip()))

for i in range(5):
    print(
        'NIST-{}'.format(i + 1),
        corpus_nist(list_of_references, hypotheses, i + 1)
    )
