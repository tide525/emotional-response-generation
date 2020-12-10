import sys

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu

target_file = sys.argv[1]
pred_file = sys.argv[2]

list_of_references = []
with open(target_file) as f:
    for line in f:
        list_of_references.append([word_tokenize(line)])

hypotheses = []
with open(pred_file) as f:
    for line in f:
        hypotheses.append(word_tokenize(line))

for i in range(4):
    weights = tuple(float(j < i + 1) / (i + 1) for j in range(4))
    print(
        'BLEU-' + str(i + 1) + ':',
        corpus_bleu(list_of_references, hypotheses, weights=weights)
    )
print()

for i in range(4):
    weights = tuple(float(j == i) for j in range(4))
    print(
        'only ' + str(i + 1) + '-gram:',
        corpus_bleu(list_of_references, hypotheses, weights=weights)
    )
