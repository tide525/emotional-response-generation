import sys

from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score

target_file = sys.argv[1]
pred_file = sys.argv[2]

list_of_references = []
with open(target_file) as f:
    for line in f:
        list_of_references.append([line.strip()])

hypotheses = []
with open(pred_file) as f:
    for line in f:
        hypotheses.append(line.strip())

sum_score = 0.0
for references, hypothesis in zip(list_of_references, hypotheses):
    sum_score += meteor_score(references, hypothesis)
sum_score /= len(list_of_references)

print('METEOR score:', sum_score)
