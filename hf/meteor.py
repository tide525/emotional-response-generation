import os
import sys

from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score

target_file = sys.argv[1]
pred_file = sys.argv[2]

with open(target_file, encoding='utf-8') as f:
    list_of_references = [[line.strip()] for line in f]

with open(pred_file, encoding='utf-8') as f:
    hypotheses = [line.strip() for line in f]

print('METEOR score:', meteor_score(list_of_references, hypotheses))
