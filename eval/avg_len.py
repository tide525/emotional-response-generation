import math
import sys
from collections import Counter

from nltk.tokenize import word_tokenize
from nltk.util import ngrams

pred_path = sys.argv[1]

sum_len = 0
num_preds = 0

with open(sys.argv[1]) as f:
    for line in f:
        tokens = word_tokenize(line.strip())
        sum_len += len(tokens)

        num_preds += 1

print('Avg Len:', sum_len / num_preds)
