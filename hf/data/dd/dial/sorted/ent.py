# https://arxiv.org/abs/1809.05972

import math
import os
import sys
from collections import Counter

from nltk.tokenize import word_tokenize

in_path = os.path.join(sys.argv[1], '{0}', 'dialogues_{0}.txt')
out_path = os.path.join(sys.argv[2], '{0}.{1}')

for split in ['train', 'validation', 'test']:
    pairs = []
    F = Counter()

    with open(in_path.format(split)) as f:
        for line in f:
            texts = line.split('__eou__')[:-1]

            for i in range(len(texts) - 1):
                pairs.append([texts[i].strip(), texts[i+1].strip()])
                for token in word_tokenize(texts[i]):
                    F[token] += 1

    sum_F = sum(freq for freq in F.values())

    ent_key = lambda pair: (
        sum(
            -F[token] * math.log (F[token] / sum_F)
            for token in word_tokenize(pair[0])
        )
        / sum_F
    )
    pairs.sort(key=ent_key)

    if split == 'validation':
        split = 'val'

    for i, lang in enumerate(['source', 'target']):
        with open(out_path.format(split, lang), 'w') as f:
            for j in range(len(pairs)):
                f.write(pairs[j][i] + '\n')
