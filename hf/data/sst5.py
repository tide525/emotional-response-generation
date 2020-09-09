import os
import sys

in_path = os.path.join(sys.argv[1], 'sst_{}.txt')
out_path = os.path.join(sys.argv[2], '{}.{}')

for category in ['train', 'test', 'dev']:
    pairs = []
    
    with open(in_path.format(category)) as f:
        pair = []
        for line in f:
            label, sentence = line.strip().split('\t')
            pairs.append((sentence, (label[-1])))

    split = 'val' if category == 'dev' else category

    for i, lang in enumerate(['source', 'target']):
        with open(out_path.format(split, lang), 'w') as f:
            for j in range(len(pairs)):
                f.write(pairs[j][i] + '\n')
