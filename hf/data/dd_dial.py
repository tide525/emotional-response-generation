import gzip
import os
import sys

input_dir = sys.argv[1]
output_dir = sys.argv[2]

for split in ['train', 'validation', 'test']:
    pairs = []
    
    input_path = os.path.join(input_dir, split, 'dial.txt.gz')
    with gzip.open(input_path) as f:
        pair = []
        for line in f:
            pair.append(line.decode().strip())
            if len(pair) == 2:
                pairs.append(pair)
                pair = []

    if split == 'validation':
        split = 'val'

    for i, lang in enumerate(['source', 'target']):
        output_path = os.path.join(output_dir, split + '.' + lang)
        with open(output_path, 'w') as f:
            for j in range(len(pairs)):
                f.write(pairs[j][i] + '\n')
