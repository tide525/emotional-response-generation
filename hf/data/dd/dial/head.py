import os
import sys

in_dir = sys.argv[1]
out_dir = sys.argv[2]

for split in ['train', 'validation', 'test']:
    in_path = os.path.join(in_dir, split, 'dialogues_' + split + '.txt')

    pairs = []
    with open(in_path) as f:
        for line in f:
            pair = []
            for u in line.rstrip().split('__eou__')[:2]:
                pair.append(u.strip())
            pairs.append(pair)
    
    if split == 'validation':
        split = 'val'

    for i, lang in enumerate(['source', 'target']):
        out_path = os.path.join(out_dir, split + '.' + lang)
        with open(out_path, 'w') as f:
            for j in range(len(pairs)):
                f.write(pairs[j][i] + '\n')
