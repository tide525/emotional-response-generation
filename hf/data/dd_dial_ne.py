import gzip
import os
import sys

in_dir = sys.argv[1]
out_dir = sys.argv[2]

for split in ['train', 'validation', 'test']:
    dial_path = os.path.join(in_dir, split, 'dial.txt.gz')
    with gzip.open(dial_path) as f:
        dials = [line.decode().rstrip() for line in f]

    emo_path = os.path.join(in_dir, split, 'emo.txt.gz')
    with gzip.open(emo_path) as f:
        emos = [line.decode().rstrip() for line in f]

    pairs = []
    for i in range(0, len(dials), 2):
        if not int(emos[i]):
        # if not int(emos[i]) and not int(emos[i+1]) :
            pairs.append([dials[i], dials[i+1]])

    if split == 'validation':
        split = 'val'

    for i, lang in enumerate(['source', 'target']):
        out_path = os.path.join(out_dir, split + '.' + lang)
        with open(out_path, 'w') as f:
            for j in range(len(pairs)):
                f.write(pairs[j][i] + '\n')
