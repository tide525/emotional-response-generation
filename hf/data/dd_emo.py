import gzip
import os
import sys

input_dir = sys.argv[1]
output_dir = sys.argv[2]

for split in ['train', 'validation', 'test']:
    pairs = []

    dial_path = os.path.join(input_dir, split, 'dial.txt.gz')
    emo_path = os.path.join(input_dir, split, 'emo.txt.gz')
    with gzip.open(dial_path) as dial_f, gzip.open(emo_path) as emo_f:
        for dial_line, emo_line in zip(dial_f, emo_f):
            dial = dial_line.decode().strip()
            emo = emo_line.decode().strip()
            if emo != '0':
                pairs.append((dial, emo))

    pairs = sorted(set(pairs), key=pairs.index)

    if split == 'validation':
        split = 'val'

    for i, lang in enumerate(['source', 'target']):
        output_path = os.path.join(output_dir, split + '.' + lang)
        with open(output_path, 'w') as f:
            for j in range(len(pairs)):
                f.write(pairs[j][i] + '\n')
