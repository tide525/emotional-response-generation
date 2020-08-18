import os
import random
import sys

input_dir = sys.argv[1]
output_dir = sys.argv[2]

pairs = []
for split in ['train', 'dev']:
    input_file = os.path.join(input_dir, split + '.tsv')
    with open(input_file, encoding='utf-8') as f:
        header = next(f)
        for line in f:
            sentence, label = line.strip().split('\t')
            pairs.append([sentence, int(label)])

random.seed(0)
random.shuffle(pairs)

pairs_len = len(pairs)
val_size = pairs_len // 20
train_size = pairs_len - 2 * val_size

splits = ['train', 'val', 'test']
list_of_pairs = [
    pairs[:train_size],
    pairs[train_size:train_size+val_size],
    pairs[train_size+val_size:]
]

for split, pairs in zip(splits, list_of_pairs):
    output_file = os.path.join(output_dir, split + '.tsv')
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence, label in pairs:
            f.write(sentence + '\t' + str(label) + '\n')
