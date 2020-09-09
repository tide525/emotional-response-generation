import os
import sys

input_path = sys.argv[1]
output_dir = sys.argv[2]

pairs = []
with open(input_path) as f:
    for line in f:
        text, emotion = line.split('\t', maxsplit=1)[1].rsplit('\t', maxsplit=1)
        pairs.append((text, emotion[3:].strip()))

val_size = len(pairs) // 10
train_size = len(pairs) - 2 * val_size

list_of_pairs = [
    pairs[:train_size],
    pairs[train_size:train_size+val_size],
    pairs[train_size+val_size:]
]

for i, split in enumerate(['train', 'val', 'test']):
    for j, lang in enumerate(['source', 'target']):
        output_path = os.path.join(output_dir, split + '.' + lang)
        with open(output_path, 'w') as f:
            for k in range(len(list_of_pairs[i])):
                f.write(list_of_pairs[i][k][j] + '\n')
