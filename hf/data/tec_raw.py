import os
import sys

emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

in_path = sys.argv[1]
out_path = os.path.join(sys.argv[2], '{}.{}')

pairs = []
with open(in_path) as f:
    for line in f:
        text, emo = line.split('\t', maxsplit=1)[1].rsplit('\t', maxsplit=1)
        pairs.append((text, emotions.index(emo[3:].rstrip())))

val_size = len(pairs) // 10
train_size = len(pairs) - 2 * val_size

list_of_pairs = [
    pairs[:train_size],
    pairs[train_size:train_size+val_size],
    pairs[train_size+val_size:]
]

for i, split in enumerate(['train', 'val', 'test']):
    for j, lang in enumerate(['source', 'target']):
        with open(out_path.format(split, lang), 'w') as f:
            for k in range(len(list_of_pairs[i])):
                f.write(str(list_of_pairs[i][k][j]) + '\n')
