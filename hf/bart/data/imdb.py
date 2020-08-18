import glob
import os
import random
import sys

input_dir = sys.argv[1]
output_dir = sys.argv[2]

pairs = []

splits = ['train', 'test']
sentiments = ['neg', 'pos']

for split in splits:
    for label, sentiment in enumerate(sentiments):
        input_files = glob.glob(
            '/'.join([input_dir, split, sentiment, '*.txt'])
        )

        for input_file in input_files:
            with open(input_file, encoding='utf-8') as f:
                text = f.read()

            pairs.append([text, label])

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
        for text, label in pairs:
            f.write('{}\t{}\n'.format(text, label))
