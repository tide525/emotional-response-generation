import os
import sys

sst2_dir = sys.argv[1]
output_dir = sys.argv[2]

data = []

input_file = os.path.join(sst2_dir, 'train.tsv')
with open(input_file, 'r', encoding='utf-8') as f:
    header = next(f)
    for line in f:
        sentence, label = line.strip().split('\t')
        sentiment = 'positive' if int(label) else 'negative'

        data.append([sentence, sentiment])

data_len = len(data)

val_size = data_len // 20
train_size = data_len - 2 * val_size

splits = ['train', 'val', 'test']
outputs = [
    data[:train_size],
    data[train_size:train_size+val_size],
    data[train_size+val_size:]
]

for split, output in zip(splits, outputs):
    output_file = os.path.join(output_dir, split + '.tsv')
    with open(output_file, 'w', encoding='utf-8') as f:
        for text, emotion in output:
            f.write('{}\t{}\n'.format(text, emotion))
