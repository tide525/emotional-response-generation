import os
import random
import sys

cmdc_dir = sys.argv[1]
output_dir = sys.argv[2]

line_file = os.path.join(cmdc_dir, 'movie_lines.txt')
conversation_file = os.path.join(cmdc_dir, 'movie_conversations.txt')

id2text = {}
with open(line_file, 'rb') as f:
    for line in f:
        try:
            row = line.decode().strip().split(' +++$+++ ')
            id2text[row[0]] = row[4]
        except:
            pass

conversations = []
with open(conversation_file, 'rb') as f:
    for line in f:
        try:
            row = line.decode().strip().split(' +++$+++ ')
            conversations.append(row[3][2:-2].split('\', \''))
        except:
            pass

data = []
for lines in conversations:
    for i in range(len(lines) - 1):
        try:
            input_ = id2text[lines[i]]
            target = id2text[lines[i+1]]
            if '\t' not in input_ and '\t' not in target:
                data.append([input_, target])
        except:
            pass

random.seed(0)
random.shuffle(data)

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
