import json
import os
import random
import sys

jsonl = sys.argv[1]
output_dir = sys.argv[2]

basic_emotions = ['joy', 'anger', 'sadness', 'disgust', 'fear', 'surprise']

data = []

with open(jsonl, 'r', encoding='utf-8') as f:
    for line in f:
        line_dict = json.loads(line)
        if 'labeled' not in line_dict:
            continue
        if  line_dict['labeled'] != 'single':
            continue

        text = line_dict['text']
        if not text:
            continue
        if '\t' in text:
            continue
        emotions = line_dict['emotions']

        top_emotions = []
        for emotion, value in emotions.items():
            if emotion in basic_emotions and value == 1:
                top_emotions.append(emotion)

        if len(top_emotions) != 1:
            continue
        top_emotion = top_emotions[0]

        data.append([text, top_emotion])

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
