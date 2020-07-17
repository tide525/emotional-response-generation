import json
import os
import random
import sys
from collections import Counter

jsonl = sys.argv[1]
output_dir = sys.argv[2]

basic_emotions = ['joy', 'anger', 'sadness', 'disgust', 'fear', 'surprise']

data = []
emotion_counter = Counter()

with open(jsonl, 'r', encoding='utf-8') as f:
    for line in f:
        line_dict = json.loads(line)
        if 'labeled' not in line_dict:
            # print('\'labeled\' not in dict', file=sys.stderr)
            continue
        if line_dict['labeled'] != 'single':
            # print('not \'single\'', file=sys.stderr)
            continue

        text = line_dict['text']
        if not text:
            # print('\'text\' not in dict', file=sys.stderr)
            continue
        emotions = line_dict['emotions']

        top_emotions = []
        for emotion, value in emotions.items():
            if emotion in basic_emotions and value == 1:
                top_emotions.append(emotion)

        if len(top_emotions) != 1:
            # print('too many or no emotions', file=sys.stderr)
            continue
        top_emotion = top_emotions[0]

        data.append([text, top_emotion])
        emotion_counter[top_emotion] += 1

counter_min = min(emotion_counter.values())

random.seed(0)
random.shuffle(data)

balanced_data = []
data_counter = Counter()

for text, emotion in data:
    if data_counter[emotion] < counter_min:
        balanced_data.append([text, emotion])
        data_counter[emotion] += 1

random.shuffle(balanced_data)
data_len = len(balanced_data)

val_size = data_len // 20
train_size = data_len - 2 * val_size

splits = ['train', 'val', 'test']
outputs = [
    balanced_data[:train_size],
    balanced_data[train_size:train_size+val_size],
    balanced_data[train_size+val_size:]
]

for split, output in zip(splits, outputs):
    output_file = os.path.join(output_dir, split + '.tsv')
    with open(output_file, 'w', encoding='utf-8') as f:
        for text, emotion in output:
            f.write('{}\t{}\n'.format(text, emotion))
