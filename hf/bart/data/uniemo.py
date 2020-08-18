import json
import os
import random
import sys

input_file = sys.argv[1]
output_dir = sys.argv[2]

ekman_emotions = ['joy', 'anger', 'sadness', 'disgust', 'fear', 'surprise']

pairs = []
with open(input_file, encoding='utf-8') as f:
    for line in f:
        line_dict = json.loads(line)

        text = line_dict['text']
        if not text or '\t' in text:
            continue
        emotion_dict = line_dict['emotions']

        top_emotions = []
        for ekman_emotion in ekman_emotions:
            if emotion_dict[ekman_emotion] == 1:
                top_emotions.append(ekman_emotion)

        if len(top_emotions) != 1:
            continue
        label = ekman_emotions.index(top_emotions[0])

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
            f.write(text + '\t' + str(label) + '\n')
