import csv
import os
import random
import sys

random.seed(0)

input_path = sys.argv[1]
output_dir = sys.argv[2]

# anger, boredom, empty, enthusiasm, fun, happiness, hate,
# love, neutral, relief, sadness, surprise, worry

# https://www.aclweb.org/anthology/C18-1179/
label_map = {
    'boredom': 'disgust',
    'empty': None,
    'enthusiasm': 'happiness',
    'fun': 'happiness',
    'hate': 'disgust',
    'love': 'love',
    'neutral': 'no emotion',
    'relief': 'happiness',
    'worry': 'fear'
}

emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

pairs = []

with open(input_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        emotion = row['sentiment']
        if emotion in label_map:
            emotion = label_map[emotion]

        if emotion in emotions:
        # if emotion == 'no emotion' or emotion in emotions:
            text = row['content']
            pairs.append((text, emotion))

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
