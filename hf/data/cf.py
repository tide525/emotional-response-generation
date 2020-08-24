import csv
import os
import random
import sys

random.seed(0)

input_path = sys.argv[1]
output_dir = sys.argv[2]

# anger, disgust, fear, joy, sadness, surprise,
# enthusiasm, fun, hate, neutral, love, boredom, relief, empty

label_map = {
    'enthusiasm': 'joy',
    'fun': 'joy',
    'hate': 'disgust',
    'neutral': 'noemo',
    'love': 'love',
    'boredom': 'disgust',
    'relief': 'joy',
    'empty': None
}
emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

pairs = []

with open(input_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        emotion = row['sentiment']
        if emotion in label_map:
            emotion = label_map[emotion]
        
        if emotion is None or emotion in emotions:
            text = row['content']
            pairs.append((text, str(emotion)))

random.shuffle(pairs)

total_size = len(pairs)

val_size = total_size // 20
train_size = total_size - 2 * val_size

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
