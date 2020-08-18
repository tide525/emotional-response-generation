import csv
import os
import random
import sys

cf_file = sys.argv[1]
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

with open(cf_file, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        emotion = row['sentiment']
        if emotion in label_map:
            emotion = label_map[emotion]
        
        if emotion in emotions:
            text = row['content']

            pairs.append([text, emotion])

random.seed(0)
random.shuffle(pairs)

total_size = len(pairs)

val_size = total_size // 20
train_size = total_size - 2 * val_size

output_dict = {
    'train': pairs[:train_size],
    'val': pairs[train_size:train_size+val_size],
    'test': pairs[train_size+val_size:],
}

for split, output in output_dict.items():
    output_file = os.path.join(output_dir, split + '.tsv')

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join('\t'.join(pair) for pair in output) + '\n')
