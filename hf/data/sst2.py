import os
import sys

input_dir = sys.argv[1]
output_dir = sys.argv[2]

input_file = os.path.join(input_dir, 'test.tsv')
with open(input_file) as f:
    test_size = len(f.readlines())

pairs_dict = {}

for split in ['train', 'dev']:
    pairs = []

    input_file = os.path.join(input_dir, split + '.tsv')
    with open(input_file) as f:
        header = next(f)
        for line in f:
            pairs.append(line.strip().split('\t'))
    
    if split == 'train':
        pairs_dict['train'] = pairs[:-test_size]
        pairs_dict['test'] = pairs[-test_size:]
    else:
        pairs_dict['val'] = pairs

# pairs_dict['train'] = pairs_dict['train'][:len(pairs_dict['train'])//4]

for split, pairs in pairs_dict.items():
    for i, lang in enumerate(['source', 'target']):
        output_path = os.path.join(output_dir, split + '.' + lang)
        with open(output_path, 'w') as f:
            for j in range(len(pairs)):
                f.write(pairs[j][i] + '\n')
